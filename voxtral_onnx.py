"""
Voxtral Mini Realtime Model Wrapper

This module provides a wrapper for running Voxtral Mini Realtime 2602 model
for real-time speech-to-text transcription using HuggingFace Transformers.

The model architecture consists of:
1. Audio encoder (~970M params) - causal audio encoder for streaming
2. Decoder (~3.4B params) - Mistral-based language model for text generation

Key features:
- Natively streaming architecture with configurable delay (80ms-2400ms)
- Supports 13 languages including Chinese (zh) and English
- Uses KV cache for efficient incremental inference
"""

import os
import numpy as np
from typing import Optional, Dict, Any, AsyncIterable, Generator
from pathlib import Path
from threading import Thread

try:
    import torch
    from transformers import (
        VoxtralRealtimeProcessor,
        VoxtralRealtimeForConditionalGeneration,
        TextIteratorStreamer,
    )
except ImportError:
    raise ImportError(
        "Please install transformers and torch: "
        "pip install transformers accelerate torch"
    )


# Audio constants from model config
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
HOP_LENGTH = 160  # 10ms at 16kHz
WIN_LENGTH = 400  # 25ms at 16kHz
DEFAULT_DELAY_MS = 480  # Sweet spot for accuracy/latency


class VoxtralRealtime:
    """Voxtral Mini Realtime model wrapper for streaming transcription."""

    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602",
        device: str = "cuda:0",
        transcription_delay_ms: int = 480,
        use_onnx: bool = False,
        onnx_model_path: Optional[str] = None
    ):
        """
        Initialize Voxtral Realtime model.

        Args:
            model_id: HuggingFace model ID
            device: Device to run on (cuda:0, cpu, etc.)
            transcription_delay_ms: Delay in ms (80-2400, multiples of 80)
            use_onnx: Whether to use ONNX model variant
            onnx_model_path: Path to ONNX model directory (if use_onnx=True)
        """
        self.model_id = model_id
        self.device = device
        self.transcription_delay_ms = transcription_delay_ms

        if use_onnx and onnx_model_path:
            self.model_id = onnx_model_path

        # Load processor and model
        self._load_model()

        # Streaming state
        self._reset_state()

    def _load_model(self):
        """Load processor and model."""
        print(f"Loading model from: {self.model_id}")

        self.processor = VoxtralRealtimeProcessor.from_pretrained(self.model_id)

        # Determine dtype based on device
        if self.device.startswith("cuda"):
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=dtype
        )

        print(f"Model loaded on {self.device}")

    def _reset_state(self):
        """Reset streaming state."""
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._is_first_chunk = True

    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio to model's expected format.

        Args:
            audio: Input audio waveform
            sample_rate: Input sample rate

        Returns:
            Preprocessed audio at 16kHz
        """
        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            except ImportError:
                # Simple linear resampling fallback
                ratio = SAMPLE_RATE / sample_rate
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio), new_length),
                    np.arange(len(audio)),
                    audio
                ).astype(np.float32)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        return audio

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        max_new_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text (non-streaming).

        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate
            language: Target language hint (auto-detected if None)
            max_new_tokens: Maximum tokens to generate

        Returns:
            dict with 'text', 'language', 'confidence', 'audio_duration'
        """
        # Preprocess audio
        audio = self._preprocess_audio(audio, sample_rate)
        audio_duration = len(audio) / SAMPLE_RATE

        # Prepare inputs
        inputs = self.processor(
            audio=audio,
            is_streaming=False,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0  # Recommended for transcription
            )

        # Decode
        # Skip the input tokens, only decode generated tokens
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Detect language from text
        detected_lang = self._detect_language(text) if language is None else language

        return {
            'text': text,
            'language': detected_lang,
            'confidence': 0.95,
            'audio_duration': audio_duration
        }

    def transcribe_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_callback: Optional[callable] = None
    ) -> str:
        """
        Stream transcription of audio with incremental text output.

        This uses the streaming architecture with KV cache for efficient
        incremental inference.

        Args:
            audio: Full audio waveform
            sample_rate: Audio sample rate
            chunk_callback: Optional callback for each text chunk

        Returns:
            Full transcribed text
        """
        # Preprocess audio
        audio = self._preprocess_audio(audio, sample_rate)

        # Pad audio for proper chunking (right padding)
        num_right_pad_tokens = self.processor.num_right_pad_tokens
        raw_audio_length_per_tok = self.processor.raw_audio_length_per_tok
        audio = np.pad(audio, (0, num_right_pad_tokens * raw_audio_length_per_tok))

        # Prepare first chunk inputs
        num_samples_first = self.processor.num_samples_first_audio_chunk
        first_chunk_inputs = self.processor(
            audio[:num_samples_first],
            is_streaming=True,
            is_first_audio_chunk=True,
            return_tensors="pt"
        )
        first_chunk_inputs = first_chunk_inputs.to(
            self.model.device, dtype=self.model.dtype
        )

        # Create input features generator
        def input_features_generator():
            yield first_chunk_inputs.input_features

            mel_frame_idx = self.processor.num_mel_frames_first_audio_chunk
            hop_length = self.processor.feature_extractor.hop_length
            win_length = self.processor.feature_extractor.win_length

            start_idx = mel_frame_idx * hop_length - win_length // 2
            num_samples_per_chunk = self.processor.num_samples_per_audio_chunk

            while (end_idx := start_idx + num_samples_per_chunk) < audio.shape[0]:
                chunk = audio[start_idx:end_idx]
                inputs = self.processor(
                    chunk,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.model.device, dtype=self.model.dtype)
                yield inputs.input_features

                mel_frame_idx += self.processor.audio_length_per_tok
                start_idx = mel_frame_idx * hop_length - win_length // 2

        # Create streamer for text output
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Run generation in thread
        generate_kwargs = {
            "input_ids": first_chunk_inputs.input_ids,
            "input_features": input_features_generator(),
            "num_delay_tokens": first_chunk_inputs.num_delay_tokens,
            "streamer": streamer,
            "temperature": 0.0,
        }

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        # Collect streaming output
        full_text = ""
        for text_chunk in streamer:
            if chunk_callback:
                chunk_callback(text_chunk)
            full_text += text_chunk
            print(text_chunk, end="", flush=True)  # Live output

        thread.join()
        return full_text

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterable[np.ndarray],
        chunk_size_ms: int = DEFAULT_DELAY_MS
    ) -> AsyncIterable[Dict[str, Any]]:
        """
        Async streaming transcription for real-time use.

        Args:
            audio_chunks: Async iterable of audio chunks
            chunk_size_ms: Size of chunks for inference

        Yields:
            Partial transcription results with 'text', 'is_final', 'language'
        """
        chunk_samples = int(SAMPLE_RATE * chunk_size_ms / 1000)
        buffer = []

        async for chunk in audio_chunks:
            buffer.extend(chunk)

            while len(buffer) >= chunk_samples:
                audio_chunk = np.array(buffer[:chunk_samples], dtype=np.float32)
                buffer = buffer[chunk_samples:]

                result = self.transcribe(audio_chunk, SAMPLE_RATE)
                result['is_final'] = False
                yield result

        # Process remaining buffer
        if len(buffer) > 0:
            audio_chunk = np.array(buffer, dtype=np.float32)
            result = self.transcribe(audio_chunk, SAMPLE_RATE)
            result['is_final'] = True
            yield result

    def reset_stream(self):
        """Reset streaming state for new utterance."""
        self._reset_state()

    def _detect_language(self, text: str) -> str:
        """Detect language from text content."""
        # Check for Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return 'zh'
        return 'en'

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            'model_name': 'Voxtral-Mini-4B-Realtime-2602',
            'model_id': self.model_id,
            'languages': ['en', 'fr', 'es', 'de', 'ru', 'zh', 'ja', 'it', 'pt', 'nl', 'ar', 'hi', 'ko'],
            'sample_rate': SAMPLE_RATE,
            'transcription_delay_ms': self.transcription_delay_ms,
            'device': self.device,
            'processor_config': {
                'num_mel_bins': self.processor.feature_extractor.feature_size if hasattr(self.processor.feature_extractor, 'feature_size') else NUM_MEL_BINS,
                'hop_length': self.processor.feature_extractor.hop_length if hasattr(self.processor.feature_extractor, 'hop_length') else HOP_LENGTH,
            }
        }


# Alias for backwards compatibility
VoxtralONNX = VoxtralRealtime


def transcribe_file(
    audio_path: str,
    model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602",
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Transcribe an audio file.

    Args:
        audio_path: Path to audio file
        model_id: Model ID to use
        device: Device for inference

    Returns:
        Transcription result dict
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile: pip install soundfile")

    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    model = VoxtralRealtime(model_id=model_id, device=device)
    return model.transcribe(audio, sr)