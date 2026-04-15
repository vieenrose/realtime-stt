"""
Voxtral Mini Realtime ONNX Model Wrapper

This module provides a wrapper for running Voxtral Mini Realtime 2602 ONNX
model with CUDA acceleration for real-time speech-to-text transcription.

The model architecture consists of:
1. Audio encoder (~970M params) - processes 16kHz audio into features
2. Decoder (~3.4B params) - generates text tokens from audio features

Key parameters from config:
- num_mel_bins: 128 (audio feature dimension)
- audio_length_per_tok: 8 (each token = 80ms audio)
- default_num_delay_tokens: 6 (480ms delay)
- vocab_size: 131072 (Mistral tokenizer)
- bos_token_id: 1, eos_token_id: 2
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterable

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Please install onnxruntime or onnxruntime-gpu: pip install onnxruntime-gpu")


# Audio constants
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
HOP_LENGTH = 160  # 10ms at 16kHz
WIN_LENGTH = 400  # 25ms at 16kHz
CHUNK_SIZE_MS = 480  # Default transcription delay
CHUNK_SIZE_SAMPLES = int(SAMPLE_RATE * CHUNK_SIZE_MS / 1000)  # 7680 samples


class VoxtralONNX:
    """Voxtral Mini Realtime ONNX wrapper with CUDA acceleration."""

    def __init__(
        self,
        model_path: str = "./model",
        use_quantized: bool = False,
        use_fp16: bool = True,
        device_id: int = 0,
        transcription_delay_ms: int = 480
    ):
        """
        Initialize Voxtral ONNX model.

        Args:
            model_path: Path to the model directory containing ONNX files
            use_quantized: Use quantized model variant for lower memory
            use_fp16: Use FP16 model variant for faster inference
            device_id: CUDA device ID (0 by default)
            transcription_delay_ms: Transcription delay in ms (80-2400, multiples of 80)
        """
        self.model_path = Path(model_path)
        self.transcription_delay_ms = transcription_delay_ms

        # Load config
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Select model variant based on options
        onnx_dir = self.model_path / "onnx"

        # Audio encoder selection
        if use_quantized:
            encoder_name = "audio_encoder_q4f16.onnx" if use_fp16 else "audio_encoder_q4.onnx"
        elif use_fp16:
            encoder_name = "audio_encoder_fp16.onnx"
        else:
            encoder_name = "audio_encoder.onnx"

        # Decoder selection (use merged model for end-to-end)
        if use_quantized:
            decoder_name = "decoder_model_merged_q4f16.onnx" if use_fp16 else "decoder_model_merged_q4.onnx"
        elif use_fp16:
            decoder_name = "decoder_model_merged_fp16.onnx"
        else:
            decoder_name = "decoder_model_merged.onnx"

        encoder_path = onnx_dir / encoder_name
        decoder_path = onnx_dir / decoder_name

        # Fallback to base models if variants don't exist
        if not encoder_path.exists():
            encoder_path = onnx_dir / "audio_encoder.onnx"
        if not decoder_path.exists():
            decoder_path = onnx_dir / "decoder_model_merged.onnx"

        # Configure execution providers
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit per session
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]

        # Create sessions
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.encoder_session = ort.InferenceSession(
            str(encoder_path),
            sess_options=sess_options,
            providers=providers
        )

        # Note: decoder_model_merged combines encoder projection + decoder
        # For standalone transcription, we may only need this model
        self.decoder_session = ort.InferenceSession(
            str(decoder_path),
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output info from sessions
        self._input_names = self._get_input_info()
        self._output_names = self._get_output_info()

        # Initialize tokenizer (we'll need to handle token decoding)
        self._init_tokenizer()

        # State for streaming
        self._buffer = np.zeros(0, dtype=np.float32)
        self._generated_tokens = []

    def _get_input_info(self) -> Dict[str, Any]:
        """Extract input tensor names and shapes from encoder session."""
        inputs = {}
        for inp in self.encoder_session.get_inputs():
            inputs[inp.name] = {
                'shape': inp.shape,
                'type': inp.type
            }
        return inputs

    def _get_output_info(self) -> Dict[str, Any]:
        """Extract output tensor names and shapes from encoder session."""
        outputs = {}
        for out in self.encoder_session.get_outputs():
            outputs[out.name] = {
                'shape': out.shape,
                'type': out.type
            }
        return outputs

    def _init_tokenizer(self):
        """Initialize tokenizer for decoding output tokens."""
        # Voxtral uses Mistral tokenizer (vocab_size: 131072)
        # For ONNX models, tokenizer is usually embedded or needs external vocab
        # We'll need to find tekken.json or similar in the model directory

        tokenizer_path = self.model_path / "tekken.json"
        if tokenizer_path.exists():
            with open(tokenizer_path) as f:
                self.tokenizer_config = json.load(f)
        else:
            # Fallback: tokenizer info from generation_config
            self.tokenizer_config = {
                'bos_token_id': 1,
                'eos_token_id': 2,
                'pad_token_id': 11
            }

    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio to model's expected format.

        Args:
            audio: Input audio waveform (float32)
            sample_rate: Input sample rate

        Returns:
            Preprocessed audio features (mel spectrogram or raw waveform)
        """
        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            # Simple linear resampling (for production, use librosa.resample)
            ratio = SAMPLE_RATE / sample_rate
            new_length = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)

        # Normalize audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio

    def _compute_mel_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram features from audio.

        This matches Voxtral's audio encoder preprocessing.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("Please install librosa: pip install librosa")

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=NUM_MEL_BINS,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            fmin=0,
            fmax=SAMPLE_RATE // 2
        )

        # Convert to log scale
        mel = librosa.power_to_db(mel, ref=np.max)

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        return mel.astype(np.float32)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        return_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio waveform (float32, any sample rate)
            sample_rate: Audio sample rate (will resample if not 16kHz)
            language: Target language hint (auto-detected if None)
            return_timestamps: Whether to return word timestamps

        Returns:
            dict with 'text', 'language', 'confidence', and optionally 'timestamps'
        """
        # Preprocess audio
        audio = self._preprocess_audio(audio, sample_rate)

        # Compute mel features
        mel_features = self._compute_mel_features(audio)

        # Prepare inputs for ONNX inference
        # The exact input names depend on the model structure
        # Typical inputs: input_features, attention_mask, etc.

        # Add batch dimension
        mel_features = mel_features[np.newaxis, ...]  # [1, n_mels, time]

        # Transpose to expected format [batch, time, features] or [batch, features, time]
        # Voxtral expects [batch, time, features]
        mel_features = mel_features.transpose(0, 2, 1)

        # Create attention mask (all ones for valid audio)
        attention_mask = np.ones(mel_features.shape[:2], dtype=np.int64)

        # Run encoder
        try:
            encoder_inputs = {
                'input_features': mel_features,
                'attention_mask': attention_mask
            }
            encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        except Exception as e:
            # Try alternative input names
            # Some ONNX exports use different naming
            input_name = self.encoder_session.get_inputs()[0].name
            encoder_inputs = {input_name: mel_features}
            encoder_outputs = self.encoder_session.run(None, encoder_inputs)

        # Encoder outputs contain audio embeddings
        audio_embeds = encoder_outputs[0]

        # Run decoder with audio embeddings
        # The merged decoder handles the full transcription loop
        decoder_inputs = {
            'audio_embeds': audio_embeds,
            'attention_mask': attention_mask,
            'max_length': np.array([512], dtype=np.int64),
            'min_length': np.array([1], dtype=np.int64),
        }

        try:
            decoder_outputs = self.decoder_session.run(None, decoder_inputs)
        except Exception as e:
            # Simplified inference with just audio features
            decoder_input_name = self.decoder_session.get_inputs()[0].name
            decoder_outputs = self.decoder_session.run(
                None,
                {decoder_input_name: audio_embeds}
            )

        # Decode output tokens to text
        # Output format: [batch, sequence_length] containing token IDs
        output_tokens = decoder_outputs[0]

        # Decode tokens
        text = self._decode_tokens(output_tokens[0])

        # Detect language (simplified - actual implementation would use model output)
        detected_lang = self._detect_language(text) if language is None else language

        return {
            'text': text,
            'language': detected_lang,
            'confidence': 0.95,  # Placeholder
        }

    def _decode_tokens(self, tokens: np.ndarray) -> str:
        """
        Decode token IDs to text string.

        Args:
            tokens: Array of token IDs

        Returns:
            Decoded text string
        """
        # Find EOS token and truncate
        eos_id = self.tokenizer_config.get('eos_token_id', 2)
        eos_positions = np.where(tokens == eos_id)[0]
        if len(eos_positions) > 0:
            tokens = tokens[:eos_positions[0]]

        # Remove BOS token
        bos_id = self.tokenizer_config.get('bos_token_id', 1)
        tokens = tokens[tokens != bos_id]

        # Remove padding
        pad_id = self.tokenizer_config.get('pad_token_id', 11)
        tokens = tokens[tokens != pad_id]

        # For full implementation, we need the actual vocabulary mapping
        # This requires tekken.json or similar tokenizer file
        # For now, return placeholder (actual decoding needs vocab lookup)

        if len(tokens) == 0:
            return ""

        # Placeholder: indicate that token decoding requires vocab
        return f"[tokens: {len(tokens)}]"

    def _detect_language(self, text: str) -> str:
        """Detect language from text content."""
        # Simplified language detection
        # Check for Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return 'zh'
        return 'en'

    def transcribe_stream(
        self,
        audio_chunks: AsyncIterable[np.ndarray],
        chunk_size_ms: int = CHUNK_SIZE_MS
    ) -> AsyncIterable[Dict[str, Any]]:
        """
        Streaming transcription for real-time use.

        Args:
            audio_chunks: Async iterable of audio chunks
            chunk_size_ms: Size of chunks for inference (default 480ms)

        Yields:
            Partial transcription results
        """
        chunk_samples = int(SAMPLE_RATE * chunk_size_ms / 1000)
        buffer = []

        async for chunk in audio_chunks:
            buffer.extend(chunk)

            # Process when buffer reaches chunk size
            while len(buffer) >= chunk_samples:
                audio_chunk = np.array(buffer[:chunk_samples], dtype=np.float32)
                buffer = buffer[chunk_samples:]

                # Transcribe chunk
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
        self._buffer = np.zeros(0, dtype=np.float32)
        self._generated_tokens = []

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        return {
            'model_name': 'Voxtral-Mini-4B-Realtime-2602-ONNX',
            'languages': ['en', 'fr', 'es', 'de', 'ru', 'zh', 'ja', 'it', 'pt', 'nl', 'ar', 'hi', 'ko'],
            'sample_rate': SAMPLE_RATE,
            'transcription_delay_ms': self.transcription_delay_ms,
            'encoder_inputs': self._input_names,
            'encoder_outputs': self._output_names,
            'providers': self.encoder_session.get_providers(),
        }


# Convenience function for quick transcription
def transcribe_file(audio_path: str, model_path: str = "./model") -> Dict[str, Any]:
    """
    Transcribe an audio file.

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        model_path: Path to ONNX model directory

    Returns:
        Transcription result dict
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile: pip install soundfile")

    # Load audio
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Initialize model
    model = VoxtralONNX(model_path)

    # Transcribe
    return model.transcribe(audio, sr)