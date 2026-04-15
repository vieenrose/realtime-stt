# Voxtral Mini Realtime STT Service

Real-time speech-to-text transcription service using Voxtral Mini 4B Realtime 2602 model with CUDA GPU acceleration.

## Overview

This project provides a FastAPI-based transcription service optimized for NVIDIA DGX Spark (Grace-Blackwell with 128GB unified memory). The model supports:

- **13 languages**: en, fr, es, de, ru, zh, ja, it, pt, nl, ar, hi, ko
- **Real-time transcription**: ~480ms latency at Whisper-quality accuracy
- **Mixed language support**: Handles zh-TW/English code-switching

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  HTTP Client    │────▶│  FastAPI Server  │────▶│  JSON Response  │
│  (Audio Upload) │     │                  │     │  (Transcript)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌──────────────────┐
                        │  Voxtral Model   │
                        │  (HuggingFace    │
                        │   Transformers)  │
                        └──────────────────┘
```

## Project Structure

```
voxtral-stt/
├── CLAUDE.md           # Project guidance for Claude Code
├── Dockerfile          # CUDA-enabled container
├── requirements.txt    # Python dependencies
├── voxtral_onnx.py     # Voxtral model wrapper (transformers-based)
├── server.py           # FastAPI server with endpoints
├── benchmark.py        # Performance measurement script
├── scripts/
│   └── download_samples.py  # TTS test sample generator
├── test_samples/       # Directory for test audio files
└── model/              # Voxtral ONNX model (optional, for ONNX variant)
```

## Installation

### Local Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install mistral-common (required for Voxtral)
pip install mistral-common
```

### Model Usage

The service uses HuggingFace Transformers to load the Voxtral model automatically:

```bash
# Default: downloads model from HuggingFace
python server.py

# Or use local ONNX model
export VOXTRAL_USE_ONNX=true
export VOXTRAL_ONNX_PATH=./model
python server.py
```

### Docker Build

```bash
docker build -t voxtral-stt .
docker run --gpus all -p 8000:8000 voxtral-stt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transcribe` | POST | Upload audio file for transcription |
| `/stream` | WebSocket | Real-time streaming transcription |
| `/transcribe_bytes` | POST | Transcribe raw audio bytes |
| `/health` | GET | Health check with GPU status |
| `/info` | GET | Model information |
| `/` | GET | API overview |

## Usage Examples

### File Transcription

```bash
curl -X POST -F "file=@audio.wav" http://localhost:8000/transcribe
```

Response:
```json
{
  "text": "Hello, this is a test.",
  "language": "en",
  "confidence": 0.95,
  "audio_duration": 5.2
}
```

### Streaming Transcription (WebSocket)

```python
import websocket
import numpy as np

ws = websocket.create_connection("ws://localhost:8000/stream")

# Send audio chunks (16kHz mono float32)
audio_chunk = np.random.randn(7680).astype(np.float32)  # 480ms
ws.send_binary(audio_chunk.tobytes())

# Receive transcription
result = ws.recv()
print(result)  # {"text": "...", "is_final": false, ...}
```

### Generate Test Samples

```bash
pip install edge-tts
python scripts/download_samples.py --generate
```

## Benchmarking

```bash
python benchmark.py --generate-test --iterations 10 --output results.json
```

Metrics tracked:
- Latency (first-packet and total)
- Real-time factor (RTF)
- GPU memory usage
- Throughput

## Model Details

- **Model**: [Voxtral-Mini-4B-Realtime-2602-ONNX](https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX)
- **Parameters**: ~4B (3.4B LLM + 970M audio encoder)
- **Architecture**: Natively streaming ASR with causal audio encoder
- **Latency**: 480ms default (configurable: 80ms-2400ms)
- **License**: Apache 2.0

## Technical Notes

### Audio Format
- Input: 16kHz mono PCM (standard for ASR models)
- Features: 128 mel bins
- Chunk size: 480ms = 7680 samples for streaming

### Model Configuration
```python
from voxtral_onnx import VoxtralRealtime

# Load from HuggingFace
model = VoxtralRealtime(
    model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
    device="cuda:0",
    transcription_delay_ms=480  # Sweet spot for accuracy/latency
)

# Or use local ONNX model
model = VoxtralRealtime(
    use_onnx=True,
    onnx_model_path="./model",
    device="cuda:0"
)
```

### DGX Spark Optimizations
- Unified memory architecture eliminates GPU-CPU transfer overhead
- 128GB memory allows full model loading without memory pressure
- Use bfloat16 for optimal GPU inference
- Set temperature=0.0 for best transcription quality

## Environment Variables

- `VOXTRAL_MODEL_ID`: HuggingFace model ID (default: mistralai/Voxtral-Mini-4B-Realtime-2602)
- `VOXTRAL_DEVICE`: Device for inference (default: cuda:0)
- `VOXTRAL_USE_ONNX`: Use local ONNX model (default: false)
- `VOXTRAL_ONNX_PATH`: Path to ONNX model directory (default: ./model)

## References

- [Voxtral Paper](https://arxiv.org/abs/2602.11298)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [HuggingFace Model Card](https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX)

## Status

- [x] FastAPI server implementation
- [x] Transformers-based model wrapper
- [x] Docker configuration
- [x] Benchmark script
- [x] Test sample generator
- [x] Model working with transformers library
- [x] GPU acceleration working (NVIDIA GB10)
- [x] Streaming transcription working
- [x] Chinese-English code-switching validated
- [ ] Traditional Chinese (zh-TW) output (model outputs Simplified)

## GPU Benchmark Results (NVIDIA GB10 Grace-Blackwell)

| Test | Audio | Latency | RTF | Result |
|------|-------|---------|-----|--------|
| English (20s) | Obama speech | 16s | 0.799 | ✓ Faster than realtime |
| Chinese (5s) | TTS sample | 4.8s | 0.96 | ✓ Faster than realtime |
| Mixed (7s) | zh+EN code-switch | 5.7s | 0.79 | ✓ Handles both languages |
| Streaming (30s) | Live transcription | 24s | 0.81 | ✓ First text at ~1.9s |

### Chinese-English Code-Switching Test

**Input (Traditional Chinese + English):**
> 我喜歡用 Python programming language 來開發 software applications。Today is a good day for coding。

**Output (Simplified Chinese + English):**
> 我喜欢用Python programming language来开发software applicationsToday is a good day for coding

Note: Model outputs Simplified Chinese characters (zh) instead of Traditional (zh-TW). This is expected as Voxtral supports "zh" language.