# Voxtral Mini Realtime STT Service

Real-time speech-to-text transcription service using Voxtral Mini 4B Realtime 2602 ONNX model with CUDA GPU acceleration.

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
                        │  Voxtral ONNX    │
                        │  (CUDA GPU)      │
                        └──────────────────┘
```

## Project Structure

```
voxtral-stt/
├── CLAUDE.md           # Project guidance for Claude Code
├── Dockerfile          # CUDA-enabled container
├── requirements.txt    # Python dependencies
├── voxtral_onnx.py     # ONNX model wrapper
├── server.py           # FastAPI server with endpoints
├── benchmark.py        # Performance measurement script
├── scripts/
│   └── download_samples.py  # TTS test sample generator
├── test_samples/       # Directory for test audio files
└── model/              # Voxtral ONNX model (downloaded)
```

## Installation

### Local Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download model
hf download onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX --local-dir ./model
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

### ONNX Runtime Configuration
```python
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### DGX Spark Optimizations
- Unified memory architecture eliminates GPU-CPU transfer overhead
- 128GB memory allows full model loading without memory pressure
- CUDA graphs available for repeated inference patterns

## References

- [Voxtral Paper](https://arxiv.org/abs/2602.11298)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [HuggingFace Model Card](https://huggingface.co/onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX)

## Status

- [x] FastAPI server implementation
- [x] ONNX model wrapper
- [x] Docker configuration
- [x] Benchmark script
- [x] Test sample generator
- [ ] Model download (in progress)
- [ ] zh-TW/English validation
- [ ] Performance optimization for DGX Spark