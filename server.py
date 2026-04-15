"""
Voxtral STT FastAPI Server

Provides HTTP and WebSocket endpoints for real-time speech-to-text
transcription using Voxtral Mini Realtime ONNX model with CUDA acceleration.

Endpoints:
- POST /transcribe - Upload audio file for transcription
- WebSocket /stream - Real-time streaming transcription
- GET /health - Health check endpoint
- GET /info - Model information
"""

import os
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

from voxtral_onnx import VoxtralONNX, SAMPLE_RATE

# Create FastAPI app
app = FastAPI(
    title="Voxtral STT API",
    description="Real-time speech-to-text transcription using Voxtral Mini Realtime ONNX",
    version="1.0.0",
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (lazy loaded)
_model: Optional[VoxtralONNX] = None


def get_model() -> VoxtralONNX:
    """Get or initialize the global model instance."""
    global _model
    if _model is None:
        model_path = os.environ.get("VOXTRAL_MODEL_PATH", "./model")
        _model = VoxtralONNX(model_path)
    return _model


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    try:
        model = get_model()
        print(f"Model loaded successfully: {model.get_model_info()}")
    except Exception as e:
        print(f"Warning: Model not loaded on startup: {e}")


@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Target language hint"),
    return_timestamps: bool = Query(False, description="Return word timestamps")
):
    """
    Transcribe an uploaded audio file.

    Supports WAV, MP3, FLAC, OGG, and other formats.
    Audio is automatically resampled to 16kHz if needed.

    Returns JSON with transcription text, detected language, and confidence.
    """
    if sf is None:
        raise HTTPException(status_code=500, detail="soundfile not installed")

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Save to temp file and read (soundfile needs file-like or bytes)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            audio, sr = sf.read(tmp.name)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Transcribe
        model = get_model()
        result = model.transcribe(
            audio,
            sample_rate=sr,
            language=language,
            return_timestamps=return_timestamps
        )

        return JSONResponse({
            "text": result["text"],
            "language": result["language"],
            "confidence": result["confidence"],
            "audio_duration": len(audio) / sr,
            "filename": file.filename,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/stream")
async def transcribe_stream(
    websocket: WebSocket,
    language: Optional[str] = None,
    chunk_size_ms: int = 480
):
    """
    Real-time streaming transcription via WebSocket.

    Client sends raw audio bytes (16kHz mono float32 PCM recommended).
    Server responds with JSON transcription results.

    Message format:
    - Client: Binary audio data (float32 samples)
    - Server: JSON {"text": "...", "is_final": bool, "language": "..."}
    """
    await websocket.accept()

    model = get_model()
    chunk_samples = int(SAMPLE_RATE * chunk_size_ms / 1000)

    # Audio buffer for chunking
    buffer = np.zeros(0, dtype=np.float32)

    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()

            # Convert bytes to float32 array
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            # Accumulate in buffer
            buffer = np.concatenate([buffer, audio_chunk])

            # Process when buffer reaches chunk size
            while len(buffer) >= chunk_samples:
                chunk_to_process = buffer[:chunk_samples].copy()
                buffer = buffer[chunk_samples:]

                # Transcribe chunk
                result = model.transcribe(chunk_to_process, SAMPLE_RATE, language)

                # Send result
                await websocket.send_json({
                    "text": result["text"],
                    "is_final": False,
                    "language": result["language"],
                    "confidence": result["confidence"],
                })

    except WebSocketDisconnect:
        # Process remaining buffer on disconnect
        if len(buffer) > 0:
            result = model.transcribe(buffer, SAMPLE_RATE, language)
            try:
                await websocket.send_json({
                    "text": result["text"],
                    "is_final": True,
                    "language": result["language"],
                    "confidence": result["confidence"],
                })
            except:
                pass

    except Exception as e:
        await websocket.send_json({"error": str(e)})


@app.post("/transcribe_bytes")
async def transcribe_bytes(
    data: bytes,
    sample_rate: int = Query(16000, description="Audio sample rate"),
    language: Optional[str] = Query(None, description="Target language hint")
):
    """
    Transcribe raw audio bytes.

    Input should be float32 PCM mono audio.
    Specify sample_rate if different from 16kHz.
    """
    try:
        # Convert bytes to float32 array
        audio = np.frombuffer(data, dtype=np.float32)

        # Transcribe
        model = get_model()
        result = model.transcribe(audio, sample_rate, language)

        return JSONResponse({
            "text": result["text"],
            "language": result["language"],
            "confidence": result["confidence"],
            "audio_duration": len(audio) / sample_rate,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns status and GPU availability.
    """
    try:
        model = get_model()
        model_info = model.get_model_info()
        return {
            "status": "ok",
            "gpu_available": "CUDAExecutionProvider" in model_info["providers"],
            "providers": model_info["providers"],
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/info")
async def model_info():
    """
    Get model information.

    Returns supported languages, model parameters, and capabilities.
    """
    model = get_model()
    return JSONResponse(model.get_model_info())


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Voxtral STT API",
        "version": "1.0.0",
        "endpoints": {
            "/transcribe": "POST - Upload audio file for transcription",
            "/stream": "WebSocket - Real-time streaming transcription",
            "/transcribe_bytes": "POST - Transcribe raw audio bytes",
            "/health": "GET - Health check",
            "/info": "GET - Model information",
        },
        "docs": "/docs",
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()