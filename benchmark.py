"""
Voxtral STT Benchmark Script

Measures performance metrics for Voxtral Mini Realtime ONNX:
- Latency (first-packet and total)
- Real-time factor (RTF)
- GPU memory usage
- Throughput (tokens/second)

Usage:
    python benchmark.py --audio test.wav --iterations 10
    python benchmark.py --generate-test --duration 10
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import os

try:
    import soundfile as sf
except ImportError:
    print("Please install soundfile: pip install soundfile")
    exit(1)

from voxtral_onnx import VoxtralRealtime, SAMPLE_RATE


def benchmark_transcription(
    model: VoxtralRealtime,
    audio: np.ndarray,
    sample_rate: int,
    iterations: int = 10
) -> Dict[str, Any]:
    """
    Run benchmark on transcription performance.

    Args:
        model: VoxtralONNX model instance
        audio: Audio waveform
        sample_rate: Audio sample rate
        iterations: Number of iterations to run

    Returns:
        Benchmark results dict
    """
    latencies = []
    first_token_latencies = []
    audio_duration = len(audio) / sample_rate

    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Running {iterations} iterations...")

    for i in range(iterations):
        # Measure total latency
        start_time = time.perf_counter()
        result = model.transcribe(audio, sample_rate)
        end_time = time.perf_counter()

        latency = end_time - start_time
        latencies.append(latency)

        # Print progress
        print(f"  Iteration {i+1}: {latency*1000:.2f}ms - {result['text'][:50]}...")

    # Calculate statistics
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    # Real-time factor (RTF)
    rtf = mean_latency / audio_duration

    # Throughput (audio seconds processed per second)
    throughput = audio_duration / mean_latency

    return {
        "audio_duration": audio_duration,
        "iterations": iterations,
        "latency_ms": {
            "mean": mean_latency * 1000,
            "std": std_latency * 1000,
            "min": min_latency * 1000,
            "max": max_latency * 1000,
        },
        "real_time_factor": rtf,
        "throughput": throughput,
        "transcript": result["text"],
        "language": result["language"],
    }


def benchmark_streaming(
    model: VoxtralRealtime,
    audio: np.ndarray,
    chunk_size_ms: int = 480
) -> Dict[str, Any]:
    """
    Benchmark streaming transcription performance.

    Args:
        model: VoxtralONNX model instance
        audio: Audio waveform
        chunk_size_ms: Chunk size in milliseconds

    Returns:
        Streaming benchmark results
    """
    chunk_samples = int(SAMPLE_RATE * chunk_size_ms / 1000)
    num_chunks = len(audio) // chunk_samples
    audio_duration = len(audio) / SAMPLE_RATE

    print(f"Streaming benchmark: {num_chunks} chunks of {chunk_size_ms}ms")

    chunk_latencies = []
    first_token_latency = None
    tokens_generated = 0

    for i in range(num_chunks):
        chunk = audio[i * chunk_samples:(i + 1) * chunk_samples]

        start_time = time.perf_counter()
        result = model.transcribe(chunk, SAMPLE_RATE)
        end_time = time.perf_counter()

        latency = end_time - start_time
        chunk_latencies.append(latency)

        # Count tokens (approximate by text length)
        tokens_generated += len(result["text"]) // 2  # Rough estimate

        # Track first token latency
        if i == 0:
            first_token_latency = latency

        print(f"  Chunk {i+1}: {latency*1000:.2f}ms")

    # Calculate streaming metrics
    mean_chunk_latency = np.mean(chunk_latencies)
    tokens_per_second = tokens_generated / (num_chunks * chunk_size_ms / 1000)

    return {
        "chunk_size_ms": chunk_size_ms,
        "num_chunks": num_chunks,
        "audio_duration": audio_duration,
        "first_token_latency_ms": first_token_latency * 1000 if first_token_latency else None,
        "chunk_latency_ms": {
            "mean": mean_chunk_latency * 1000,
            "std": np.std(chunk_latencies) * 1000,
        },
        "tokens_per_second": tokens_per_second,
    }


def benchmark_memory():
    """Benchmark GPU memory usage."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Get memory info before loading model
        mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory before: {mem_before.used / 1024**2:.2f} MB used")

        # Load model
        model = VoxtralRealtime()

        # Get memory info after loading model
        mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory after: {mem_after.used / 1024**2:.2f} MB used")

        model_memory = mem_after.used - mem_before.used
        print(f"Model memory footprint: {model_memory / 1024**2:.2f} MB")

        return {
            "gpu_memory_before_mb": mem_before.used / 1024**2,
            "gpu_memory_after_mb": mem_after.used / 1024**2,
            "model_memory_mb": model_memory / 1024**2,
            "gpu_total_mb": mem_after.total / 1024**2,
        }

    except ImportError:
        print("pynvml not installed, skipping memory benchmark")
        return None
    except Exception as e:
        print(f"Could not measure GPU memory: {e}")
        return None


def generate_test_audio(
    duration: float = 10.0,
    output_path: str = "test_audio.wav"
) -> np.ndarray:
    """
    Generate synthetic test audio (silence with some noise).

    For real testing, use actual speech samples.
    """
    num_samples = int(SAMPLE_RATE * duration)
    # Generate some noise as placeholder
    audio = np.random.randn(num_samples).astype(np.float32) * 0.01

    # Save to file
    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"Generated test audio: {output_path} ({duration}s)")

    return audio


def find_test_samples(directory: str = "./test_samples") -> List[Path]:
    """Find test audio samples in directory."""
    test_dir = Path(directory)
    if not test_dir.exists():
        return []

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return [f for f in test_dir.iterate() if f.suffix.lower() in audio_extensions]


def main():
    parser = argparse.ArgumentParser(description="Voxtral STT Benchmark")
    parser.add_argument(
        "--audio", type=str, default=None,
        help="Path to audio file for benchmarking"
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--model-id", type=str, default="mistralai/Voxtral-Mini-4B-Realtime-2602",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--onnx-model-path", type=str, default="./model",
        help="Path to ONNX model directory (alternative to model-id)"
    )
    parser.add_argument(
        "--use-onnx", action="store_true",
        help="Use local ONNX model instead of downloading"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for inference (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Also run streaming benchmark"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=480,
        help="Chunk size in ms for streaming benchmark"
    )
    parser.add_argument(
        "--memory", action="store_true",
        help="Include GPU memory benchmark"
    )
    parser.add_argument(
        "--generate-test", action="store_true",
        help="Generate synthetic test audio"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Duration of synthetic test audio (seconds)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for benchmark results (JSON)"
    )

    args = parser.parse_args()

    # Generate test audio if requested
    if args.generate_test:
        audio = generate_test_audio(args.duration)
        sample_rate = SAMPLE_RATE
    elif args.audio:
        audio, sample_rate = sf.read(args.audio)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        print(f"Loaded audio: {args.audio} ({len(audio)/sample_rate:.2f}s)")
    else:
        print("No audio file specified. Use --audio or --generate-test")
        return

    # Initialize model
    print(f"\nLoading model...")
    if args.use_onnx:
        print(f"Using ONNX model from: {args.onnx_model_path}")
        model = VoxtralRealtime(
            use_onnx=True,
            onnx_model_path=args.onnx_model_path,
            device=args.device
        )
    else:
        print(f"Downloading/using model: {args.model_id}")
        model = VoxtralRealtime(
            model_id=args.model_id,
            device=args.device
        )
    model_info = model.get_model_info()
    print(f"Model info: {json.dumps(model_info, indent=2)}")

    # Run transcription benchmark
    print("\n=== Transcription Benchmark ===")
    trans_results = benchmark_transcription(model, audio, sample_rate, args.iterations)

    print(f"\nResults:")
    print(f"  Mean latency: {trans_results['latency_ms']['mean']:.2f}ms")
    print(f"  Std latency: {trans_results['latency_ms']['std']:.2f}ms")
    print(f"  RTF: {trans_results['real_time_factor']:.3f}")
    print(f"  Throughput: {trans_results['throughput']:.2f}x realtime")
    print(f"  Transcript: {trans_results['transcript']}")

    # Run streaming benchmark if requested
    stream_results = None
    if args.streaming:
        print("\n=== Streaming Benchmark ===")
        stream_results = benchmark_streaming(model, audio, args.chunk_size)

        print(f"\nStreaming Results:")
        print(f"  First token latency: {stream_results['first_token_latency_ms']:.2f}ms")
        print(f"  Chunk latency: {stream_results['chunk_latency_ms']['mean']:.2f}ms")
        print(f"  Tokens/second: {stream_results['tokens_per_second']:.2f}")

    # Run memory benchmark if requested
    mem_results = None
    if args.memory:
        print("\n=== Memory Benchmark ===")
        mem_results = benchmark_memory()

    # Compile all results
    all_results = {
        "model_info": model_info,
        "transcription": trans_results,
        "streaming": stream_results,
        "memory": mem_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Print summary
    print("\n=== Summary ===")
    rtf = trans_results['real_time_factor']
    if rtf < 1.0:
        print(f"  ✓ RTF < 1.0: Faster than realtime! ({rtf:.3f})")
    else:
        print(f"  ✗ RTF > 1.0: Slower than realtime ({rtf:.3f})")

    latency = trans_results['latency_ms']['mean']
    if latency < 500:
        print(f"  ✓ Latency < 500ms: Good for realtime ({latency:.2f}ms)")
    else:
        print(f"  ✗ Latency > 500ms: May impact realtime experience ({latency:.2f}ms)")


if __name__ == "__main__":
    main()