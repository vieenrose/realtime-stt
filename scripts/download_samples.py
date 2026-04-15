#!/usr/bin/env python3
"""
Download or generate test audio samples for Voxtral benchmarking.

This script helps obtain audio samples for testing:
1. zh-TW (Traditional Chinese) speech
2. English speech
3. Mixed zh-TW/English speech (code-switching)

Usage:
    python scripts/download_samples.py
    python scripts/download_samples.py --generate
"""

import argparse
import os
import json
from pathlib import Path


# Sample audio sources (public datasets or TTS generation)
SAMPLE_CONFIG = {
    "zh_tw": {
        "description": "Traditional Chinese speech sample",
        "sources": [
            # Common Voice dataset (Mozilla)
            "https://commonvoice.mozilla.org/",
            # Or use TTS to generate
        ],
        "tts_prompt": "台灣是一個美麗的島嶼，擁有豐富的文化和歷史。",
    },
    "english": {
        "description": "English speech sample",
        "sources": [
            "https://commonvoice.mozilla.org/",
        ],
        "tts_prompt": "Hello, this is a test of the speech recognition system.",
    },
    "mixed": {
        "description": "Mixed zh-TW and English speech (code-switching)",
        "sources": [
            # Generate with TTS
        ],
        "tts_prompt": "我喜歡用 Python programming language 來開發 software applications。Today is a good day for coding。",
    },
}


def generate_with_tts(prompt: str, output_path: str, language: str = "zh-TW"):
    """
    Generate test audio using TTS.

    Uses edge-tts (Microsoft Edge TTS, free) as default.
    """
    try:
        import edge_tts
    except ImportError:
        print("edge-tts not installed. Install with: pip install edge-tts")
        return False

    # Select appropriate voice
    voices = {
        "zh-TW": "zh-TW-HsiaoChenNeural",
        "en": "en-US-AriaNeural",
        "mixed": "zh-TW-HsiaoChenNeural",  # Will speak Chinese with English words
    }

    voice = voices.get(language, "en-US-AriaNeural")

    print(f"Generating audio with voice: {voice}")
    print(f"Prompt: {prompt}")

    communicate = edge_tts.Communicate(prompt, voice)
    communicate.save(output_path)

    print(f"Saved: {output_path}")
    return True


def download_common_voice(language: str, output_dir: str):
    """
    Download samples from Mozilla Common Voice dataset.

    Note: Requires manual download from https://commonvoice.mozilla.org/
    """
    print(f"To download Common Voice samples for {language}:")
    print("1. Visit https://commonvoice.mozilla.org/")
    print("2. Download the dataset for your language")
    print(f"3. Place samples in {output_dir}")
    return False


def generate_all_samples(output_dir: str = "./test_samples"):
    """Generate all test samples using TTS."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    for name, config in SAMPLE_CONFIG.items():
        output_path = os.path.join(output_dir, f"{name}.wav")
        prompt = config["tts_prompt"]

        # Determine language for TTS
        language = "zh-TW" if name == "zh_tw" else ("en" if name == "english" else "zh-TW")

        success = generate_with_tts(prompt, output_path, language)
        results[name] = {
            "path": output_path,
            "success": success,
            "prompt": prompt,
        }

    # Save metadata
    meta_path = os.path.join(output_dir, "samples_meta.json")
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMetadata saved: {meta_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Download/generate test audio samples")
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate samples using TTS (edge-tts)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./test_samples",
        help="Output directory for samples"
    )
    parser.add_argument(
        "--language", type=str, choices=["zh_tw", "english", "mixed", "all"],
        default="all",
        help="Which language sample to generate"
    )

    args = parser.parse_args()

    if args.generate:
        print("Generating test samples with TTS...")
        results = generate_all_samples(args.output_dir)

        print("\n=== Summary ===")
        for name, info in results.items():
            status = "✓" if info["success"] else "✗"
            print(f"  {status} {name}: {info['path']}")

        # Check if edge-tts is available
        try:
            import edge_tts
            print("\nTip: Install edge-tts for TTS generation: pip install edge-tts")
        except ImportError:
            print("\nNote: edge-tts not installed. Some samples may not be generated.")
            print("Install with: pip install edge-tts")
    else:
        print("Test sample sources:")
        for name, config in SAMPLE_CONFIG.items():
            print(f"\n{name}:")
            print(f"  Description: {config['description']}")
            print(f"  TTS prompt: {config['tts_prompt']}")

        print("\nUsage:")
        print("  python scripts/download_samples.py --generate")


if __name__ == "__main__":
    main()