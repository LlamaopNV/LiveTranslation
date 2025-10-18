"""
Live Translation Application
Real-time English to German speech-to-speech translation

Usage:
    python main.py              # Launch GUI
    python main.py --cli        # CLI mode for testing
    python main.py --test       # Test models only
"""

import os
import sys
import argparse
from pathlib import Path

# Fix OpenMP conflict (common with PyTorch + other ML libraries)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add espeak-ng to PATH if not already there
espeak_path = r"C:\Program Files\eSpeak NG"
if os.path.exists(espeak_path) and espeak_path not in os.environ['PATH']:
    os.environ['PATH'] = espeak_path + os.pathsep + os.environ['PATH']

# Add CUDA 12.4 bin to PATH for CTranslate2 compatibility
cuda_12_4_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if os.path.exists(cuda_12_4_path) and cuda_12_4_path not in os.environ['PATH']:
    os.environ['PATH'] = cuda_12_4_path + os.pathsep + os.environ['PATH']

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_gui():
    """Run the GUI application"""
    from PyQt6.QtWidgets import QApplication
    from src.gui.main_window import LiveTranslationWindow

    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show window
    window = LiveTranslationWindow()
    window.show()

    sys.exit(app.exec())


def run_cli():
    """Run in CLI mode (for testing/debugging)"""
    from src.translation.streaming_translator import test_streaming_translation

    print("=" * 60)
    print("Live Translation - CLI Mode")
    print("=" * 60)
    print()

    test_streaming_translation()

def test_models():
    """Test all models individually"""
    print("=" * 60)
    print("Model Testing Mode")
    print("=" * 60)
    print()

    # Test 1: Audio devices
    print("[1/5] Testing Audio Devices...")
    try:
        from src.audio.device_manager import list_audio_devices
        list_audio_devices()
        print("✓ Audio devices OK\n")
    except Exception as e:
        print(f"✗ Audio device error: {e}\n")

    # Test 2: Whisper
    print("[2/5] Testing Whisper Model...")
    try:
        from src.translation.whisper_engine import WhisperEngine
        import numpy as np

        engine = WhisperEngine(model_size="tiny", device="cuda")
        info = engine.get_model_info()
        print(f"  Model: {info['model_size']}")
        print(f"  Device: {info['device']}")
        print(f"  GPU: {info['gpu_name']}")
        print(f"  VRAM: {info['vram_allocated_gb']:.2f} GB")
        engine.unload()
        print("✓ Whisper OK\n")
    except Exception as e:
        print(f"✗ Whisper error: {e}\n")

    # Test 3: Translator
    print("[3/5] Testing Translation Model...")
    try:
        from src.translation.translator import EnglishToGermanTranslator

        translator = EnglishToGermanTranslator(device="cuda")
        test_text = "Hello, how are you?"
        german = translator.translate(test_text)
        print(f"  EN: {test_text}")
        print(f"  DE: {german}")
        translator.unload()
        print("✓ Translator OK\n")
    except Exception as e:
        print(f"✗ Translator error: {e}\n")

    # Test 4: TTS
    print("[4/5] Testing TTS Model...")
    try:
        from src.tts.german_tts import GermanTTS

        tts = GermanTTS(device="cuda")
        test_text = "Hallo Welt"
        audio = tts.synthesize(test_text)
        print(f"  Text: {test_text}")
        print(f"  Audio samples: {len(audio)}")
        tts.unload()
        print("✓ TTS OK\n")
    except Exception as e:
        print(f"✗ TTS error: {e}\n")

    # Test 5: VAD
    print("[5/5] Testing VAD Model...")
    try:
        from src.audio.vad_processor import VADProcessor
        import numpy as np

        vad = VADProcessor()
        # Test with dummy audio (512 samples for 16kHz as required by Silero VAD)
        dummy_audio = np.random.randn(512).astype(np.float32) * 0.01
        result = vad.process_chunk(dummy_audio)
        print(f"  VAD result: {result}")
        print("✓ VAD OK\n")
    except Exception as e:
        print(f"✗ VAD error: {e}\n")

    print("=" * 60)
    print("Testing Complete!")
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Live Translation - Real-time English to German speech translation"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode (for testing)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test all models individually"
    )
    parser.add_argument(
        "--model-size",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="Whisper model size (default: small)"
    )

    args = parser.parse_args()

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Translation will be slower on CPU.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    except ImportError:
        print("ERROR: PyTorch not installed. Please install requirements.txt")
        sys.exit(1)

    # Run appropriate mode
    if args.test:
        test_models()
    elif args.cli:
        run_cli()
    else:
        run_gui()


if __name__ == "__main__":
    main()
