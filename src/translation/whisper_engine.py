"""
Whisper Translation Engine
GPU-accelerated speech recognition and translation using faster-whisper
"""

import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Literal, Tuple, List
import torch


class WhisperEngine:
    """
    Manages Whisper model for speech-to-text translation
    Optimized for GPU inference with minimal latency
    """

    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        cpu_threads: int = 4,
        num_workers: int = 1
    ):
        """
        Initialize Whisper engine

        Args:
            model_size: Model size (tiny/base/small/medium/large)
            device: Device to run on ("cuda" or "cpu")
            compute_type: Computation type ("float16", "int8" for GPU, "int8" for CPU)
            cpu_threads: Number of CPU threads for preprocessing
            num_workers: Number of workers for parallel processing
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.compute_type = "int8"

        print(f"Loading Whisper model: {model_size} on {self.device}...")

        # Load model with faster-whisper
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )

        print(f"Whisper model loaded successfully")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        task: Literal["transcribe", "translate"] = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None
    ) -> Tuple[str, List[dict]]:
        """
        Transcribe or translate audio

        Args:
            audio: Audio data as numpy array (float32, 16kHz)
            language: Source language code (e.g., "en", "de")
            task: "transcribe" for same language, "translate" for English translation
            beam_size: Beam search size (higher = better quality but slower)
            vad_filter: Enable Voice Activity Detection filtering
            vad_parameters: Custom VAD parameters

        Returns:
            Tuple of (full_text, segments_list)
        """
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize audio to [-1, 1] range if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.abs(audio).max()

        # Set default VAD parameters for better segmentation
        if vad_parameters is None:
            vad_parameters = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500
            }

        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            without_timestamps=False
        )

        # Collect segments
        text_segments = []
        full_text = ""

        for segment in segments:
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            }
            text_segments.append(segment_dict)
            full_text += segment.text

        full_text = full_text.strip()

        return full_text, text_segments

    def translate_to_german(
        self,
        audio: np.ndarray,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> str:
        """
        Translate English audio to German text

        Args:
            audio: Audio data (English speech)
            beam_size: Beam search size
            vad_filter: Enable VAD filtering

        Returns:
            Translated German text
        """
        # Note: Whisper's "translate" task always translates TO English
        # So we need a different approach for EN->DE translation

        # First, transcribe English audio to English text
        english_text, _ = self.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter
        )

        # Return English text (will be translated by separate model)
        # This is a placeholder - we'll integrate proper translation next
        return english_text

    def transcribe_english(
        self,
        audio: np.ndarray,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> str:
        """
        Transcribe English audio to English text

        Args:
            audio: Audio data (English speech)
            beam_size: Beam search size
            vad_filter: Enable VAD filtering

        Returns:
            Transcribed English text
        """
        text, _ = self.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter
        )
        return text

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        }

    def unload(self) -> None:
        """Unload model and free GPU memory"""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Whisper model unloaded")


def test_whisper_engine():
    """Test function for Whisper engine"""
    import sounddevice as sd

    print("Testing Whisper engine...")

    # Create engine
    engine = WhisperEngine(model_size="small", device="cuda")

    # Print model info
    info = engine.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Record test audio
    print("\nRecording 5 seconds of audio...")
    print("Speak in English: 'Hello, how are you today?'")

    sample_rate = 16000
    duration = 5
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()

    print("Recording complete, transcribing...")

    # Transcribe
    text = engine.transcribe_english(audio.flatten())
    print(f"\nTranscribed text: {text}")

    # Cleanup
    engine.unload()


if __name__ == "__main__":
    test_whisper_engine()
