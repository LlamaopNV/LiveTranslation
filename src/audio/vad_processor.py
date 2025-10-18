"""
Voice Activity Detection (VAD) Processor
Uses Silero VAD for accurate speech detection and sentence boundary detection
"""

import numpy as np
import torch
from typing import Optional, Callable
from collections import deque


class VADProcessor:
    """
    Voice Activity Detection processor using Silero VAD
    Detects speech segments and sentence boundaries
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 30
    ):
        """
        Initialize VAD processor

        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000 for Silero)
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration to consider valid
            min_silence_duration_ms: Minimum silence to end speech segment
            speech_pad_ms: Padding around speech segments
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sample_rate * speech_pad_ms / 1000)

        # Load Silero VAD model
        print("Loading Silero VAD model...")
        try:
            # Load from torch hub
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            print("VAD model loaded successfully")
        except Exception as e:
            print(f"Failed to load Silero VAD: {e}")
            print("Using fallback energy-based VAD")
            self.model = None

        # State tracking
        self.is_speech = False
        self.speech_start = 0
        self.silence_duration = 0
        self.speech_duration = 0

        # Buffer for speech segments
        self.current_speech = []

        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable[[np.ndarray], None]] = None

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        return_speech_prob: bool = False
    ) -> Optional[dict]:
        """
        Process audio chunk and detect voice activity

        Args:
            audio_chunk: Audio data (float32, mono)
            return_speech_prob: If True, return speech probability

        Returns:
            Dictionary with VAD results or None
        """
        if self.model is None:
            # Fallback to energy-based VAD
            return self._energy_based_vad(audio_chunk)

        # Ensure audio is float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_chunk)

        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        is_speech = speech_prob >= self.threshold

        result = {
            'is_speech': is_speech,
            'speech_prob': speech_prob,
            'speech_segment': None
        }

        # State machine for speech detection
        if is_speech:
            if not self.is_speech:
                # Speech started
                self.is_speech = True
                self.speech_start = 0
                self.silence_duration = 0
                self.current_speech = []

                if self.on_speech_start:
                    self.on_speech_start()

            # Accumulate speech
            self.current_speech.append(audio_chunk)
            self.speech_duration += len(audio_chunk)
            self.silence_duration = 0

        else:
            if self.is_speech:
                # Silence during speech
                self.current_speech.append(audio_chunk)
                self.silence_duration += len(audio_chunk)

                # Check if silence is long enough to end speech
                if self.silence_duration >= self.min_silence_samples:
                    # Speech ended
                    if self.speech_duration >= self.min_speech_samples:
                        # Valid speech segment
                        speech_array = np.concatenate(self.current_speech)
                        result['speech_segment'] = speech_array

                        if self.on_speech_end:
                            self.on_speech_end(speech_array)

                    # Reset state
                    self.is_speech = False
                    self.current_speech = []
                    self.speech_duration = 0
                    self.silence_duration = 0

        return result

    def _energy_based_vad(self, audio_chunk: np.ndarray) -> dict:
        """
        Fallback energy-based VAD
        Simple RMS-based voice activity detection
        """
        rms = np.sqrt(np.mean(audio_chunk**2))
        is_speech = rms > 0.01  # Energy threshold

        return {
            'is_speech': is_speech,
            'speech_prob': min(rms * 10, 1.0),  # Rough approximation
            'speech_segment': None
        }

    def get_current_speech(self) -> Optional[np.ndarray]:
        """Get currently accumulated speech segment"""
        if len(self.current_speech) > 0:
            return np.concatenate(self.current_speech)
        return None

    def reset(self) -> None:
        """Reset VAD state"""
        self.is_speech = False
        self.speech_start = 0
        self.silence_duration = 0
        self.speech_duration = 0
        self.current_speech = []

    def finalize_speech(self) -> Optional[np.ndarray]:
        """
        Finalize current speech segment (if any)
        Useful for ending detection when stream stops
        """
        if len(self.current_speech) > 0 and self.speech_duration >= self.min_speech_samples:
            speech_array = np.concatenate(self.current_speech)
            self.reset()
            return speech_array
        self.reset()
        return None


class SentenceBoundaryDetector:
    """
    Detects sentence boundaries in continuous speech
    Combines VAD with pause detection
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        pause_threshold_ms: int = 700,  # Typical inter-sentence pause
        max_sentence_duration_s: float = 10.0
    ):
        """
        Initialize sentence boundary detector

        Args:
            sample_rate: Audio sample rate
            pause_threshold_ms: Pause duration to detect sentence boundary
            max_sentence_duration_s: Maximum sentence duration before forcing split
        """
        self.sample_rate = sample_rate
        self.pause_threshold_samples = int(sample_rate * pause_threshold_ms / 1000)
        self.max_sentence_samples = int(sample_rate * max_sentence_duration_s)

        # VAD processor
        self.vad = VADProcessor(
            sample_rate=sample_rate,
            min_silence_duration_ms=pause_threshold_ms
        )

        # Sentence buffer
        self.sentence_buffer = []
        self.sentence_duration = 0

        # Callbacks
        self.on_sentence_end: Optional[Callable[[np.ndarray], None]] = None

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio chunk and detect sentence boundaries

        Args:
            audio_chunk: Audio data

        Returns:
            Complete sentence audio if boundary detected, None otherwise
        """
        # Process through VAD
        result = self.vad.process_chunk(audio_chunk)

        if result['is_speech']:
            # Accumulate speech
            self.sentence_buffer.append(audio_chunk)
            self.sentence_duration += len(audio_chunk)

            # Check for max duration
            if self.sentence_duration >= self.max_sentence_samples:
                return self._finalize_sentence()

        else:
            # Check if we have accumulated speech to finalize
            if result['speech_segment'] is not None:
                return self._finalize_sentence()

        return None

    def _finalize_sentence(self) -> Optional[np.ndarray]:
        """Finalize and return current sentence"""
        if len(self.sentence_buffer) == 0:
            return None

        sentence_audio = np.concatenate(self.sentence_buffer)

        # Reset buffer
        self.sentence_buffer = []
        self.sentence_duration = 0

        # Callback
        if self.on_sentence_end:
            self.on_sentence_end(sentence_audio)

        return sentence_audio

    def finalize(self) -> Optional[np.ndarray]:
        """Force finalize current sentence"""
        return self._finalize_sentence()

    def reset(self) -> None:
        """Reset detector state"""
        self.vad.reset()
        self.sentence_buffer = []
        self.sentence_duration = 0


def test_vad():
    """Test VAD processor"""
    import sounddevice as sd
    import time

    print("Testing VAD processor...")
    print("Speak into your microphone")

    vad = VADProcessor(sample_rate=16000)

    def on_start():
        print("\n🎤 Speech started...")

    def on_end(audio):
        duration = len(audio) / 16000
        print(f"✅ Speech ended (duration: {duration:.2f}s, samples: {len(audio)})")

    vad.on_speech_start = on_start
    vad.on_speech_end = on_end

    # Record and process
    sample_rate = 16000
    chunk_size = int(sample_rate * 0.1)  # 100ms chunks

    def audio_callback(indata, frames, time_info, status):
        audio = indata[:, 0].copy().astype(np.float32)
        result = vad.process_chunk(audio)
        if result['is_speech']:
            print("█", end="", flush=True)
        else:
            print("░", end="", flush=True)

    print("\nRecording... (Ctrl+C to stop)\n")

    stream = sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=chunk_size,
        callback=audio_callback
    )

    try:
        with stream:
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n\nStopping...")

    print("Test complete")


if __name__ == "__main__":
    test_vad()
