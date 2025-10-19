"""
English Text-to-Speech
Local TTS using Coqui TTS for English voice synthesis
"""

import torch
import numpy as np
import sounddevice as sd
from TTS.api import TTS
from typing import Optional, List
import queue
import threading


class EnglishTTS:
    """
    English text-to-speech engine using Coqui TTS
    GPU-accelerated with real-time playback
    """

    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        device: str = "cuda",
        output_device_id: Optional[int] = None
    ):
        """
        Initialize English TTS engine

        Args:
            model_name: TTS model name (default is English LJSpeech voice)
            device: Device to run on ("cuda" or "cpu")
            output_device_id: Output device for audio playback (None = default)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.output_device_id = output_device_id

        print(f"Loading TTS model: {model_name}...")

        # Initialize TTS
        self.tts = TTS(model_name=model_name, progress_bar=False).to(self.device)

        print(f"TTS model loaded on {self.device}")
        if self.device == "cuda":
            print(f"VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        if output_device_id is not None:
            print(f"TTS output device: {output_device_id}")

        # Audio playback queue
        self.playback_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread: Optional[threading.Thread] = None
        self.stop_playback_event = threading.Event()

        # Recording settings
        self.save_recordings = False
        self.recording_counter = 0

    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None
    ) -> np.ndarray:
        """
        Synthesize English text to speech

        Args:
            text: English text to synthesize
            output_file: Optional file path to save audio

        Returns:
            Audio data as numpy array
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        # Generate speech
        if output_file:
            # Save to file
            self.tts.tts_to_file(text=text, file_path=output_file)
            # Also return audio data
            audio = self.tts.tts(text=text)
        else:
            # Just return audio data
            audio = self.tts.tts(text=text)

        # Convert to numpy array if needed
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)

        return audio

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int = 22050,
        blocking: bool = True
    ) -> None:
        """
        Play audio through default output device

        Args:
            audio: Audio data to play
            sample_rate: Sample rate of audio
            blocking: If True, wait for playback to complete
        """
        if len(audio) == 0:
            return

        # Ensure audio is in correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # Play audio to specified device
        sd.play(audio, samplerate=sample_rate, device=self.output_device_id)

        if blocking:
            sd.wait()

    def speak(
        self,
        text: str,
        blocking: bool = True
    ) -> None:
        """
        Synthesize and play text in one call

        Args:
            text: English text to speak
            blocking: If True, wait for speech to complete
        """
        audio = self.synthesize(text)
        if len(audio) > 0:
            self.play(audio, sample_rate=22050, blocking=blocking)

    def start_playback_queue(self) -> None:
        """
        Start background playback queue
        Allows queueing multiple texts for sequential playback
        """
        if self.is_playing:
            return

        self.is_playing = True
        self.stop_playback_event.clear()

        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        self.playback_thread.start()
        print("Playback queue started")

    def stop_playback_queue(self) -> None:
        """Stop background playback queue"""
        if not self.is_playing:
            return

        self.is_playing = False
        self.stop_playback_event.set()

        # Clear queue
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for thread
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
            self.playback_thread = None

        # Stop any ongoing playback
        sd.stop()
        print("Playback queue stopped")

    def queue_text(self, text: str) -> None:
        """
        Add text to playback queue

        Args:
            text: English text to queue for playback
        """
        if not self.is_playing:
            self.start_playback_queue()

        self.playback_queue.put(text)

    def _playback_worker(self) -> None:
        """Background worker that processes playback queue"""
        while not self.stop_playback_event.is_set():
            try:
                # Get text from queue
                text = self.playback_queue.get(timeout=0.1)

                # Synthesize and play
                audio = self.synthesize(text)
                if len(audio) > 0:
                    # Save English audio if enabled
                    if self.save_recordings:
                        self._save_english_audio(audio, text)

                    self.play(audio, blocking=True)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Playback error: {e}")

    def _save_english_audio(self, audio_data, text):
        """Save English TTS audio to WAV"""
        try:
            from scipy.io import wavfile
            import os
            from datetime import datetime

            # Create recordings directory
            os.makedirs("recordings", exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_counter += 1
            wav_path = f"recordings/english_output_{timestamp}_{self.recording_counter}.wav"

            # Convert to int16 and save as WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(wav_path, 22050, audio_int16)

            print(f"✓ Saved English output audio: {wav_path}")

        except Exception as e:
            print(f"Error saving English audio: {e}")

    def set_save_recordings(self, enabled: bool):
        """Enable/disable recording saving"""
        self.save_recordings = enabled

    def get_available_models(self) -> List[str]:
        """Get list of available English TTS models"""
        all_models = TTS().list_models()
        english_models = [m for m in all_models if "/en/" in m or "english" in m.lower()]
        return english_models

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        }

    def unload(self) -> None:
        """Unload model and free GPU memory"""
        # Stop playback
        self.stop_playback_queue()

        # Unload model
        del self.tts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("TTS model unloaded")


def test_english_tts():
    """Test function for English TTS"""
    print("Testing English TTS...")

    # Create TTS engine
    tts = EnglishTTS(device="cuda")

    # Print model info
    info = tts.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test sentences
    test_sentences = [
        "Hello! How are you today?",
        "I am learning German language.",
        "The weather is beautiful today.",
        "What time is it?",
        "Nice to meet you!"
    ]

    print("\nPlaying test sentences...")
    for i, text in enumerate(test_sentences, 1):
        print(f"{i}. {text}")
        tts.speak(text, blocking=True)

    print("\nTesting playback queue...")
    tts.start_playback_queue()
    for text in test_sentences:
        tts.queue_text(text)

    # Wait for queue to finish
    import time
    while not tts.playback_queue.empty():
        time.sleep(0.5)
    time.sleep(2)  # Wait for last audio to finish

    tts.stop_playback_queue()

    # Cleanup
    tts.unload()
    print("Test complete")


if __name__ == "__main__":
    test_english_tts()
