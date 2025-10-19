"""
Bidirectional Streaming Translation Engine
Handles both English→German and German→English translation
"""

import numpy as np
import torch
from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional, Callable
from dataclasses import dataclass
import time


@dataclass
class TranslationSegment:
    """Represents a segment of translated speech"""
    source_text: str
    target_text: str
    audio_data: Optional[np.ndarray]
    timestamp: float
    mode: str  # "en_to_de" or "de_to_en"
    is_final: bool = False


class BidirectionalTranslationEngine:
    """
    Bidirectional streaming translation engine

    Supports:
    - English → German (EN→DE)
    - German → English (DE→EN)

    Pipeline:
    Audio Stream → Whisper ASR → Translation → TTS → Audio Output
    """

    def __init__(
        self,
        whisper_model_size: str = "large-v3",
        device: str = "cuda",
        mode: str = "en_to_de",  # "en_to_de" or "de_to_en"
        shared_whisper=None,  # Share Whisper instance to save VRAM
        tts_output_device=None  # Output device for TTS playback
    ):
        """
        Initialize bidirectional translation engine

        Args:
            whisper_model_size: Whisper model size (ignored if shared_whisper provided)
            device: Device to use (cuda or cpu)
            mode: Translation mode ("en_to_de" or "de_to_en")
            shared_whisper: Optional shared WhisperEngine instance to save VRAM
        """
        from .whisper_engine import WhisperEngine
        from .translator import EnglishToGermanTranslator, GermanToEnglishTranslator
        from ..tts.german_tts import GermanTTS
        from ..tts.english_tts import EnglishTTS
        import config

        self.device = device
        self.mode = mode

        print(f"Initializing bidirectional translation engine (mode: {mode})...")

        # Load or share Whisper for speech recognition
        if shared_whisper is not None:
            print("  Using shared Whisper instance (saves VRAM)")
            self.whisper = shared_whisper
            self.owns_whisper = False
        else:
            print("  Loading new Whisper instance")
            self.whisper = WhisperEngine(model_size=whisper_model_size, device=device)
            self.owns_whisper = True

        # Load translation models based on mode
        if mode == "en_to_de":
            self.translator = EnglishToGermanTranslator(device=device)
            self.tts = GermanTTS(
                model_name=config.TTS_MODEL,
                device=device,
                output_device_id=tts_output_device
            )
            print("Mode: English → German")
            if tts_output_device is not None:
                print(f"   TTS output routed to device {tts_output_device}")
        else:  # de_to_en
            self.translator = GermanToEnglishTranslator(device=device)
            self.tts = EnglishTTS(
                device=device,
                output_device_id=tts_output_device
            )
            print("Mode: German → English")

        # Queues and buffers
        self.translation_queue = Queue()

        # State tracking
        self.is_running = False
        self.stop_event = Event()

        # Processing thread
        self.translation_thread: Optional[Thread] = None

        # Callbacks
        self.on_source_text: Optional[Callable[[str], None]] = None
        self.on_target_text: Optional[Callable[[str], None]] = None

        # Performance metrics
        self.total_latency = 0.0
        self.segments_processed = 0

        print("Bidirectional translation engine ready!")

    def start(self) -> bool:
        """
        Start the translation engine

        Returns:
            True if started successfully
        """
        if self.is_running:
            print("Engine already running")
            return False

        self.is_running = True
        self.stop_event.clear()

        # Start TTS playback queue
        self.tts.start_playback_queue()

        # Start translation thread
        self.translation_thread = Thread(target=self._translation_worker, daemon=True)
        self.translation_thread.start()

        print(f"Translation started - mode: {self.mode}")
        return True

    def stop(self) -> None:
        """Stop the translation engine"""
        if not self.is_running:
            return

        print("Stopping translation...")
        self.is_running = False
        self.stop_event.set()

        # Stop TTS playback
        self.tts.stop_playback_queue()

        # Wait for thread
        if self.translation_thread:
            self.translation_thread.join(timeout=2.0)

        # Clear queue
        while not self.translation_queue.empty():
            try:
                self.translation_queue.get_nowait()
            except Empty:
                break

        print("Translation stopped")

    def process_audio(self, audio_data: np.ndarray) -> None:
        """
        Process audio chunk and add to translation queue (non-blocking)

        Args:
            audio_data: Audio data to process
        """
        if not self.is_running:
            return

        # Process in background thread to avoid blocking UI
        Thread(target=self._process_audio_async, args=(audio_data,), daemon=True).start()

    def _process_audio_async(self, audio_data: np.ndarray) -> None:
        """
        Async audio processing (runs in background thread)

        Args:
            audio_data: Audio data to process
        """
        start_time = time.time()

        try:
            # Step 1: Transcribe audio based on mode
            if self.mode == "en_to_de":
                # English audio → English text
                source_text = self.whisper.transcribe_english(
                    audio_data,
                    beam_size=1,  # Fast for real-time
                    vad_filter=False
                )
                source_lang = "English"
                target_lang = "German"
            else:  # de_to_en
                # German audio → German text
                # Use VAD filtering for loopback audio to reduce background noise
                source_text = self.whisper.transcribe_german(
                    audio_data,
                    beam_size=1,
                    vad_filter=True  # Enable VAD for better quality on system audio
                )
                source_lang = "German"
                target_lang = "English"

            if not source_text or not source_text.strip():
                return

            print(f"\n[Translation] {source_lang}: {source_text}")

            # Notify callback
            if self.on_source_text:
                self.on_source_text(source_text)

            # Add to translation queue
            self.translation_queue.put((source_text, start_time))

        except Exception as e:
            print(f"Error processing audio: {e}")

    def _translation_worker(self) -> None:
        """Background worker that processes translation queue"""
        while not self.stop_event.is_set():
            try:
                # Get source text from queue
                source_text, timestamp = self.translation_queue.get(timeout=0.1)

                # Translate to target language with high beam size for quality
                target_text = self.translator.translate(source_text, num_beams=5)

                # Calculate latency
                latency = time.time() - timestamp
                print(f"[Translation] Target: {target_text} (latency: {latency:.2f}s)")

                # Notify callback
                if self.on_target_text:
                    self.on_target_text(target_text)

                # Queue for TTS playback
                if target_text and target_text.strip():
                    self.tts.queue_text(target_text)

                # Update metrics
                self.total_latency += latency
                self.segments_processed += 1

            except Empty:
                continue
            except Exception as e:
                print(f"Translation error: {e}")

    def get_average_latency(self) -> float:
        """Get average translation latency"""
        if self.segments_processed == 0:
            return 0.0
        return self.total_latency / self.segments_processed

    def unload(self) -> None:
        """Unload all models"""
        self.stop()
        self.whisper.unload()
        self.translator.unload()
        self.tts.unload()
        print("Bidirectional translation engine unloaded")


def test_bidirectional_translation():
    """Test function for bidirectional translation"""
    print("Testing bidirectional translation...")

    # Test EN→DE mode
    print("\n" + "="*60)
    print("Testing English → German")
    print("="*60)

    engine_en_de = BidirectionalTranslationEngine(
        whisper_model_size="small",
        device="cuda",
        mode="en_to_de"
    )

    def on_english(text):
        print(f"  [EN] {text}")

    def on_german(text):
        print(f"  [DE] {text}")

    engine_en_de.on_source_text = on_english
    engine_en_de.on_target_text = on_german

    engine_en_de.start()

    # Simulate audio input (would come from microphone)
    print("\nSpeak in English now...")
    input("Press Enter when done...")

    engine_en_de.stop()
    engine_en_de.unload()

    # Test DE→EN mode
    print("\n" + "="*60)
    print("Testing German → English")
    print("="*60)

    engine_de_en = BidirectionalTranslationEngine(
        whisper_model_size="small",
        device="cuda",
        mode="de_to_en"
    )

    def on_german_input(text):
        print(f"  [DE] {text}")

    def on_english_output(text):
        print(f"  [EN] {text}")

    engine_de_en.on_source_text = on_german_input
    engine_de_en.on_target_text = on_english_output

    engine_de_en.start()

    print("\nSpeak in German now...")
    input("Press Enter when done...")

    engine_de_en.stop()
    engine_de_en.unload()

    print("\nTest complete!")


if __name__ == "__main__":
    test_bidirectional_translation()
