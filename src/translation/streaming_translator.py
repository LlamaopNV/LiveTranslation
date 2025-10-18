"""
Streaming Real-Time Translation Engine
Handles live English→German speech-to-speech translation with minimal latency
Designed for interview/live demonstration scenarios
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
    english_text: str
    german_text: str
    audio_data: Optional[np.ndarray]
    timestamp: float
    is_final: bool = False


class StreamingTranslationEngine:
    """
    Real-time streaming translation engine

    Pipeline:
    Audio Stream → VAD → Streaming ASR → Incremental Translation → TTS Queue

    Key Features:
    - Processes audio in small chunks (~250ms) for minimal latency
    - Uses VAD to detect sentence boundaries
    - Translates complete sentences immediately when detected
    - Queues German audio for immediate playback
    """

    def __init__(
        self,
        whisper_model_size: str = "small",
        device: str = "cuda",
        min_chunk_duration: float = 0.3,  # 300ms minimum for processing
        vad_threshold: float = 0.5
    ):
        """
        Initialize streaming translation engine

        Args:
            whisper_model_size: Whisper model size (small recommended for speed)
            device: Device to use (cuda or cpu)
            min_chunk_duration: Minimum audio chunk duration in seconds
            vad_threshold: Voice activity detection threshold
        """
        from .whisper_engine import WhisperEngine
        from .translator import EnglishToGermanTranslator
        from ..tts.german_tts import GermanTTS
        import config

        self.device = device
        self.min_chunk_duration = min_chunk_duration
        self.vad_threshold = vad_threshold

        print("Initializing streaming translation engine...")

        # Load models
        self.whisper = WhisperEngine(model_size=whisper_model_size, device=device)
        self.translator = EnglishToGermanTranslator(device=device)
        self.tts = GermanTTS(model_name=config.TTS_MODEL, device=device)

        # Queues and buffers
        self.audio_queue = Queue()
        self.translation_queue = Queue()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.english_buffer = ""

        # State tracking
        self.is_running = False
        self.stop_event = Event()

        # Processing threads
        self.asr_thread: Optional[Thread] = None
        self.translation_thread: Optional[Thread] = None
        self.tts_thread: Optional[Thread] = None

        # Callbacks
        self.on_english_text: Optional[Callable[[str], None]] = None
        self.on_german_text: Optional[Callable[[str], None]] = None
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[], None]] = None

        # Performance metrics
        self.total_latency = 0.0
        self.segments_processed = 0

        print("Streaming translation engine ready!")

    def start(self) -> bool:
        """
        Start the streaming translation engine

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

        # Start processing threads
        self.asr_thread = Thread(target=self._asr_worker, daemon=True)
        self.translation_thread = Thread(target=self._translation_worker, daemon=True)

        self.asr_thread.start()
        self.translation_thread.start()

        print("Streaming translation started - speak English to hear German!")
        return True

    def stop(self) -> None:
        """Stop the streaming translation engine"""
        if not self.is_running:
            return

        print("Stopping streaming translation...")
        self.is_running = False
        self.stop_event.set()

        # Stop TTS playback
        self.tts.stop_playback_queue()

        # Wait for threads
        if self.asr_thread:
            self.asr_thread.join(timeout=2.0)
        if self.translation_thread:
            self.translation_thread.join(timeout=2.0)

        # Clear buffers
        self.audio_buffer = np.array([], dtype=np.float32)
        self.english_buffer = ""

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break

        while not self.translation_queue.empty():
            try:
                self.translation_queue.get_nowait()
            except Empty:
                break

        print("Streaming translation stopped")
        if self.segments_processed > 0:
            avg_latency = self.total_latency / self.segments_processed
            print(f"Average latency: {avg_latency:.2f}s per segment")

    def process_audio(self, audio_chunk: np.ndarray) -> None:
        """
        Process incoming audio chunk

        Args:
            audio_chunk: Audio data from microphone
        """
        if not self.is_running:
            return

        self.audio_queue.put(audio_chunk)

    def _asr_worker(self) -> None:
        """
        Background worker for ASR processing
        Continuously processes audio chunks and outputs English text
        """
        print("[ASR Worker] Started")

        while not self.stop_event.is_set():
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)

                # Add to buffer
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

                # Check if we have enough audio to process
                duration = len(self.audio_buffer) / 16000
                if duration < self.min_chunk_duration:
                    continue

                # Check for voice activity (simple energy-based VAD)
                rms = np.sqrt(np.mean(self.audio_buffer**2))

                if rms < 0.01:  # Silence threshold
                    # If buffer has accumulated speech, process it
                    if len(self.english_buffer) > 0:
                        self._finalize_sentence()
                    self.audio_buffer = np.array([], dtype=np.float32)
                    continue

                # Transcribe accumulated audio
                english_text = self.whisper.transcribe_english(
                    self.audio_buffer,
                    beam_size=1,  # Use beam_size=1 for speed
                    vad_filter=False  # We're doing VAD ourselves
                )

                if english_text and english_text.strip():
                    # Check if this looks like a sentence ending
                    is_sentence_end = (
                        english_text.rstrip().endswith(('.', '!', '?')) or
                        duration > 3.0  # Force finalization after 3 seconds
                    )

                    if is_sentence_end:
                        # Complete sentence detected
                        self.english_buffer = english_text
                        self._finalize_sentence()
                        self.audio_buffer = np.array([], dtype=np.float32)
                    else:
                        # Partial sentence - update but don't translate yet
                        self.english_buffer = english_text
                        if self.on_english_text:
                            self.on_english_text(f"[Partial] {english_text}")

            except Empty:
                # Check if we have pending text to finalize
                if len(self.english_buffer) > 0:
                    buffer_duration = len(self.audio_buffer) / 16000
                    if buffer_duration > 2.0:  # Finalize after 2s of accumulation
                        self._finalize_sentence()
                        self.audio_buffer = np.array([], dtype=np.float32)
                continue
            except Exception as e:
                print(f"[ASR Worker] Error: {e}")

        print("[ASR Worker] Stopped")

    def _finalize_sentence(self) -> None:
        """Finalize and queue English sentence for translation"""
        if not self.english_buffer or not self.english_buffer.strip():
            return

        # Send to translation queue
        timestamp = time.time()
        self.translation_queue.put((self.english_buffer, timestamp))

        # Notify callback
        if self.on_english_text:
            self.on_english_text(self.english_buffer)

        # Clear buffer
        self.english_buffer = ""

    def _translation_worker(self) -> None:
        """
        Background worker for translation and TTS
        Translates English text to German and queues for playback
        """
        print("[Translation Worker] Started")

        while not self.stop_event.is_set():
            try:
                # Get English text with timeout
                english_text, timestamp = self.translation_queue.get(timeout=0.1)

                # Translate to German with higher beam size for better quality
                german_text = self.translator.translate(english_text, num_beams=5)

                # Notify callback
                if self.on_german_text:
                    self.on_german_text(german_text)

                # Queue for TTS playback
                self.tts.queue_text(german_text)

                # Track latency
                latency = time.time() - timestamp
                self.total_latency += latency
                self.segments_processed += 1

                print(f"[Translation] EN: {english_text}")
                print(f"[Translation] DE: {german_text} (latency: {latency:.2f}s)")

            except Empty:
                continue
            except Exception as e:
                print(f"[Translation Worker] Error: {e}")

        print("[Translation Worker] Stopped")

    def get_stats(self) -> dict:
        """Get performance statistics"""
        avg_latency = (
            self.total_latency / self.segments_processed
            if self.segments_processed > 0
            else 0.0
        )

        return {
            "segments_processed": self.segments_processed,
            "average_latency_s": avg_latency,
            "is_running": self.is_running,
            "audio_queue_size": self.audio_queue.qsize(),
            "translation_queue_size": self.translation_queue.qsize(),
            "tts_queue_size": self.tts.playback_queue.qsize()
        }

    def unload(self) -> None:
        """Unload all models and free resources"""
        self.stop()
        self.whisper.unload()
        self.translator.unload()
        self.tts.unload()
        print("Streaming engine unloaded")


def test_streaming_translation():
    """Test streaming translation with live microphone"""
    import sounddevice as sd

    print("Testing streaming translation...")
    print("This will use your microphone for live English→German translation")
    print("Speak English sentences and hear them translated to German in real-time!")

    # Create engine
    engine = StreamingTranslationEngine(
        whisper_model_size="small",
        device="cuda"
    )

    # Set up callbacks for visual feedback
    def on_english(text: str):
        print(f"\n🗣️  YOU (EN): {text}")

    def on_german(text: str):
        print(f"🔊 AI (DE): {text}\n")

    engine.on_english_text = on_english
    engine.on_german_text = on_german

    # Start engine
    if not engine.start():
        print("Failed to start engine")
        return

    # Set up audio capture
    sample_rate = 16000
    chunk_size = int(sample_rate * 0.25)  # 250ms chunks

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")

        # Send audio to engine
        audio_data = indata[:, 0].copy().astype(np.float32)
        engine.process_audio(audio_data)

    # Start audio stream
    print("\n🎤 Starting microphone... Speak English!\n")
    print("Press Ctrl+C to stop\n")

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

    # Stop and show stats
    engine.stop()
    stats = engine.get_stats()
    print(f"\nSession Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    engine.unload()


if __name__ == "__main__":
    test_streaming_translation()
