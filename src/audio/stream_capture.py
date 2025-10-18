"""
Real-time Audio Stream Capture
Handles continuous audio streaming with buffering and chunking
"""

import sounddevice as sd
import numpy as np
from queue import Queue, Empty
from typing import Optional, Callable
import threading


class AudioStreamCapture:
    """
    Captures audio from microphone in real-time with buffering
    Optimized for low-latency streaming translation
    """

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 2.0,
        overlap_duration: float = 0.5
    ):
        """
        Initialize audio stream capture

        Args:
            device_id: Audio device ID (None for default)
            sample_rate: Sample rate in Hz (16000 optimal for Whisper)
            channels: Number of audio channels (1 for mono)
            chunk_duration: Duration of each audio chunk in seconds
            overlap_duration: Overlap between chunks for smoother processing
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration

        # Calculate samples
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.overlap_samples = int(sample_rate * overlap_duration)

        # Audio buffer
        self.audio_queue: Queue = Queue()
        self.buffer = np.array([], dtype=np.float32)

        # Stream state
        self.stream: Optional[sd.InputStream] = None
        self.is_streaming = False
        self.callback_func: Optional[Callable] = None

        # Threading
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback for audio stream (called by sounddevice)

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Status flags
        """
        if status:
            print(f"Audio stream status: {status}")

        # Convert to mono if needed and add to queue
        audio_data = indata[:, 0] if self.channels == 1 else indata.mean(axis=1)
        self.audio_queue.put(audio_data.copy())

    def start(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """
        Start audio capture stream

        Args:
            callback: Optional callback function to call with each audio chunk

        Returns:
            True if stream started successfully
        """
        if self.is_streaming:
            print("Stream already running")
            return False

        self.callback_func = callback
        self.stop_event.clear()

        try:
            # Open audio stream
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            )

            self.stream.start()
            self.is_streaming = True

            # Start processing thread if callback provided
            if self.callback_func:
                self.processing_thread = threading.Thread(
                    target=self._processing_loop,
                    daemon=True
                )
                self.processing_thread.start()

            print(f"Audio stream started (device: {self.device_id}, "
                  f"sample_rate: {self.sample_rate} Hz)")
            return True

        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            return False

    def stop(self) -> None:
        """Stop audio capture stream"""
        if not self.is_streaming:
            return

        self.is_streaming = False
        self.stop_event.set()

        # Stop the stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None

        # Clear buffers
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break

        self.buffer = np.array([], dtype=np.float32)
        print("Audio stream stopped")

    def _processing_loop(self) -> None:
        """
        Background thread that processes audio chunks
        Calls the callback function with each complete chunk
        """
        while not self.stop_event.is_set():
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.1)

                # Add to buffer
                self.buffer = np.concatenate([self.buffer, audio_data])

                # Process complete chunks
                while len(self.buffer) >= self.chunk_samples:
                    # Extract chunk
                    chunk = self.buffer[:self.chunk_samples]

                    # Call callback with chunk
                    if self.callback_func:
                        try:
                            self.callback_func(chunk)
                        except Exception as e:
                            print(f"Callback error: {e}")

                    # Remove processed samples, keep overlap
                    self.buffer = self.buffer[self.chunk_samples - self.overlap_samples:]

            except Empty:
                continue
            except Exception as e:
                print(f"Processing loop error: {e}")

    def get_next_chunk(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Get next audio chunk (blocking)
        Use this if not using callback mode

        Args:
            timeout: Maximum time to wait for chunk in seconds

        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        if not self.is_streaming:
            return None

        start_buffer_len = len(self.buffer)
        import time
        start_time = time.time()

        # Collect data until we have a full chunk
        while len(self.buffer) < self.chunk_samples:
            if time.time() - start_time > timeout:
                return None

            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                self.buffer = np.concatenate([self.buffer, audio_data])
            except Empty:
                continue

        # Extract chunk
        chunk = self.buffer[:self.chunk_samples]

        # Remove processed samples, keep overlap
        self.buffer = self.buffer[self.chunk_samples - self.overlap_samples:]

        return chunk

    def get_current_buffer(self) -> np.ndarray:
        """Get current audio buffer (for visualization, etc.)"""
        return self.buffer.copy()

    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        self.buffer = np.array([], dtype=np.float32)
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break


def test_audio_capture():
    """Test function for audio capture"""
    import time

    def on_audio_chunk(chunk: np.ndarray):
        # Calculate RMS volume
        rms = np.sqrt(np.mean(chunk**2))
        print(f"Received chunk: {len(chunk)} samples, RMS: {rms:.4f}")

    print("Testing audio capture for 10 seconds...")
    capture = AudioStreamCapture(chunk_duration=2.0)

    if capture.start(callback=on_audio_chunk):
        print("Recording... speak into your microphone!")
        time.sleep(10)
        capture.stop()
        print("Test complete")
    else:
        print("Failed to start capture")


if __name__ == "__main__":
    test_audio_capture()
