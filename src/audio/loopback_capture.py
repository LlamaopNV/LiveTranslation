"""
System Audio Loopback Capture
Captures audio playing through system output devices
"""

import numpy as np
import pyaudiowpatch as pyaudio
from typing import Optional, Callable
import threading
import queue


class LoopbackCapture:
    """
    Captures system audio using WASAPI loopback
    This can capture audio from any output device (speakers, etc.)
    """

    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize loopback capture

        Args:
            device_id: Device ID to capture from (None = default output)
        """
        self.device_id = device_id
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_capturing = False
        self.audio_queue = queue.Queue()
        self.callback: Optional[Callable[[np.ndarray], None]] = None

    def get_loopback_devices(self):
        """Get all output devices that support loopback capture"""
        devices = []

        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)

            # Check if it's a loopback device
            # In PyAudioWPatch, loopback devices are marked
            if device_info['maxOutputChannels'] > 0:
                # This is an output device - we can capture from it with loopback
                devices.append({
                    'id': i,
                    'name': device_info['name'],
                    'channels': device_info['maxOutputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                })

        return devices

    def start_capture(self, callback: Optional[Callable[[np.ndarray], None]] = None):
        """
        Start capturing system audio

        Args:
            callback: Function to call with audio chunks (numpy arrays)
        """
        if self.is_capturing:
            return

        self.callback = callback
        self.is_capturing = True

        # Get the loopback device for the specified device
        if self.device_id is not None:
            # User specified a device - find its loopback version
            device_info = self.p.get_device_info_by_index(self.device_id)

            if device_info.get('isLoopbackDevice', False):
                # Already a loopback device
                print(f"Using loopback device: {device_info['name']}")
            else:
                # Need to find the loopback version
                print(f"Finding loopback for: {device_info['name']}")
                original_name = device_info['name']

                # Try to find loopback device with matching name
                loopback_found = False
                for i in range(self.p.get_device_count()):
                    info = self.p.get_device_info_by_index(i)
                    if info.get('isLoopbackDevice', False):
                        # Check if names match (loopback usually has same name with [Loopback] suffix)
                        if original_name in info['name'] or info['name'] in original_name:
                            print(f"Found loopback device: {info['name']}")
                            self.device_id = i
                            loopback_found = True
                            break

                if not loopback_found:
                    print(f"⚠️  Could not find loopback device for {original_name}")
                    print(f"   Using original device and hoping for the best...")

        # Get default loopback device if none specified
        if self.device_id is None:
            try:
                # Get default output device
                default_output = self.p.get_default_output_device_info()

                # Find the loopback version
                print(f"Looking for loopback device for: {default_output['name']}")
                for i in range(self.p.get_device_count()):
                    info = self.p.get_device_info_by_index(i)
                    if info.get('isLoopbackDevice', False) and default_output['name'] in info['name']:
                        self.device_id = i
                        break
            except Exception as e:
                print(f"Error finding loopback device: {e}")
                return

        # Get device info
        device_info = self.p.get_device_info_by_index(self.device_id)
        print(f"Capturing from: {device_info['name']}")

        # Use device's native sample rate and channels
        sample_rate = int(device_info['defaultSampleRate'])
        channels = device_info['maxOutputChannels'] if device_info['maxOutputChannels'] > 0 else device_info['maxInputChannels']
        # Clamp to 1-2 channels
        channels = max(1, min(2, channels))

        print(f"Using sample rate: {sample_rate} Hz")
        print(f"Using channels: {channels}")

        # Store channels for callback
        self.channels = channels

        # Open stream in loopback mode
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=4096,
                stream_callback=self._audio_callback
            )

            self.stream.start_stream()
            print("✓ Loopback capture started")

        except Exception as e:
            print(f"Error starting loopback capture: {e}")
            self.is_capturing = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for stream"""
        if status:
            print(f"Loopback status: {status}")

        if self.is_capturing:
            # Convert bytes to numpy array (keep as int16 initially)
            audio_data = np.frombuffer(in_data, dtype=np.int16)

            # Convert to mono if stereo (BEFORE converting to float for efficiency)
            if hasattr(self, 'channels') and self.channels == 2 and len(audio_data) > 0:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)

            # Now convert to float32
            audio_data = audio_data.astype(np.float32) / 32768.0

            # Add to queue
            self.audio_queue.put(audio_data)

            # Call user callback if provided
            if self.callback:
                self.callback(audio_data)

        return (in_data, pyaudio.paContinue)

    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get next audio chunk from queue

        Args:
            timeout: Timeout in seconds

        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop_capture(self):
        """Stop capturing"""
        self.is_capturing = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def __del__(self):
        """Cleanup"""
        self.stop_capture()
        if hasattr(self, 'p'):
            self.p.terminate()


def test_loopback():
    """Test loopback capture"""
    print("Testing system audio loopback capture...")

    capture = LoopbackCapture()

    # List devices
    devices = capture.get_loopback_devices()
    print("\nAvailable output devices for loopback capture:")
    for dev in devices:
        print(f"  ID {dev['id']}: {dev['name']}")

    # Start capture
    print("\nStarting capture from default output device...")
    print("Play some audio (music, video, etc.)")

    captured_chunks = []

    def callback(audio_data):
        captured_chunks.append(audio_data)
        # Print level meter
        level = np.abs(audio_data).max()
        bars = int(level * 50)
        print(f"\rLevel: {'█' * bars}{' ' * (50 - bars)} {level:.3f}", end='')

    capture.start_capture(callback)

    # Capture for 10 seconds
    import time
    time.sleep(10)

    capture.stop_capture()

    print(f"\n\nCaptured {len(captured_chunks)} chunks")
    print(f"Total samples: {sum(len(c) for c in captured_chunks)}")

    if len(captured_chunks) > 0:
        print("✓ Loopback capture working!")
    else:
        print("✗ No audio captured - make sure audio is playing")


if __name__ == "__main__":
    test_loopback()
