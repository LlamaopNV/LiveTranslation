"""
Audio Device Management
Handles enumeration and selection of audio input devices
"""

import sounddevice as sd
from typing import List, Dict, Optional


class AudioDeviceManager:
    """Manages audio input device discovery and selection"""

    def __init__(self):
        self.devices: List[Dict] = []
        self.selected_device: Optional[int] = None
        self._refresh_devices()

    def _refresh_devices(self) -> None:
        """Refresh the list of available audio devices"""
        self.devices = []
        device_list = sd.query_devices()

        for idx, device in enumerate(device_list):
            # Only include input devices (with at least 1 input channel)
            if device['max_input_channels'] > 0:
                self.devices.append({
                    'id': idx,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate'],
                    'host_api': device['hostapi']
                })

    def get_devices(self) -> List[Dict]:
        """
        Get list of available input devices

        Returns:
            List of device dictionaries with id, name, channels, sample_rate
        """
        return self.devices

    def get_device_names(self) -> List[str]:
        """Get list of device names for UI display"""
        return [device['name'] for device in self.devices]

    def select_device(self, device_id: int) -> bool:
        """
        Select an audio device by ID

        Args:
            device_id: Device ID to select

        Returns:
            True if device was successfully selected
        """
        # Validate device ID
        device_ids = [d['id'] for d in self.devices]
        if device_id not in device_ids:
            return False

        self.selected_device = device_id
        return True

    def select_device_by_name(self, device_name: str) -> bool:
        """
        Select an audio device by name

        Args:
            device_name: Device name to select

        Returns:
            True if device was successfully selected
        """
        for device in self.devices:
            if device['name'] == device_name:
                self.selected_device = device['id']
                return True
        return False

    def get_selected_device(self) -> Optional[Dict]:
        """Get the currently selected device info"""
        if self.selected_device is None:
            return None

        for device in self.devices:
            if device['id'] == self.selected_device:
                return device
        return None

    def get_default_device(self) -> Optional[Dict]:
        """Get the system default input device"""
        try:
            default_id = sd.default.device[0]  # Index 0 is input device
            for device in self.devices:
                if device['id'] == default_id:
                    return device
        except Exception:
            pass

        # If no default found, return first available device
        if self.devices:
            return self.devices[0]
        return None

    def select_default_device(self) -> bool:
        """Select the system default input device"""
        default = self.get_default_device()
        if default:
            self.selected_device = default['id']
            return True
        return False

    def test_device(self, device_id: Optional[int] = None) -> bool:
        """
        Test if a device can be opened for recording

        Args:
            device_id: Device to test, or None to test selected device

        Returns:
            True if device can be opened successfully
        """
        test_id = device_id if device_id is not None else self.selected_device
        if test_id is None:
            return False

        try:
            # Try to open a stream briefly to test
            with sd.InputStream(device=test_id, channels=1,
                              samplerate=16000, blocksize=1024):
                pass
            return True
        except Exception as e:
            print(f"Device test failed: {e}")
            return False

    def refresh(self) -> None:
        """Refresh the device list (call if devices are plugged/unplugged)"""
        self._refresh_devices()


# Convenience function for quick device listing
def list_audio_devices() -> None:
    """Print all available audio input devices"""
    manager = AudioDeviceManager()
    devices = manager.get_devices()

    print("Available Audio Input Devices:")
    print("-" * 60)
    for device in devices:
        print(f"ID: {device['id']}")
        print(f"  Name: {device['name']}")
        print(f"  Channels: {device['channels']}")
        print(f"  Sample Rate: {device['sample_rate']} Hz")
        print()

    default = manager.get_default_device()
    if default:
        print(f"Default Device: {default['name']} (ID: {default['id']})")


if __name__ == "__main__":
    # Test the device manager
    list_audio_devices()
