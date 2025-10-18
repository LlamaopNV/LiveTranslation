"""
Main Application Window
Clean GUI for live translation demonstration
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QTextEdit, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont, QColor, QPalette
from typing import Optional
import numpy as np


class TranslationWorker(QObject):
    """Worker thread for running translation engine"""

    english_text = pyqtSignal(str)
    german_text = pyqtSignal(str)
    error = pyqtSignal(str)
    started = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self, device_id: Optional[int], model_size: str = "small"):
        super().__init__()
        self.device_id = device_id
        self.model_size = model_size
        self.engine = None
        self.is_running = False
        self.is_recording = False
        self.audio_buffer = []
        self.save_recordings = False
        self.recording_counter = 0

    def start_translation(self):
        """Start the translation engine"""
        try:
            from ..translation.streaming_translator import StreamingTranslationEngine

            # Initialize engine
            self.engine = StreamingTranslationEngine(
                whisper_model_size=self.model_size,
                device="cuda",
                min_chunk_duration=0.3
            )

            # Set up callbacks
            self.engine.on_english_text = lambda text: self.english_text.emit(text)
            self.engine.on_german_text = lambda text: self.german_text.emit(text)

            # Start engine
            if self.engine.start():
                self.is_running = True
                self.started.emit()

                # Start audio capture
                self._capture_audio()
            else:
                self.error.emit("Failed to start translation engine")

        except Exception as e:
            self.error.emit(f"Engine error: {str(e)}")

    def _capture_audio(self):
        """Capture audio and feed to engine"""
        import sounddevice as sd

        sample_rate = 16000
        chunk_size = int(sample_rate * 0.25)  # 250ms chunks

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")

            if self.is_running and self.is_recording and self.engine:
                audio_data = indata[:, 0].copy().astype(np.float32)
                self.audio_buffer.append(audio_data)

        # Open audio stream
        self.stream = sd.InputStream(
            device=self.device_id,
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size,
            callback=audio_callback
        )

        self.stream.start()

    def enable_recording(self):
        """Start recording audio (PTT button pressed)"""
        self.is_recording = True
        self.audio_buffer = []

    def disable_recording(self):
        """Stop recording and process audio (PTT button released)"""
        self.is_recording = False

        if len(self.audio_buffer) > 0 and self.engine:
            # Concatenate all audio chunks
            full_audio = np.concatenate(self.audio_buffer)

            # Save English audio if enabled
            if self.save_recordings:
                self._save_english_audio(full_audio)

            # Process the complete audio
            self.engine.process_audio(full_audio)
            self.audio_buffer = []

    def _save_english_audio(self, audio_data):
        """Save English audio to WAV"""
        try:
            from scipy.io import wavfile
            import os
            from datetime import datetime

            # Create recordings directory
            os.makedirs("recordings", exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_counter += 1
            wav_path = f"recordings/english_{timestamp}_{self.recording_counter}.wav"

            # Save as WAV
            wavfile.write(wav_path, 16000, (audio_data * 32767).astype(np.int16))

            print(f"✓ Saved English audio: {wav_path}")

        except Exception as e:
            print(f"Error saving English audio: {e}")

    def set_save_recordings(self, enabled: bool):
        """Enable/disable recording saving"""
        self.save_recordings = enabled

    def stop_translation(self):
        """Stop the translation engine"""
        self.is_running = False

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        if self.engine:
            self.engine.stop()
            self.engine = None

        self.stopped.emit()


class LiveTranslationWindow(QMainWindow):
    """
    Main application window for live translation
    Clean, professional interface for interview demonstration
    """

    def __init__(self):
        super().__init__()
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[TranslationWorker] = None
        self.is_translating = False

        self.init_ui()
        self.load_audio_devices()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Live Translation - English → German")
        self.setGeometry(100, 100, 900, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header_label = QLabel("🎤 Live Speech Translation")
        header_font = QFont("Arial", 20, QFont.Weight.Bold)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        subtitle = QLabel("English → German | Real-time Translation")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 12pt;")
        layout.addWidget(subtitle)

        layout.addSpacing(10)

        # Device selection
        device_group = QGroupBox("Audio Input")
        device_layout = QVBoxLayout()

        device_label = QLabel("Select Microphone:")
        self.device_combo = QComboBox()
        self.device_combo.setMinimumHeight(35)

        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Push-to-Talk button
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.ptt_button = QPushButton("🎤 HOLD TO SPEAK")
        self.ptt_button.setMinimumHeight(80)
        self.ptt_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 18pt;
                font-weight: bold;
                border-radius: 12px;
                border: 3px solid #1976D2;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #4CAF50;
                border: 3px solid #388E3C;
            }
        """)
        self.ptt_button.pressed.connect(self.start_speaking)
        self.ptt_button.released.connect(self.stop_speaking)

        button_layout.addWidget(self.ptt_button)
        layout.addLayout(button_layout)

        # Save recordings checkbox
        self.save_checkbox = QCheckBox("💾 Save recordings to WAV (English input + German output)")
        self.save_checkbox.setStyleSheet("font-size: 11pt; padding: 5px;")
        self.save_checkbox.setChecked(False)
        self.save_checkbox.stateChanged.connect(self.on_save_checkbox_changed)
        layout.addWidget(self.save_checkbox)

        # Status indicator
        self.status_label = QLabel("⚪ Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.status_label)

        # Translation display
        display_layout = QHBoxLayout()
        display_layout.setSpacing(10)

        # English text box
        english_group = QGroupBox("🗣️ You Speak (English)")
        english_layout = QVBoxLayout()
        self.english_text = QTextEdit()
        self.english_text.setReadOnly(True)
        self.english_text.setMinimumHeight(200)
        self.english_text.setStyleSheet("""
            QTextEdit {
                background-color: #e3f2fd;
                color: #000000;
                font-size: 13pt;
                border: 2px solid #2196F3;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        english_layout.addWidget(self.english_text)
        english_group.setLayout(english_layout)
        display_layout.addWidget(english_group)

        # German text box
        german_group = QGroupBox("🔊 AI Speaks (German)")
        german_layout = QVBoxLayout()
        self.german_text = QTextEdit()
        self.german_text.setReadOnly(True)
        self.german_text.setMinimumHeight(200)
        self.german_text.setStyleSheet("""
            QTextEdit {
                background-color: #e8f5e9;
                color: #000000;
                font-size: 13pt;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        german_layout.addWidget(self.german_text)
        german_group.setLayout(german_layout)
        display_layout.addWidget(german_group)

        layout.addLayout(display_layout)

        # Footer with instructions
        instructions = QLabel(
            "💡 HOLD the button while speaking in English. Release when done. Translation happens in real-time!"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; font-size: 10pt; padding: 10px;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        central_widget.setLayout(layout)

        # Initialize translation worker on startup
        self.init_worker()

    def load_audio_devices(self):
        """Load available audio input devices"""
        from ..audio.device_manager import AudioDeviceManager

        manager = AudioDeviceManager()
        devices = manager.get_devices()

        for device in devices:
            # Show full name with ID to distinguish duplicates
            display_name = f"[ID {device['id']}] {device['name']}"
            self.device_combo.addItem(display_name, device['id'])

        # Select default device
        default = manager.get_default_device()
        if default:
            index = self.device_combo.findData(default['id'])
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

    def init_worker(self):
        """Initialize translation worker on startup"""
        device_id = self.device_combo.currentData()

        # Create worker thread
        self.worker_thread = QThread()
        self.worker = TranslationWorker(device_id, model_size="medium")
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.english_text.connect(self.on_english_text)
        self.worker.german_text.connect(self.on_german_text)
        self.worker.error.connect(self.on_error)
        self.worker.started.connect(self.on_started)
        self.worker.stopped.connect(self.on_stopped)

        self.worker_thread.started.connect(self.worker.start_translation)

        # Start thread (models load in background)
        self.worker_thread.start()

    def start_speaking(self):
        """Called when PTT button is pressed"""
        if not self.is_translating:
            return

        self.status_label.setText("🔴 RECORDING...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #ffcdd2;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)

        # Tell worker to start recording
        if self.worker:
            self.worker.enable_recording()

    def stop_speaking(self):
        """Called when PTT button is released"""
        if not self.is_translating:
            return

        self.status_label.setText("🟢 Processing...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #c8e6c9;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)

        # Tell worker to stop recording and process
        if self.worker:
            self.worker.disable_recording()

    def on_english_text(self, text: str):
        """Handle English text update"""
        if text.startswith("[Partial]"):
            # Show partial text in lighter color
            self.english_text.setHtml(f'<span style="color: #999;">{text[9:]}</span>')
        else:
            # Final text
            current = self.english_text.toPlainText()
            if current:
                self.english_text.append("\n" + text)
            else:
                self.english_text.setText(text)

    def on_german_text(self, text: str):
        """Handle German text update"""
        current = self.german_text.toPlainText()
        if current:
            self.german_text.append("\n" + text)
        else:
            self.german_text.setText(text)

    def on_error(self, error_msg: str):
        """Handle error"""
        self.status_label.setText(f"🔴 Error: {error_msg}")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #ffcdd2;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.device_combo.setEnabled(True)
        self.is_translating = False

    def on_started(self):
        """Handle translation started"""
        self.is_translating = True
        self.status_label.setText("🟢 Ready - Hold button to speak")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #c8e6c9;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)

    def on_stopped(self):
        """Handle translation stopped"""
        self.is_translating = False
        self.status_label.setText("⚪ Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.device_combo.setEnabled(True)

        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

    def on_save_checkbox_changed(self, state):
        """Handle save checkbox state change"""
        enabled = state == Qt.CheckState.Checked.value

        if self.worker:
            # Enable for English audio saving
            self.worker.set_save_recordings(enabled)

            # Enable for German TTS audio saving
            if hasattr(self.worker, 'engine') and self.worker.engine:
                if hasattr(self.worker.engine, 'tts') and self.worker.engine.tts:
                    self.worker.engine.tts.set_save_recordings(enabled)
                    print(f"✓ TTS recording {'enabled' if enabled else 'disabled'}")

        if enabled:
            print("✓ Recording saving enabled - files will be saved to 'recordings/' folder")
        else:
            print("✗ Recording saving disabled")

    def closeEvent(self, event):
        """Handle window close"""
        if self.is_translating:
            self.stop_translation()
            if self.worker_thread:
                self.worker_thread.quit()
                self.worker_thread.wait()
        event.accept()
