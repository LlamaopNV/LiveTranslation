"""
Dual Translation Window - Bidirectional Real-time Translation
Supports simultaneous English→German and German→English translation
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QTextEdit, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont
from typing import Optional
import numpy as np


class DualTranslationWorker(QObject):
    """Worker that runs BOTH translation engines simultaneously"""

    english_text = pyqtSignal(str)
    german_text = pyqtSignal(str)
    error = pyqtSignal(str)
    started = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self, input_device_id: Optional[int], output_device_id: Optional[int], tts_output_id: Optional[int] = None, model_size: str = "large-v3"):
        super().__init__()
        self.input_device_id = input_device_id    # Microphone
        self.output_device_id = output_device_id  # System audio (for capturing)
        self.tts_output_id = tts_output_id        # TTS playback device
        self.model_size = model_size

        # Two engines
        self.engine_en_to_de = None
        self.engine_de_to_en = None

        # State
        self.is_running = False
        self.is_recording_en = False
        self.is_recording_de = False

        # Buffers
        self.audio_buffer_en = []
        self.audio_buffer_de = []

        # Recording
        self.save_recordings = False
        self.recording_counter = 0

    def start_engines(self):
        """Load and start both translation engines"""
        try:
            from ..translation.bidirectional_translator import BidirectionalTranslationEngine
            from ..translation.whisper_engine import WhisperEngine

            # Load shared Whisper model ONCE (saves ~5-6GB VRAM)
            print("🔄 Loading shared Whisper model...")
            shared_whisper = WhisperEngine(model_size=self.model_size, device="cuda")
            print(f"   ✓ Whisper {self.model_size} loaded (~5-6GB VRAM)")

            print("🔄 Loading English → German engine...")
            self.engine_en_to_de = BidirectionalTranslationEngine(
                whisper_model_size=self.model_size,
                device="cuda",
                mode="en_to_de",
                shared_whisper=shared_whisper,  # Share the Whisper instance
                tts_output_device=self.tts_output_id  # Route German TTS to this device
            )
            self.engine_en_to_de.on_source_text = lambda text: self.english_text.emit(f"[You] {text}")
            self.engine_en_to_de.on_target_text = lambda text: self.german_text.emit(f"[You] {text}")

            print("🔄 Loading German → English engine...")
            self.engine_de_to_en = BidirectionalTranslationEngine(
                whisper_model_size=self.model_size,
                device="cuda",
                mode="de_to_en",
                shared_whisper=shared_whisper  # Share the same Whisper instance
            )
            self.engine_de_to_en.on_source_text = lambda text: self.german_text.emit(f"[Them] {text}")
            self.engine_de_to_en.on_target_text = lambda text: self.english_text.emit(f"[Them] {text}")

            # Start both
            if self.engine_en_to_de.start() and self.engine_de_to_en.start():
                self.is_running = True
                self.started.emit()

                # Start audio capture
                self._start_audio_capture()
            else:
                self.error.emit("Failed to start engines")

        except Exception as e:
            self.error.emit(f"Engine error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _start_audio_capture(self):
        """Start capturing from both input and output devices"""
        import sounddevice as sd
        from ..audio.loopback_capture import LoopbackCapture
        import scipy.signal

        sample_rate = 16000
        chunk_size = int(sample_rate * 0.25)

        # Microphone callback (English input)
        def input_callback(indata, frames, time_info, status):
            if status:
                print(f"Input: {status}")
            if self.is_running and self.is_recording_en:
                self.audio_buffer_en.append(indata[:, 0].copy().astype(np.float32))

        # Open input stream (microphone)
        self.input_stream = sd.InputStream(
            device=self.input_device_id,
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size,
            callback=input_callback
        )
        self.input_stream.start()
        print(f"✓ Microphone capture started (device {self.input_device_id})")

        # Open output stream using loopback capture (if device selected)
        if self.output_device_id is not None:
            try:
                self.loopback = LoopbackCapture(device_id=self.output_device_id)

                # Get device native sample rate for resampling
                import pyaudiowpatch as pyaudio
                p = pyaudio.PyAudio()
                device_info = p.get_device_info_by_index(self.output_device_id)
                native_rate = int(device_info['defaultSampleRate'])
                p.terminate()

                # Callback for loopback audio
                def loopback_callback(audio_data):
                    if self.is_running and self.is_recording_de:
                        # Resample to 16kHz if needed
                        if native_rate != 16000:
                            # Use faster polyphase resampling
                            from scipy.signal import resample_poly
                            # Calculate GCD for optimal downsampling
                            from math import gcd
                            g = gcd(16000, native_rate)
                            up = 16000 // g
                            down = native_rate // g
                            audio_data = resample_poly(audio_data, up, down)

                        self.audio_buffer_de.append(audio_data.astype(np.float32))

                self.loopback.start_capture(callback=loopback_callback)
                print(f"✓ System audio loopback started on device {self.output_device_id}")
                print(f"   Native rate: {native_rate} Hz → 16000 Hz")

            except Exception as e:
                print(f"⚠ Could not start loopback capture: {e}")
                import traceback
                traceback.print_exc()
                self.loopback = None

    def start_recording_english(self):
        """Start recording English (PTT pressed)"""
        self.is_recording_en = True
        self.audio_buffer_en = []
        print("🎤 Recording English...")

    def stop_recording_english(self):
        """Stop recording English and process"""
        self.is_recording_en = False
        if len(self.audio_buffer_en) > 0 and self.engine_en_to_de:
            audio = np.concatenate(self.audio_buffer_en)
            if self.save_recordings:
                self._save_audio(audio, "english_input")
            self.engine_en_to_de.process_audio(audio)
            self.audio_buffer_en = []

    def start_recording_german(self):
        """Start recording German (PTT pressed)"""
        self.is_recording_de = True
        self.audio_buffer_de = []
        print("🎧 Recording German...")

    def stop_recording_german(self):
        """Stop recording German and process"""
        self.is_recording_de = False

        print(f"🛑 Stopped recording German")
        print(f"   Buffer chunks: {len(self.audio_buffer_de)}")

        if len(self.audio_buffer_de) > 0:
            audio = np.concatenate(self.audio_buffer_de)
            print(f"   Total samples: {len(audio)}")
            print(f"   Duration: {len(audio) / 16000:.2f}s")
            print(f"   Max amplitude (pre-norm): {np.abs(audio).max():.4f}")

            # Normalize audio to improve detection quality
            max_amp = np.abs(audio).max()
            if max_amp > 0.01:  # Only normalize if there's actual audio
                audio = audio / max_amp * 0.95  # Normalize to 95% to avoid clipping
                print(f"   Max amplitude (post-norm): {np.abs(audio).max():.4f}")
            else:
                print("   ⚠️ Audio too quiet, skipping")
                self.audio_buffer_de = []
                return

            if self.save_recordings:
                self._save_audio(audio, "german_input")

            if self.engine_de_to_en:
                self.engine_de_to_en.process_audio(audio)
            else:
                print("⚠️  DE→EN engine not initialized!")

            self.audio_buffer_de = []
        else:
            print("⚠️  No audio captured! Check:")
            print("   1. Output device is selected")
            print("   2. Audio is playing through that device")
            print("   3. Loopback capture started successfully")

    def _save_audio(self, audio_data, prefix):
        """Save audio to WAV"""
        try:
            from scipy.io import wavfile
            import os
            from datetime import datetime

            os.makedirs("recordings", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_counter += 1
            path = f"recordings/{prefix}_{timestamp}_{self.recording_counter}.wav"
            wavfile.write(path, 16000, (audio_data * 32767).astype(np.int16))
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"Error saving audio: {e}")

    def set_save_recordings(self, enabled: bool):
        """Toggle recording save"""
        self.save_recordings = enabled

    def stop_engines(self):
        """Stop everything"""
        self.is_running = False

        # Stop input stream
        if hasattr(self, 'input_stream'):
            self.input_stream.stop()
            self.input_stream.close()

        # Stop loopback capture
        if hasattr(self, 'loopback') and self.loopback:
            self.loopback.stop_capture()

        # Stop engines
        if self.engine_en_to_de:
            self.engine_en_to_de.stop()
        if self.engine_de_to_en:
            self.engine_de_to_en.stop()

        self.stopped.emit()


class DualTranslationWindow(QMainWindow):
    """
    Dual Translation Window
    Two simultaneous translation modes:
    - English (mic) → German (output)
    - German (system audio) → English (display)
    """

    def __init__(self):
        super().__init__()
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[DualTranslationWorker] = None
        self.is_translating = False

        self.init_ui()
        self.load_audio_devices()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Live Translation - Bidirectional")
        self.setGeometry(100, 100, 1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("🎤 Bidirectional Live Translation")
        header.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        subtitle = QLabel("English ↔ German | Real-time Conversation")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 12pt;")
        layout.addWidget(subtitle)

        layout.addSpacing(10)

        # Device Selection
        device_group = QGroupBox("Audio Devices")
        device_layout = QVBoxLayout()

        # Input device (microphone)
        input_label = QLabel("🎤 Input Device (Your Microphone):")
        self.input_combo = QComboBox()
        self.input_combo.setMinimumHeight(35)

        # Output device (system audio for listening)
        output_label = QLabel("🔊 Capture Device (System Audio - for capturing German from Google Meet):")
        self.output_combo = QComboBox()
        self.output_combo.setMinimumHeight(35)
        self.output_combo.addItem("None (Disabled)", None)  # Default option

        # TTS output device (where German translation plays)
        tts_output_label = QLabel("🎵 German TTS Output (Route to Google Meet microphone):")
        self.tts_output_combo = QComboBox()
        self.tts_output_combo.setMinimumHeight(35)
        self.tts_output_combo.addItem("Default Audio Device", None)  # Default option

        device_layout.addWidget(input_label)
        device_layout.addWidget(self.input_combo)
        device_layout.addSpacing(10)
        device_layout.addWidget(output_label)
        device_layout.addWidget(self.output_combo)
        device_layout.addSpacing(10)
        device_layout.addWidget(tts_output_label)
        device_layout.addWidget(self.tts_output_combo)

        # Apply devices button
        self.apply_devices_btn = QPushButton("✅ Apply Device Changes")
        self.apply_devices_btn.setMinimumHeight(40)
        self.apply_devices_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 8px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.apply_devices_btn.clicked.connect(self.on_apply_devices)
        device_layout.addWidget(self.apply_devices_btn)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Push-to-Talk Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        # English → German button
        self.ptt_en_button = QPushButton("🇺🇸 SPEAK ENGLISH\n(Hold to translate to German)")
        self.ptt_en_button.setMinimumHeight(100)
        self.ptt_en_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14pt;
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
        self.ptt_en_button.pressed.connect(self.start_english)
        self.ptt_en_button.released.connect(self.stop_english)

        # German → English button
        self.ptt_de_button = QPushButton("🇩🇪 LISTEN TO GERMAN\n(Hold to capture & translate)")
        self.ptt_de_button.setMinimumHeight(100)
        self.ptt_de_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                border-radius: 12px;
                border: 3px solid #F57C00;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #4CAF50;
                border: 3px solid #388E3C;
            }
        """)
        self.ptt_de_button.pressed.connect(self.start_german)
        self.ptt_de_button.released.connect(self.stop_german)

        button_layout.addWidget(self.ptt_en_button)
        button_layout.addWidget(self.ptt_de_button)
        layout.addLayout(button_layout)

        # Save recordings checkbox
        self.save_checkbox = QCheckBox("💾 Save all recordings to WAV")
        self.save_checkbox.setStyleSheet("font-size: 11pt; padding: 5px;")
        self.save_checkbox.stateChanged.connect(self.on_save_changed)
        layout.addWidget(self.save_checkbox)

        # Status
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

        # Translation Display
        display_layout = QHBoxLayout()

        # English text
        en_group = QGroupBox("🗣️ English")
        en_layout = QVBoxLayout()
        self.english_text = QTextEdit()
        self.english_text.setReadOnly(True)
        self.english_text.setMinimumHeight(250)
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
        en_layout.addWidget(self.english_text)
        en_group.setLayout(en_layout)

        # German text
        de_group = QGroupBox("🔊 German")
        de_layout = QVBoxLayout()
        self.german_text = QTextEdit()
        self.german_text.setReadOnly(True)
        self.german_text.setMinimumHeight(250)
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
        de_layout.addWidget(self.german_text)
        de_group.setLayout(de_layout)

        display_layout.addWidget(en_group)
        display_layout.addWidget(de_group)
        layout.addLayout(display_layout)

        # Instructions
        instructions = QLabel(
            "💡 Blue button: Speak English → outputs German | "
            "Orange button: Captures German audio → outputs English"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; font-size: 10pt; padding: 10px;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        central_widget.setLayout(layout)

        # Initialize worker
        self.init_worker()

    def load_audio_devices(self):
        """Load audio devices"""
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()

        # Populate input combo (microphones)
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0 and not device_info.get('isLoopbackDevice', False):
                name = f"[ID {i}] {device_info['name']}"
                self.input_combo.addItem(name, i)

        # Populate output combo - OUTPUT devices with loopback support
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)

            # Show output devices (speakers, headphones)
            if device_info['maxOutputChannels'] > 0:
                # Check if it's a loopback device
                if device_info.get('isLoopbackDevice', False):
                    name = f"[ID {i}] {device_info['name']} ⭐ LOOPBACK"
                else:
                    # Regular output device - pyaudiowpatch can capture from these
                    name = f"[ID {i}] {device_info['name']}"

                self.output_combo.addItem(name, i)

        # Populate TTS output combo - OUTPUT devices only (for playback)
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)

            # Only show output devices (not loopback, as those can't play audio)
            if device_info['maxOutputChannels'] > 0 and not device_info.get('isLoopbackDevice', False):
                name = f"[ID {i}] {device_info['name']}"
                # Highlight virtual cables (common for routing to Google Meet)
                if 'VB-CABLE' in device_info['name'].upper() or 'VIRTUAL' in device_info['name'].upper():
                    name += " 🎯 VIRTUAL"
                self.tts_output_combo.addItem(name, i)

        p.terminate()

        # Select default input device
        try:
            default_input_info = pyaudio.PyAudio().get_default_input_device_info()
            default_input = default_input_info['index']
            idx = self.input_combo.findData(default_input)
            if idx >= 0:
                self.input_combo.setCurrentIndex(idx)
        except:
            pass  # Use first device if default not found

    def init_worker(self):
        """Initialize dual translation worker"""
        import config

        input_id = self.input_combo.currentData()
        output_id = self.output_combo.currentData()
        tts_output_id = self.tts_output_combo.currentData()

        print(f"🔧 Device selection:")
        print(f"   Input device (mic): {input_id}")
        print(f"   Output device (loopback): {output_id}")
        print(f"   TTS output device: {tts_output_id}")

        # Validate device selection
        if input_id is None:
            print("⚠️  Warning: No input device selected, using default")
            try:
                import pyaudiowpatch as pyaudio
                p = pyaudio.PyAudio()
                default_input = p.get_default_input_device_info()
                input_id = default_input['index']
                print(f"   Using default input: {default_input['name']}")
                p.terminate()
            except:
                pass

        self.status_label.setText("🔄 Loading models... Please wait")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #fff9c4;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)

        # Create worker
        self.worker_thread = QThread()
        self.worker = DualTranslationWorker(
            input_id,
            output_id,
            tts_output_id=tts_output_id,
            model_size=config.WHISPER_MODEL_SIZE
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.english_text.connect(self.on_english_text)
        self.worker.german_text.connect(self.on_german_text)
        self.worker.error.connect(self.on_error)
        self.worker.started.connect(self.on_started)
        self.worker.stopped.connect(self.on_stopped)

        self.worker_thread.started.connect(self.worker.start_engines)
        self.worker_thread.start()

    def start_english(self):
        """English PTT pressed"""
        if not self.is_translating:
            return
        self.status_label.setText("🔴 Recording English...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #bbdefb;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)
        if self.worker:
            self.worker.start_recording_english()

    def stop_english(self):
        """English PTT released"""
        if not self.is_translating:
            return
        self.status_label.setText("🟢 Processing English...")
        if self.worker:
            self.worker.stop_recording_english()

    def start_german(self):
        """German PTT pressed"""
        if not self.is_translating:
            return
        self.status_label.setText("🟠 Listening to German...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #ffe0b2;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)
        if self.worker:
            self.worker.start_recording_german()

    def stop_german(self):
        """German PTT released"""
        if not self.is_translating:
            return
        self.status_label.setText("🟢 Processing German...")
        if self.worker:
            self.worker.stop_recording_german()

    def on_english_text(self, text: str):
        """Handle English text"""
        current = self.english_text.toPlainText()
        if current:
            self.english_text.append(text)
        else:
            self.english_text.setText(text)

    def on_german_text(self, text: str):
        """Handle German text"""
        current = self.german_text.toPlainText()
        if current:
            self.german_text.append(text)
        else:
            self.german_text.setText(text)

    def on_error(self, msg: str):
        """Handle error"""
        self.status_label.setText(f"🔴 Error: {msg}")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #ffcdd2;
                padding: 10px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)
        print(f"ERROR: {msg}")

    def on_started(self):
        """Engines started"""
        self.is_translating = True
        self.status_label.setText("🟢 Ready - Hold buttons to speak/listen")
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
        """Engines stopped"""
        self.is_translating = False
        self.status_label.setText("⚪ Stopped")

    def on_save_changed(self, state):
        """Toggle recording save"""
        enabled = state == Qt.CheckState.Checked.value
        if self.worker:
            self.worker.set_save_recordings(enabled)

        # Also enable for TTS outputs
        if enabled and self.worker:
            if hasattr(self.worker, 'engine_en_to_de') and self.worker.engine_en_to_de:
                if hasattr(self.worker.engine_en_to_de, 'tts'):
                    self.worker.engine_en_to_de.tts.set_save_recordings(True)
            if hasattr(self.worker, 'engine_de_to_en') and self.worker.engine_de_to_en:
                if hasattr(self.worker.engine_de_to_en, 'tts'):
                    self.worker.engine_de_to_en.tts.set_save_recordings(True)

    def on_apply_devices(self):
        """Apply device changes - restart worker with new devices"""
        print("\n" + "="*60)
        print("🔄 Applying device changes...")
        print("="*60)

        # Stop current worker
        if self.worker:
            print("Stopping current worker...")
            self.worker.stop_engines()

        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

        self.is_translating = False

        # Restart with new devices
        print("Restarting with new devices...")
        self.init_worker()

        print("✓ Device changes applied!")
        print("="*60 + "\n")

    def closeEvent(self, event):
        """Handle close"""
        if self.worker:
            self.worker.stop_engines()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()
