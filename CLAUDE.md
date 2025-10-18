# LiveTranslation - Real-time Speech-to-Speech Translation System

## Project Overview
A local, GPU-accelerated live translation system that performs real-time English to German speech-to-speech translation with minimal latency. Optimized for NVIDIA RTX 5080.

## Architecture

### System Flow
```
[Microphone Input]
    → [Audio Capture + VAD]
    → [Whisper STT + Translation]
    → [German TTS]
    → [Audio Output]
```

### Components

#### 1. Audio Module (`src/audio/`)
- **device_manager.py**: Audio device enumeration and selection
- **stream_capture.py**: Real-time audio streaming with buffering
- **vad_processor.py**: Voice Activity Detection using Silero VAD

#### 2. Translation Module (`src/translation/`)
- **whisper_engine.py**: GPU-accelerated Whisper model management
- **translator.py**: English → German translation pipeline
- Uses `faster-whisper` for optimized inference on CUDA

#### 3. TTS Module (`src/tts/`)
- **german_tts.py**: Local German text-to-speech synthesis
- Uses Coqui TTS with GPU acceleration
- Supports multiple German voices

#### 4. GUI Module (`src/gui/`)
- **main_window.py**: PyQt6 application window
- **controls.py**: Start/stop controls and status indicators
- Device selection interface

## Technical Stack

### Core Dependencies
- **Python**: 3.10+
- **GPU Framework**: PyTorch with CUDA support
- **Speech Recognition**: faster-whisper (CTranslate2 backend)
- **TTS Engine**: Coqui TTS
- **Audio I/O**: sounddevice
- **VAD**: Silero VAD
- **GUI**: PyQt6

### Hardware Requirements
- **GPU**: NVIDIA RTX 5080 (or similar with 8GB+ VRAM)
- **CUDA**: 11.8 or 12.x
- **RAM**: 16GB+ recommended
- **Microphone**: Any USB or built-in microphone

## Installation

### 1. Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install CUDA Support
Ensure PyTorch is installed with CUDA support:
```bash
# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Download Models
The application will automatically download models on first run:
- Whisper small/medium model (~1.5GB)
- Silero VAD model (~1MB)
- Coqui TTS German model (~500MB)

## Usage

### Basic Usage
```bash
python main.py
```

### GUI Controls
1. **Select Input Device**: Choose your microphone from dropdown
2. **Start Translation**: Begin live translation
3. **Stop Translation**: End translation session
4. **Status Indicator**: Shows current state (listening/translating/speaking)

### Configuration
Edit `config.py` to customize:
- Whisper model size (tiny/base/small/medium/large)
- Audio chunk duration (affects latency)
- VAD sensitivity threshold
- TTS voice selection

## Performance Optimization

### Minimizing Latency
1. **Audio Chunking**: 1-2 second chunks balance accuracy and speed
2. **Streaming Processing**: Overlapping chunks for continuous translation
3. **GPU Batching**: Process multiple segments efficiently
4. **Parallel Pipeline**: While TTS plays, next audio chunk processes

### Expected Latency
- **Audio Capture**: ~100ms
- **Whisper Inference**: ~500-800ms (medium model on RTX 5080)
- **TTS Synthesis**: ~200-400ms
- **Total Delay**: ~1-2 seconds from speech end to translation start

### GPU Memory Management
- Whisper model: ~2GB VRAM
- TTS model: ~1GB VRAM
- Headroom: ~5GB for batch processing

## Project Structure
```
LiveTranslation/
├── CLAUDE.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                   # Application entry point
├── config.py                 # Configuration settings
├── src/
│   ├── audio/
│   │   ├── device_manager.py    # Audio device handling
│   │   ├── stream_capture.py    # Real-time audio capture
│   │   └── vad_processor.py     # Voice activity detection
│   ├── translation/
│   │   ├── whisper_engine.py    # Whisper model wrapper
│   │   └── translator.py        # Translation pipeline
│   ├── tts/
│   │   └── german_tts.py        # German TTS synthesis
│   └── gui/
│       ├── main_window.py       # Main application window
│       └── controls.py          # UI controls
└── tests/                    # Unit tests

```

## Development Roadmap

### Phase 1: Core Functionality (Current)
- [x] Project structure
- [ ] Audio input with device selection
- [ ] Whisper integration with GPU
- [ ] English → German translation
- [ ] German TTS output
- [ ] Basic GUI

### Phase 2: Optimization
- [ ] VAD integration for automatic speech detection
- [ ] Latency optimization
- [ ] Memory management improvements
- [ ] Error handling and recovery

### Phase 3: Future Enhancements
- [ ] Bidirectional translation (German → English)
- [ ] Multiple language support
- [ ] Custom vocabulary/phrases
- [ ] Translation history logging
- [ ] Hotkey controls
- [ ] System tray integration

## Troubleshooting

### GPU Not Detected
```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Audio Device Issues
- Check device permissions in Windows Settings
- Ensure microphone is set as default recording device
- Try running with administrator privileges

### High Latency
- Reduce audio chunk size in config
- Switch to smaller Whisper model
- Close other GPU-intensive applications
- Check GPU utilization with `nvidia-smi`

### TTS Quality Issues
- Try different TTS voices in config
- Adjust sample rate settings
- Ensure sufficient GPU memory available

## Technical Details

### Whisper Model Selection
- **tiny**: Fastest, lowest quality (~390ms on RTX 5080)
- **base**: Fast, decent quality (~550ms)
- **small**: Good balance (~650ms) **← Recommended**
- **medium**: Better quality (~850ms)
- **large**: Best quality, slower (~1200ms)

### Audio Processing Pipeline
1. Capture 16kHz mono audio from microphone
2. Apply VAD to detect speech segments
3. Buffer 1-2 second chunks with 0.5s overlap
4. Process through Whisper for transcription + translation
5. Synthesize German audio with TTS
6. Play through default audio output

### Translation Approach
Using Whisper's built-in multilingual translation:
- Input: English speech audio
- Whisper task: "translate" (transcribe + translate to target language)
- Output: German text
- No separate translation model required

## API Reference

### WhisperEngine
```python
from src.translation.whisper_engine import WhisperEngine

engine = WhisperEngine(model_size="small", device="cuda")
german_text = engine.translate(audio_data, source_lang="en", target_lang="de")
```

### GermanTTS
```python
from src.tts.german_tts import GermanTTS

tts = GermanTTS(device="cuda")
audio = tts.synthesize("Hallo Welt")
tts.play(audio)
```

### AudioCapture
```python
from src.audio.stream_capture import AudioCapture

capture = AudioCapture(device_id=0, chunk_duration=2.0)
audio_chunk = capture.get_next_chunk()
```

## Contributing
This is a personal project. Future contributions may be accepted after core functionality is stable.

## License
MIT License

## Contact
For issues or questions, refer to the project repository.

---
**Last Updated**: 2025-10-18
**Version**: 0.1.0
**GPU Target**: NVIDIA RTX 5080
