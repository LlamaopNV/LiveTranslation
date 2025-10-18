# 🎤 Live Translation - Real-time Speech-to-Speech Translation

**English → German Live Translation System** optimized for NVIDIA RTX 5080

A professional-grade, real-time speech translation application designed for live demonstrations, interviews, and presentations. Speak in English and hear your words translated to German with minimal latency.

## ✨ Features

- **🚀 True Streaming Translation**: Minimal latency (<1s) for live conversations
- **🎯 GPU Accelerated**: Optimized for NVIDIA RTX 5080 (works on any CUDA GPU)
- **🎨 Clean GUI**: Professional interface perfect for demonstrations
- **🔊 Natural Voice**: High-quality German text-to-speech
- **⚡ Real-time Processing**: Translates as you speak, not after you finish
- **🎙️ Voice Activity Detection**: Automatic speech detection and sentence segmentation

## 🎯 Use Case

Perfect for:
- **Job interviews** where you want to demonstrate technical skills despite language barriers
- **Live presentations** with multilingual audiences
- **Language learning** and practice
- **Real-time communication** in multilingual environments

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, macOS
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- **CUDA**: 11.8 or 12.x
- **RAM**: 16GB
- **Python**: 3.10+

### Recommended Setup (Optimal Performance)
- **GPU**: NVIDIA RTX 5080 (or 4080/4090)
- **CUDA**: 12.1+
- **RAM**: 32GB
- **Storage**: 10GB free space (for models)

## 📦 Installation

### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install CUDA-enabled PyTorch
```bash
# For CUDA 12.1 (recommended for RTX 5080)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Test Installation
```bash
python main.py --test
```

This will verify all models load correctly and check CUDA availability.

## 🚀 Quick Start

### GUI Mode (Recommended)
```bash
python main.py
```

1. **Select your microphone** from the dropdown
2. **Click "Start Translation"**
3. **Speak in English** - see real-time translation!
4. **German audio plays** automatically

### CLI Mode (For Testing)
```bash
python main.py --cli
```

Runs in terminal with text output and audio playback.

### Test Models
```bash
python main.py --test
```

Tests each component individually to verify setup.

## 🎛️ Configuration

Edit `config.py` to customize:

### Performance Presets
```python
from config import apply_preset

# Ultra fast - minimal latency (~500ms)
apply_preset('ultra_fast')

# Balanced - good quality and speed (~1s)
apply_preset('balanced')  # Default

# High quality - best accuracy (~2s)
apply_preset('high_quality')

# Low resource - for limited VRAM
apply_preset('low_resource')
```

### Key Settings
```python
# Model size (tiny/base/small/medium/large)
WHISPER_MODEL_SIZE = "small"

# Audio processing
MIN_CHUNK_DURATION = 0.3  # Seconds
MIN_SILENCE_DURATION_MS = 700  # Pause detection

# VAD sensitivity
VAD_THRESHOLD = 0.5  # 0.0-1.0

# Device
DEVICE = "cuda"  # or "cpu"
```

## 📊 Expected P3erformance

### RTX 5080 (Recommended Setup)
- **Latency**: ~800ms - 1.2s (speech end to German audio start)
- **Quality**: Excellent (small/medium model)
- **VRAM Usage**: ~4-5GB
- **CPU**: Minimal usage (<20%)

### Latency Breakdown
1. **Voice Activity Detection**: ~50ms
2. **Speech Recognition (Whisper)**: ~500-700ms
3. **Translation (MarianMT)**: ~100-200ms
4. **Text-to-Speech**: ~200-300ms
5. **Total**: ~850-1250ms

## 🎯 Interview Demonstration Tips

For your interview, here are optimization tips:

### 1. Pre-load Models
```bash
# Run once before interview to download all models
python main.py --test
```

### 2. Optimize for Speed
```python
# In config.py, use ultra_fast preset
apply_preset('ultra_fast')
```

### 3. Test Your Microphone
- Use a quality USB microphone if possible
- Minimize background noise
- Speak clearly with natural pauses

### 4. Practice Sentences
Test with technical sentences:
- "I am a software developer specializing in machine learning."
- "My experience includes Python, PyTorch, and CUDA optimization."
- "I have built real-time translation systems for production use."

### 5. Showcase Technical Skills
Point out during the demo:
- GPU acceleration and CUDA optimization
- Multi-threaded architecture for parallel processing
- Real-time streaming (not batch processing)
- Voice activity detection and sentence segmentation
- Clean, production-ready code structure

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    LIVE AUDIO STREAM                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   VAD Processor         │
         │   (Sentence Detection)  │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │   Whisper ASR (GPU)     │
         │   English Speech → Text │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │   MarianMT (GPU)        │
         │   English → German Text │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │   Coqui TTS (GPU)       │
         │   German Text → Speech  │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │   AUDIO OUTPUT          │
         └─────────────────────────┘
```

### Key Components

1. **Audio Capture** (`src/audio/`)
   - Device management and selection
   - Real-time streaming with buffering
   - Voice activity detection (Silero VAD)

2. **Translation Engine** (`src/translation/`)
   - Whisper: Speech recognition (faster-whisper)
   - MarianMT: Neural machine translation
   - Streaming coordinator with multi-threading

3. **Speech Synthesis** (`src/tts/`)
   - Coqui TTS with German voice
   - Playback queue management
   - GPU-accelerated synthesis

4. **GUI** (`src/gui/`)
   - PyQt6 interface
   - Real-time text display
   - Device selection and controls

## 🔧 Troubleshooting

### CUDA Not Found
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### High Latency
1. Use `ultra_fast` preset in `config.py`
2. Close other GPU applications
3. Reduce `WHISPER_MODEL_SIZE` to "tiny" or "base"
4. Lower `MIN_CHUNK_DURATION` to 0.2s

### Poor Audio Quality
1. Use better microphone
2. Increase `WHISPER_MODEL_SIZE` to "medium"
3. Adjust `VAD_THRESHOLD` (try 0.3-0.7)
4. Ensure quiet environment

### Models Won't Download
Models download automatically on first run. If issues:
```bash
# Manually download Whisper model
python -c "from faster_whisper import WhisperModel; WhisperModel('small')"

# Manually download translation model
python -c "from transformers import MarianMTModel; MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')"
```

### Memory Issues
1. Use `low_resource` preset
2. Set `COMPUTE_TYPE = "int8"` in config
3. Close other applications
4. Use smaller model: `WHISPER_MODEL_SIZE = "tiny"`

## 📝 Development

### Project Structure
```
LiveTranslation/
├── main.py                  # Application entry point
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
├── CLAUDE.md               # Comprehensive documentation
├── README.md               # This file
└── src/
    ├── audio/
    │   ├── device_manager.py       # Audio device handling
    │   ├── stream_capture.py       # Real-time capture
    │   └── vad_processor.py        # Voice activity detection
    ├── translation/
    │   ├── whisper_engine.py       # Whisper STT
    │   ├── translator.py           # Text translation
    │   └── streaming_translator.py # Streaming coordinator
    ├── tts/
    │   └── german_tts.py          # German TTS
    └── gui/
        └── main_window.py         # Main GUI
```

### Running Tests
```bash
# Test individual components
python src/audio/device_manager.py
python src/translation/whisper_engine.py
python src/translation/translator.py
python src/tts/german_tts.py
python src/audio/vad_processor.py

# Test streaming engine
python src/translation/streaming_translator.py
```

## 🎓 Technical Highlights

Built to showcase:
- ✅ **GPU Programming**: CUDA optimization for ML models
- ✅ **Real-time Systems**: Sub-second latency streaming
- ✅ **Multi-threading**: Parallel audio/translation/TTS processing
- ✅ **ML Integration**: Whisper, Transformers, Coqui TTS
- ✅ **Audio Processing**: VAD, streaming, buffering
- ✅ **GUI Development**: PyQt6 with threaded workers
- ✅ **Production Code**: Clean architecture, configuration, error handling

## 💡 Future Enhancements

Potential additions:
- [ ] Bidirectional translation (German → English)
- [ ] Multi-language support (French, Spanish, etc.)
- [ ] Translation history and export
- [ ] Custom vocabulary/phrases
- [ ] Cloud deployment option
- [ ] Mobile app version

## 📄 License

MIT License

## 📧 Documentation

See **CLAUDE.md** for comprehensive technical documentation including:
- Detailed architecture diagrams
- API reference for all modules
- Performance optimization guide
- Complete troubleshooting reference

---

**Built for real-time multilingual communication**

*Optimized for NVIDIA RTX 5080 | Python 3.10+ | CUDA 12.1+*
