# 🎤 Live Translation - Real-time Bidirectional Speech Translation

**English ↔ German Live Translation System** optimized for NVIDIA RTX 5080

A professional-grade, real-time bidirectional speech translation application designed for Google Meet calls, live demonstrations, interviews, and presentations. Simultaneous English→German and German→English translation with push-to-talk controls.

## ✨ Features

- **🔄 Bidirectional Translation**: Simultaneous English→German and German→English
- **📞 Google Meet Integration**: Capture system audio and route German TTS to virtual microphone
- **🚀 Ultra-Low Latency**: ~0.9s end-to-end (11x faster than v1)
- **🎯 VRAM Optimized**: Reduced from 16GB to 8-11GB via shared Whisper model
- **🎨 Dual Push-to-Talk**: Separate buttons for speaking English and listening to German
- **🔊 Natural Voice**: High-quality German & English text-to-speech
- **⚡ Non-blocking UI**: Async processing prevents freezing during transcription
- **🎙️ System Audio Capture**: WASAPI loopback for Windows (captures Google Meet audio)

## 🎯 Use Cases

Perfect for:
- **Google Meet calls** - Real-time bidirectional translation for multilingual meetings
- **Job interviews** - Demonstrate technical skills despite language barriers
- **Live presentations** - Communicate with multilingual audiences
- **Language learning** - Practice conversations with instant feedback
- **Customer support** - Real-time assistance in multiple languages

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 (Linux/macOS for single-direction mode)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **CUDA**: 11.8 or 12.x
- **RAM**: 16GB
- **Python**: 3.10+
- **Virtual Audio Cable**: VB-Audio Virtual Cable (for Google Meet integration)

### Recommended Setup (Optimal Performance)
- **GPU**: NVIDIA RTX 5080 (or 4080/4090)
- **CUDA**: 12.1+
- **RAM**: 32GB
- **VRAM**: 16GB (for bidirectional mode with large-v3 models)
- **Storage**: 15GB free space (for models)

## 📦 Installation

### 1. Windows Optimizations (Recommended)

#### Enable Developer Mode for Faster Model Caching
Windows Developer Mode enables symlinks, which Hugging Face uses for efficient model caching:

1. Open **Settings** → **Privacy & Security** → **For developers**
2. Toggle **Developer Mode** to **On**
3. Restart your computer

**Benefits:**
- Faster model downloads and caching
- Reduces disk space usage for duplicate model files
- Eliminates symlink warnings

**Alternative:** Run Python as Administrator (not recommended for security)

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install CUDA-enabled PyTorch

```bash
# For RTX 5080 (Blackwell architecture - sm_120)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# For RTX 40-series and older (CUDA 12.1)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Note for RTX 5080 users:** The Blackwell architecture requires PyTorch nightly builds with CUDA 13.0 support. You'll also need CUDA 12.4 runtime libraries for CTranslate2 (faster-whisper dependency).

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

This includes:
- `huggingface_hub[hf_xet]` for faster model downloads (3-5x speedup)
- All ML frameworks (faster-whisper, transformers, TTS)
- GUI and audio libraries

### 5. Test Installation
```bash
python main.py --test
```

This will verify all models load correctly and check CUDA availability.

## 🚀 Quick Start

### Dual-Mode GUI (Recommended)
```bash
python main.py
```

**Device Setup:**
1. **Input Device**: Select your microphone
2. **Capture Device**: Select system audio output (for capturing German from Google Meet)
3. **German TTS Output**: Select virtual audio cable (e.g., VB-Cable Input)
4. Click **"✅ Apply Device Changes"**

**Using the Translation:**
- **Blue Button** (🇺🇸 SPEAK ENGLISH): Hold while speaking English
  - Transcribes your English → Translates to German → Plays German audio
  - Routes to Google Meet if virtual cable selected

- **Orange Button** (🇩🇪 LISTEN TO GERMAN): Hold to capture German audio
  - Captures system audio → Transcribes German → Displays English translation

**See [DUAL_MODE_GUIDE.md](DUAL_MODE_GUIDE.md) for complete usage instructions**

### Virtual Audio Setup for Google Meet
See [VIRTUAL_AUDIO_SETUP.md](VIRTUAL_AUDIO_SETUP.md) for VB-Cable installation and configuration.

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

## 📊 Expected Performance

### RTX 5080 (Bidirectional Mode)
- **Latency**: ~0.9s - 1.5s (speech end to audio output start)
- **Quality**: Excellent (large-v3 model)
- **VRAM Usage**: ~8-11GB (shared Whisper model)
- **CPU**: Minimal usage (<20%)

### Performance Improvements (v2 vs v1)
- **Latency**: 10.64s → 0.90s (11.8x faster)
- **VRAM**: 16GB → 8-11GB (50% reduction)
- **UI Responsiveness**: Blocking → Non-blocking async

### Latency Breakdown (per direction)
1. **Speech Recognition (Whisper large-v3)**: ~400-600ms
2. **Translation (MarianMT)**: ~100-200ms
3. **Text-to-Speech (Coqui TTS)**: ~200-400ms
4. **Audio Processing & Normalization**: ~100-200ms
5. **Total**: ~900-1500ms

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

### Bidirectional Translation Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    ENGLISH → GERMAN                          │
└──────────────────────────────────────────────────────────────┘

Your Microphone → Whisper (EN) → MarianMT (EN→DE) → German TTS
                      ↓                                    ↓
                 Shared Model                    VB-Cable → Google Meet
                      ↑
Google Meet → System Audio → WASAPI Loopback Capture
                                    ↓
            Whisper (DE) → MarianMT (DE→EN) → English Display

┌──────────────────────────────────────────────────────────────┐
│                    GERMAN → ENGLISH                          │
└──────────────────────────────────────────────────────────────┘
```

### Key Optimizations
- **Shared Whisper Model**: One model instance handles both EN and DE transcription
- **Thread-Safe Access**: Mutex locking prevents race conditions
- **Async Processing**: Non-blocking transcription keeps UI responsive
- **Audio Normalization**: Boosts quiet audio for better recognition
- **Polyphase Resampling**: Efficient 44.1kHz → 16kHz conversion

### Key Components

1. **Audio Capture** (`src/audio/`)
   - Device management and selection
   - Real-time microphone streaming
   - WASAPI loopback capture for system audio (Windows)
   - Voice activity detection (Silero VAD)

2. **Translation Engine** (`src/translation/`)
   - **Whisper**: Speech recognition (faster-whisper) with thread-safe shared access
   - **MarianMT**: Neural machine translation (EN↔DE)
   - **BidirectionalTranslator**: Coordinates both translation directions

3. **Speech Synthesis** (`src/tts/`)
   - **GermanTTS**: Coqui TTS with German Thorsten voice
   - **EnglishTTS**: Coqui TTS with English LJSpeech voice
   - Output device routing for virtual cables
   - Playback queue management

4. **GUI** (`src/gui/`)
   - PyQt6 dual-mode interface
   - Dual push-to-talk buttons
   - Real-time bilingual text display
   - Three device selectors (input, capture, TTS output)

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
├── main.py                           # Application entry point
├── config.py                         # Configuration settings
├── requirements.txt                  # Dependencies
├── README.md                         # This file
└── src/
    ├── audio/
    │   ├── device_manager.py        # Audio device handling
    │   ├── stream_capture.py        # Real-time capture
    │   ├── vad_processor.py         # Voice activity detection
    │   └── loopback_capture.py      # WASAPI loopback (system audio)
    ├── translation/
    │   ├── whisper_engine.py        # Whisper STT (thread-safe)
    │   ├── translator.py            # EN→DE & DE→EN translation
    │   ├── streaming_translator.py  # Legacy streaming coordinator
    │   └── bidirectional_translator.py  # Bidirectional engine
    ├── tts/
    │   ├── german_tts.py            # German TTS (device routing)
    │   └── english_tts.py           # English TTS (device routing)
    └── gui/
        └── main_window_dual.py      # Dual-mode GUI (primary)
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
- ✅ **Bidirectional Translation**: Simultaneous EN↔DE with shared resources
- ✅ **VRAM Optimization**: Shared Whisper model (50% memory reduction)
- ✅ **GPU Programming**: CUDA optimization for ML models
- ✅ **Real-time Systems**: Sub-second latency streaming
- ✅ **Thread-Safe Design**: Mutex locking for concurrent access
- ✅ **System Audio Capture**: WASAPI loopback integration
- ✅ **Multi-threading**: Parallel audio/translation/TTS processing
- ✅ **ML Integration**: Whisper, Transformers, Coqui TTS
- ✅ **Audio Processing**: VAD, normalization, resampling, buffering
- ✅ **GUI Development**: PyQt6 with async workers
- ✅ **Production Code**: Clean architecture, configuration, error handling


## 📄 License

MIT License

**Built for real-time multilingual communication**

*Optimized for NVIDIA RTX 5080 | Python 3.10+ | CUDA 12.1+*
