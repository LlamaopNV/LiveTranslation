"""
Configuration Settings
Customize translation behavior, model selection, and performance parameters
"""

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Whisper model size - affects quality and speed
# Options: "tiny", "base", "small", "medium", "large-v2", "large-v3"
# Recommended: "large-v3" for best quality (use for final demo)
WHISPER_MODEL_SIZE = "large-v3"

# Translation model (currently fixed to MarianMT)
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-de"

# TTS model for German speech synthesis
# Default: German Thorsten voice (high quality, male voice)
# tacotron2-DDC: Higher quality, slower, can stumble on some words
# vits: Faster, better pronunciation consistency
TTS_MODEL = "tts_models/de/thorsten/vits"
# Alternative: "tts_models/de/thorsten/tacotron2-DDC"

# ============================================================================
# DEVICE SETTINGS
# ============================================================================

# Device to run models on ("cuda" for GPU, "cpu" for CPU)
# Will automatically fall back to CPU if CUDA is not available
DEVICE = "cuda"

# PyTorch compute type for faster-whisper
# Options: "float16" (faster, GPU only), "float32" (more accurate), "int8" (fastest, lower quality)
COMPUTE_TYPE = "float16"

# ============================================================================
# AUDIO SETTINGS
# ============================================================================

# Audio sample rate (Hz)
# 16000 is optimal for Whisper and most ASR models
SAMPLE_RATE = 16000

# Audio channels (1 for mono, 2 for stereo)
# Mono is recommended for speech recognition
AUDIO_CHANNELS = 1

# Minimum audio chunk duration for processing (seconds)
# Lower = more responsive but potentially less accurate
# Higher = more accurate but higher latency
# Recommended: 0.3 - 0.5 seconds
MIN_CHUNK_DURATION = 0.5

# Audio chunk overlap (seconds)
# Helps with continuity between chunks
CHUNK_OVERLAP = 0.3

# ============================================================================
# VAD (Voice Activity Detection) SETTINGS
# ============================================================================

# VAD threshold (0.0 - 1.0)
# Lower = more sensitive (detects quieter speech)
# Higher = less sensitive (only loud/clear speech)
VAD_THRESHOLD = 0.3

# Minimum speech duration to consider valid (milliseconds)
MIN_SPEECH_DURATION_MS = 250

# Minimum silence duration to end speech segment (milliseconds)
# This effectively controls sentence boundary detection
# Lower = splits sentences more aggressively (faster response)
# Higher = waits for longer pauses (more complete sentences)
MIN_SILENCE_DURATION_MS = 1000

# Maximum sentence duration before forcing split (seconds)
# Prevents extremely long segments
MAX_SENTENCE_DURATION_S = 8.0

# ============================================================================
# TRANSLATION SETTINGS
# ============================================================================

# Whisper beam search size
# Higher = better quality but slower
# Recommended: 1 for speed, 5 for quality
WHISPER_BEAM_SIZE = 1

# Translation beam search size
# Higher = better translation quality but slower
# Recommended: 4-5 for natural German (worth the extra ~100ms)
TRANSLATION_BEAM_SIZE = 5

# Enable VAD filtering in Whisper
# Can improve quality but may add latency
WHISPER_VAD_FILTER = False

# ============================================================================
# TTS SETTINGS
# ============================================================================

# TTS sample rate (Hz)
# Default for most Coqui TTS models is 22050
TTS_SAMPLE_RATE = 22050

# TTS playback queue enabled
# If True, German audio plays sequentially
# If False, only the latest translation plays
TTS_QUEUE_ENABLED = True

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Number of CPU threads for preprocessing
CPU_THREADS = 4

# Number of workers for parallel processing
NUM_WORKERS = 1

# Enable model caching to speed up repeated loads
ENABLE_MODEL_CACHE = True

# Clear CUDA cache between translations (may reduce memory but slower)
CLEAR_CUDA_CACHE = False

# ============================================================================
# GUI SETTINGS
# ============================================================================

# Window size (width, height)
WINDOW_SIZE = (900, 700)

# Font sizes
HEADER_FONT_SIZE = 20
TEXT_FONT_SIZE = 13

# Show partial/interim results in UI
SHOW_PARTIAL_RESULTS = True

# Maximum text history to display
MAX_TEXT_HISTORY_LINES = 100

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Enable verbose logging
VERBOSE_LOGGING = True

# Log translation segments to file
LOG_TO_FILE = False
LOG_FILE_PATH = "translation_log.txt"

# Display performance metrics
SHOW_PERFORMANCE_METRICS = True

# ============================================================================
# ADVANCED SETTINGS (modify with caution)
# ============================================================================

# Audio buffer size for sounddevice (samples)
# Lower = lower latency but more CPU usage
# Higher = higher latency but less CPU usage
AUDIO_BUFFER_SIZE = None  # None = auto-calculate

# Maximum queue sizes (prevent memory buildup)
MAX_AUDIO_QUEUE_SIZE = 50
MAX_TRANSLATION_QUEUE_SIZE = 20

# Enable audio normalization
ENABLE_AUDIO_NORMALIZATION = True

# Silence threshold for simple energy-based VAD fallback
SILENCE_THRESHOLD = 0.01


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def apply_preset(preset_name: str):
    """
    Apply a preset configuration

    Available presets:
    - "ultra_fast": Minimal latency, lower quality (for demos)
    - "balanced": Good balance of speed and quality (recommended)
    - "high_quality": Best quality, higher latency (for accuracy)
    - "low_resource": Optimized for systems with limited VRAM
    """
    global WHISPER_MODEL_SIZE, WHISPER_BEAM_SIZE, TRANSLATION_BEAM_SIZE
    global MIN_CHUNK_DURATION, MIN_SILENCE_DURATION_MS, COMPUTE_TYPE

    if preset_name == "ultra_fast":
        WHISPER_MODEL_SIZE = "tiny"
        WHISPER_BEAM_SIZE = 1
        TRANSLATION_BEAM_SIZE = 1
        MIN_CHUNK_DURATION = 0.2
        MIN_SILENCE_DURATION_MS = 500
        print("Applied preset: Ultra Fast")

    elif preset_name == "balanced":
        WHISPER_MODEL_SIZE = "small"
        WHISPER_BEAM_SIZE = 1
        TRANSLATION_BEAM_SIZE = 1
        MIN_CHUNK_DURATION = 0.3
        MIN_SILENCE_DURATION_MS = 700
        print("Applied preset: Balanced (default)")

    elif preset_name == "high_quality":
        WHISPER_MODEL_SIZE = "medium"
        WHISPER_BEAM_SIZE = 5
        TRANSLATION_BEAM_SIZE = 4
        MIN_CHUNK_DURATION = 0.5
        MIN_SILENCE_DURATION_MS = 1000
        print("Applied preset: High Quality")

    elif preset_name == "low_resource":
        WHISPER_MODEL_SIZE = "base"
        WHISPER_BEAM_SIZE = 1
        TRANSLATION_BEAM_SIZE = 1
        MIN_CHUNK_DURATION = 0.4
        COMPUTE_TYPE = "int8"
        print("Applied preset: Low Resource")

    else:
        print(f"Unknown preset: {preset_name}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("Current Configuration:")
    print(f"  Whisper Model: {WHISPER_MODEL_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Min Chunk Duration: {MIN_CHUNK_DURATION}s")
    print(f"  VAD Threshold: {VAD_THRESHOLD}")
    print()

    print("Available presets:")
    print("  - ultra_fast: Minimal latency (~500ms)")
    print("  - balanced: Good balance (~1s latency)")
    print("  - high_quality: Best quality (~2s latency)")
    print("  - low_resource: For limited VRAM")
    print()

    print("To use a preset:")
    print("  from config import apply_preset")
    print("  apply_preset('ultra_fast')")
