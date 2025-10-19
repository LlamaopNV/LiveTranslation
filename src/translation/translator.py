"""
Bidirectional Translation (English ↔ German)
Uses MarianMT models for high-quality neural machine translation
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Optional


class GermanToEnglishTranslator:
    """
    Translates German text to English using MarianMT
    Optimized for GPU acceleration
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize translator

        Args:
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = "Helsinki-NLP/opus-mt-de-en"

        print(f"Loading translation model: {self.model_name}...")

        # Load model and tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

        # Move to GPU
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"Translation model loaded on {self.device}")
        if self.device == "cuda":
            print(f"VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    def translate(
        self,
        text: str,
        max_length: int = 512,
        num_beams: int = 5
    ) -> str:
        """
        Translate German text to English

        Args:
            text: German text to translate
            max_length: Maximum length of generated translation
            num_beams: Number of beams for beam search (higher = better quality)

        Returns:
            English translation
        """
        if not text or not text.strip():
            return ""

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            translated = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        # Decode output
        english_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)

        return english_text

    def translate_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        num_beams: int = 5
    ) -> List[str]:
        """
        Translate multiple German texts to English in batch

        Args:
            texts: List of German texts
            max_length: Maximum length of generated translations
            num_beams: Number of beams for beam search

        Returns:
            List of English translations
        """
        if not texts:
            return []

        # Filter empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [""] * len(texts)

        # Tokenize all inputs
        inputs = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Generate translations
        with torch.no_grad():
            translated = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        # Decode outputs
        english_texts = [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in translated
        ]

        return english_texts

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        }

    def unload(self) -> None:
        """Unload model and free GPU memory"""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Translation model unloaded")


class EnglishToGermanTranslator:
    """
    Translates English text to German using MarianMT
    Optimized for GPU acceleration
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize translator

        Args:
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = "Helsinki-NLP/opus-mt-en-de"

        print(f"Loading translation model: {self.model_name}...")

        # Load model and tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

        # Move to GPU
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"Translation model loaded on {self.device}")
        if self.device == "cuda":
            print(f"VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    def translate(
        self,
        text: str,
        max_length: int = 512,
        num_beams: int = 4
    ) -> str:
        """
        Translate English text to German

        Args:
            text: English text to translate
            max_length: Maximum length of generated translation
            num_beams: Number of beams for beam search (higher = better quality)

        Returns:
            German translation
        """
        if not text or not text.strip():
            return ""

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            translated = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        # Decode output
        german_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)

        return german_text

    def translate_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        num_beams: int = 4
    ) -> List[str]:
        """
        Translate multiple English texts to German in batch

        Args:
            texts: List of English texts
            max_length: Maximum length of generated translations
            num_beams: Number of beams for beam search

        Returns:
            List of German translations
        """
        if not texts:
            return []

        # Filter empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [""] * len(texts)

        # Tokenize all inputs
        inputs = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Generate translations
        with torch.no_grad():
            translated = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        # Decode outputs
        german_texts = [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in translated
        ]

        return german_texts

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        }

    def unload(self) -> None:
        """Unload model and free GPU memory"""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Translation model unloaded")


class TranslationPipeline:
    """
    Complete translation pipeline combining Whisper and MarianMT
    English speech → English text → German text
    """

    def __init__(
        self,
        whisper_model_size: str = "small",
        device: str = "cuda"
    ):
        """
        Initialize complete translation pipeline

        Args:
            whisper_model_size: Whisper model size
            device: Device to run on
        """
        from .whisper_engine import WhisperEngine

        self.device = device
        print("Initializing translation pipeline...")

        # Load Whisper for speech recognition
        self.whisper = WhisperEngine(
            model_size=whisper_model_size,
            device=device
        )

        # Load MarianMT for translation
        self.translator = EnglishToGermanTranslator(device=device)

        print("Translation pipeline ready!")

    def translate_audio(
        self,
        audio,
        whisper_beam_size: int = 5,
        translation_beam_size: int = 4,
        vad_filter: bool = True
    ) -> tuple[str, str]:
        """
        Translate English audio to German text

        Args:
            audio: Audio data (numpy array)
            whisper_beam_size: Beam size for Whisper
            translation_beam_size: Beam size for translation
            vad_filter: Enable VAD filtering

        Returns:
            Tuple of (english_text, german_text)
        """
        # Step 1: Transcribe English audio to English text
        english_text = self.whisper.transcribe_english(
            audio,
            beam_size=whisper_beam_size,
            vad_filter=vad_filter
        )

        if not english_text or not english_text.strip():
            return "", ""

        # Step 2: Translate English text to German
        german_text = self.translator.translate(
            english_text,
            num_beams=translation_beam_size
        )

        return english_text, german_text

    def get_pipeline_info(self) -> dict:
        """Get information about the pipeline"""
        whisper_info = self.whisper.get_model_info()
        translator_info = self.translator.get_model_info()

        return {
            "whisper": whisper_info,
            "translator": translator_info,
            "total_vram_gb": whisper_info.get("vram_allocated_gb", 0) + translator_info.get("vram_allocated_gb", 0)
        }

    def unload(self) -> None:
        """Unload all models"""
        self.whisper.unload()
        self.translator.unload()
        print("Translation pipeline unloaded")


def test_translator():
    """Test function for translator"""
    print("Testing English to German translator...")

    # Create translator
    translator = EnglishToGermanTranslator(device="cuda")

    # Test translations
    test_sentences = [
        "Hello, how are you today?",
        "I am learning to speak German.",
        "The weather is beautiful today.",
        "What time is it?",
        "Nice to meet you!"
    ]

    print("\nTranslation tests:")
    for english in test_sentences:
        german = translator.translate(english)
        print(f"EN: {english}")
        print(f"DE: {german}")
        print()

    # Cleanup
    translator.unload()


if __name__ == "__main__":
    test_translator()
