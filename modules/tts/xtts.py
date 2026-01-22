"""
Coqui XTTS-v2 Text-to-Speech Module
High-quality multi-language TTS with voice cloning support.
"""

import torch
from pathlib import Path
from typing import Optional, Union
import numpy as np

from TTS.api import TTS

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.config import TTSConfig, get_config
from core.utils import get_device, ensure_dir, clear_gpu_memory


class XTTSGenerator:
    """
    Coqui XTTS-v2 Text-to-Speech generator.
    Supports multi-language synthesis and voice cloning.
    """
    
    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or get_config().tts
        self.device = device or get_device()
        self.tts: Optional[TTS] = None
        self._loaded = False
    
    def load_model(self) -> None:
        """Load the XTTS-v2 model."""
        if self._loaded:
            return
        
        print(f"Loading XTTS-v2 model: {self.config.model_name}")
        
        # Initialize TTS with XTTS-v2
        self.tts = TTS(
            model_name=self.config.model_name,
            progress_bar=True,
        )
        
        # Move to GPU if available
        if self.device.type == "cuda":
            self.tts = self.tts.to(self.device)
        
        self._loaded = True
        print("XTTS-v2 model loaded successfully")
    
    def unload_model(self) -> None:
        """Unload model and free memory."""
        if self.tts is not None:
            del self.tts
            self.tts = None
        self._loaded = False
        clear_gpu_memory()
        print("XTTS-v2 model unloaded")
    
    def synthesize(
        self,
        text: str,
        output_path: Path,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        split_sentences: Optional[bool] = None,
    ) -> Path:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the audio file
            speaker_wav: Optional path to reference audio for voice cloning
            language: Language code (e.g., 'en', 'es', 'fr')
            split_sentences: Whether to split text into sentences
        
        Returns:
            Path to the generated audio file
        """
        self.load_model()
        
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        # Use config defaults if not specified
        language = language or self.config.language
        speaker_wav = speaker_wav or self.config.speaker_wav
        split_sentences = split_sentences if split_sentences is not None else self.config.split_sentences
        
        print(f"Synthesizing speech: {len(text)} chars, language={language}")
        
        if speaker_wav:
            # Voice cloning mode
            speaker_wav = str(speaker_wav)
            print(f"Using voice reference: {speaker_wav}")
            
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=speaker_wav,
                language=language,
                split_sentences=split_sentences,
            )
        else:
            # Default voice mode
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                language=language,
                split_sentences=split_sentences,
            )
        
        print(f"Audio saved to: {output_path}")
        return output_path
    
    def synthesize_to_array(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech and return as numpy array.
        
        Args:
            text: Text to synthesize
            speaker_wav: Optional voice reference
            language: Language code
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self.load_model()
        
        language = language or self.config.language
        speaker_wav = str(speaker_wav) if speaker_wav else self.config.speaker_wav
        
        if speaker_wav:
            wav = self.tts.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
            )
        else:
            wav = self.tts.tts(
                text=text,
                language=language,
            )
        
        # Get sample rate from the model
        sample_rate = self.tts.synthesizer.output_sample_rate
        
        return np.array(wav), sample_rate
    
    def list_languages(self) -> list:
        """Get list of supported languages."""
        self.load_model()
        if hasattr(self.tts, 'languages'):
            return self.tts.languages
        return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", 
                "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]


# Convenience function
def generate_speech(
    text: str,
    output_path: Optional[Path] = None,
    speaker_wav: Optional[str] = None,
    language: str = "en",
) -> Path:
    """
    Quick function to generate speech from text.
    
    Args:
        text: Text to speak
        output_path: Where to save (auto-generated if not provided)
        speaker_wav: Optional voice reference for cloning
        language: Language code
    
    Returns:
        Path to generated audio
    """
    config = get_config()
    
    if output_path is None:
        from core.utils import generate_job_id
        job_id = generate_job_id()
        output_path = config.outputs_dir / f"{job_id}_audio.wav"
    
    generator = XTTSGenerator()
    try:
        return generator.synthesize(
            text=text,
            output_path=output_path,
            speaker_wav=speaker_wav,
            language=language,
        )
    finally:
        generator.unload_model()


if __name__ == "__main__":
    # Quick test
    test_text = "Hello! This is a test of the XTTS text-to-speech system. It sounds quite natural, doesn't it?"
    output = generate_speech(test_text)
    print(f"Test audio generated: {output}")
