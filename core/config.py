"""
Configuration management for the Text-to-Video Generator.
Uses pydantic-settings for environment variable loading.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class VideoGeneratorConfig(BaseSettings):
    """CogVideoX video generation settings."""
    model_id: str = Field(default="THUDM/CogVideoX-2b", description="HuggingFace model ID")
    height: int = Field(default=480, description="Output video height")
    width: int = Field(default=720, description="Output video width")
    num_frames: int = Field(default=49, description="Number of frames (6s at 8fps)")
    fps: int = Field(default=8, description="Frames per second")
    guidance_scale: float = Field(default=6.0, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(default=50, description="Denoising steps")
    enable_cpu_offload: bool = Field(default=True, description="Offload to CPU to save VRAM")
    enable_vae_tiling: bool = Field(default=True, description="Tile VAE for memory efficiency")
    
    class Config:
        env_prefix = "VIDEO_"


class TTSConfig(BaseSettings):
    """Coqui XTTS-v2 text-to-speech settings."""
    model_name: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2")
    language: str = Field(default="en", description="Target language code")
    speaker_wav: Optional[str] = Field(default=None, description="Path to reference voice sample")
    split_sentences: bool = Field(default=True, description="Split text into sentences")
    
    class Config:
        env_prefix = "TTS_"


class LipSyncConfig(BaseSettings):
    """Wav2Lip lip synchronization settings."""
    checkpoint_path: str = Field(default="checkpoints/wav2lip_gan.pth")
    face_det_checkpoint: str = Field(default="checkpoints/s3fd.pth")
    static_fps: int = Field(default=25, description="Output video FPS")
    face_det_batch_size: int = Field(default=16)
    wav2lip_batch_size: int = Field(default=128)
    resize_factor: int = Field(default=1, description="Downscale factor for faster processing")
    crop: list = Field(default=[0, -1, 0, -1], description="Crop region [top, bottom, left, right]")
    box: list = Field(default=[-1, -1, -1, -1], description="Face bounding box override")
    nosmooth: bool = Field(default=False, description="Disable face detection smoothing")
    
    class Config:
        env_prefix = "LIPSYNC_"


class UpscalerConfig(BaseSettings):
    """Real-ESRGAN upscaling settings."""
    model_name: str = Field(default="RealESRGAN_x4plus", description="Upscaler model")
    scale: int = Field(default=4, description="Upscaling factor")
    tile_size: int = Field(default=400, description="Tile size for large images")
    tile_pad: int = Field(default=10, description="Tile padding")
    denoise_strength: float = Field(default=0.5, description="Denoising strength (0-1)")
    gpu_id: int = Field(default=0, description="GPU device ID")
    
    class Config:
        env_prefix = "UPSCALE_"


class AppConfig(BaseSettings):
    """Main application configuration."""
    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent)
    outputs_dir: Path = Field(default=Path("outputs"))
    checkpoints_dir: Path = Field(default=Path("checkpoints"))
    temp_dir: Path = Field(default=Path("temp"))
    
    # API Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: list = Field(default=["*"])
    
    # Processing Settings
    max_concurrent_jobs: int = Field(default=1, description="Max parallel generation jobs")
    cleanup_temp_files: bool = Field(default=True)
    
    # GPU Settings
    cuda_visible_devices: str = Field(default="0")
    torch_dtype: str = Field(default="float16", description="float16, bfloat16, or float32")
    
    # Module Configs
    video: VideoGeneratorConfig = Field(default_factory=VideoGeneratorConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    lip_sync: LipSyncConfig = Field(default_factory=LipSyncConfig)
    upscaler: UpscalerConfig = Field(default_factory=UpscalerConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "APP_"


# Global config instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the application configuration."""
    return config
