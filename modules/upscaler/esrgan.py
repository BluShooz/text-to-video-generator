"""
Real-ESRGAN Video Upscaling Module
AI-powered 4x video enhancement.
"""

import torch
from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2
import tempfile
import subprocess

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.config import UpscalerConfig, get_config
from core.utils import (
    get_device, ensure_dir, clear_gpu_memory,
    video_to_frames, frames_to_video, get_video_info
)


class RealESRGANUpscaler:
    """
    Real-ESRGAN video upscaler.
    Upscales video frames using AI super-resolution.
    """
    
    def __init__(
        self,
        config: Optional[UpscalerConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or get_config().upscaler
        self.device = device or get_device()
        self.model = None
        self._loaded = False
    
    def load_model(self) -> None:
        """Load Real-ESRGAN model."""
        if self._loaded:
            return
        
        print(f"Loading Real-ESRGAN model: {self.config.model_name}")
        
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Configure model architecture based on model name
            if self.config.model_name == "RealESRGAN_x4plus":
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4
                )
                netscale = 4
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            elif self.config.model_name == "RealESRGAN_x4plus_anime_6B":
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=6,
                    num_grow_ch=32,
                    scale=4
                )
                netscale = 4
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            else:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4
                )
                netscale = 4
                model_url = None
            
            # Initialize upscaler
            self.model = RealESRGANer(
                scale=netscale,
                model_path=model_url,
                model=model,
                tile=self.config.tile_size,
                tile_pad=self.config.tile_pad,
                pre_pad=0,
                half=self.device.type == "cuda",
                gpu_id=self.config.gpu_id if self.device.type == "cuda" else None,
            )
            
        except ImportError:
            print("Real-ESRGAN not available, using fallback upscaling")
            self.model = None
        
        self._loaded = True
        print("Real-ESRGAN model loaded successfully")
    
    def unload_model(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        clear_gpu_memory()
        print("Real-ESRGAN model unloaded")
    
    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale a single frame.
        
        Args:
            frame: Input frame (RGB, uint8)
        
        Returns:
            Upscaled frame
        """
        self.load_model()
        
        if self.model is None:
            # Fallback: OpenCV upscaling
            return cv2.resize(
                frame,
                None,
                fx=self.config.scale,
                fy=self.config.scale,
                interpolation=cv2.INTER_LANCZOS4
            )
        
        # Convert RGB to BGR for Real-ESRGAN
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Upscale
        output, _ = self.model.enhance(frame_bgr, outscale=self.config.scale)
        
        # Convert back to RGB
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        return output_rgb
    
    def upscale_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """
        Upscale entire video.
        
        Args:
            input_path: Path to input video
            output_path: Path to save upscaled video
            progress_callback: Optional callback(current_frame, total_frames)
        
        Returns:
            Path to upscaled video
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        print(f"Upscaling video: {input_path}")
        
        # Get video info
        info = get_video_info(input_path)
        total_frames = int(info.get('duration', 0) * info.get('fps', 25))
        
        # Extract frames
        frames, fps = video_to_frames(input_path)
        total_frames = len(frames)
        
        print(f"Processing {total_frames} frames at {self.config.scale}x...")
        
        # Upscale each frame
        upscaled_frames = []
        for i, frame in enumerate(frames):
            upscaled = self.upscale_frame(frame)
            upscaled_frames.append(upscaled)
            
            if progress_callback:
                progress_callback(i + 1, total_frames)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{total_frames} frames")
        
        # Reassemble video
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video = Path(temp_dir) / "upscaled.mp4"
            frames_to_video(upscaled_frames, temp_video, fps=fps)
            
            # Check if original has audio
            has_audio = self._check_audio(input_path)
            
            if has_audio:
                # Extract and merge audio
                self._merge_with_audio(input_path, temp_video, output_path)
            else:
                import shutil
                shutil.move(str(temp_video), str(output_path))
        
        print(f"Upscaled video saved to: {output_path}")
        return output_path
    
    def _check_audio(self, video_path: Path) -> bool:
        """Check if video has audio stream."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(video_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return bool(result.stdout.strip())
        except Exception:
            return False
    
    def _merge_with_audio(
        self,
        original_path: Path,
        video_path: Path,
        output_path: Path,
    ) -> None:
        """Merge upscaled video with original audio."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(original_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)


# Convenience function
def upscale_video(
    input_path: Path,
    output_path: Optional[Path] = None,
    scale: int = 4,
) -> Path:
    """
    Quick function to upscale a video.
    
    Args:
        input_path: Input video path
        output_path: Output path (auto-generated if not provided)
        scale: Upscaling factor (default 4x)
    
    Returns:
        Path to upscaled video
    """
    config = get_config()
    
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}"
    
    upscaler = RealESRGANUpscaler()
    upscaler.config.scale = scale
    
    try:
        return upscaler.upscale_video(input_path, output_path)
    finally:
        upscaler.unload_model()


if __name__ == "__main__":
    print("Real-ESRGAN upscaler module loaded. Requires video file to test.")
