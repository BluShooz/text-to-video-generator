"""
Utility functions for the Text-to-Video Generator.
"""

import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

import torch
import numpy as np
from PIL import Image


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_torch_dtype(dtype_str: str = "float16") -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)


def generate_job_id() -> str:
    """Generate a unique job ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"job_{timestamp}_{short_uuid}"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_temp_files(temp_dir: Path, job_id: str) -> None:
    """Remove temporary files for a completed job."""
    job_temp_dir = temp_dir / job_id
    if job_temp_dir.exists():
        shutil.rmtree(job_temp_dir)


def get_video_info(video_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        
        video_stream = next(
            (s for s in data.get("streams", []) if s["codec_type"] == "video"),
            {}
        )
        
        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "duration": float(data.get("format", {}).get("duration", 0)),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream.get("r_frame_rate") else 0,
            "codec": video_stream.get("codec_name", "unknown"),
        }
    except Exception:
        return {}


def frames_to_video(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 25,
    codec: str = "libx264",
    crf: int = 18,
) -> Path:
    """Convert numpy frames to video file using ffmpeg."""
    import imageio
    
    output_path = Path(output_path)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec=codec,
        quality=None,
        output_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
    )
    
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        writer.append_data(frame)
    
    writer.close()
    return output_path


def video_to_frames(
    video_path: Path,
    output_dir: Optional[Path] = None,
) -> Tuple[List[np.ndarray], int]:
    """Extract frames from video file."""
    import cv2
    
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if output_dir:
        output_dir = ensure_dir(output_dir)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(output_dir / f"frame_{i:06d}.png")
    
    return frames, fps


def merge_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    video_fps: Optional[int] = None,
) -> Path:
    """Merge audio and video streams using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_vram_usage(
    model_name: str,
    dtype: str = "float16",
) -> float:
    """Estimate VRAM usage in GB for a given model."""
    vram_estimates = {
        "cogvideox-2b": {"float16": 12.0, "float32": 24.0},
        "cogvideox-5b": {"float16": 16.0, "float32": 32.0},
        "xtts-v2": {"float16": 4.0, "float32": 8.0},
        "wav2lip": {"float16": 4.0, "float32": 8.0},
        "realesrgan": {"float16": 2.0, "float32": 4.0},
    }
    
    model_key = model_name.lower().replace("_", "-")
    for key, estimates in vram_estimates.items():
        if key in model_key:
            return estimates.get(dtype, estimates.get("float16", 8.0))
    
    return 8.0  # Default estimate
