"""
CogVideoX Video Generation Module
Using HuggingFace Diffusers for text-to-video synthesis.
"""

import torch
from pathlib import Path
from typing import Optional, List, Callable
import numpy as np

from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.config import VideoGeneratorConfig, get_config
from core.utils import get_device, get_torch_dtype, ensure_dir, clear_gpu_memory


class CogVideoGenerator:
    """
    CogVideoX-based video generator.
    Generates video frames from text prompts using diffusion models.
    """
    
    def __init__(
        self,
        config: Optional[VideoGeneratorConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or get_config().video
        self.device = device or get_device()
        self.dtype = get_torch_dtype(get_config().torch_dtype)
        self.pipeline: Optional[CogVideoXPipeline] = None
        self._loaded = False
    
    def load_model(self) -> None:
        """Load the CogVideoX model into memory."""
        if self._loaded:
            return
        
        print(f"Loading CogVideoX model: {self.config.model_id}")
        
        # Load pipeline with memory optimizations
        self.pipeline = CogVideoXPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
        )
        
        # Apply memory optimizations
        if self.config.enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)
        
        if self.config.enable_vae_tiling:
            self.pipeline.vae.enable_tiling()
        
        # Use DPM scheduler for faster inference
        self.pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        self._loaded = True
        print("CogVideoX model loaded successfully")
    
    def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self._loaded = False
        clear_gpu_memory()
        print("CogVideoX model unloaded")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        """
        Generate video frames from a text prompt.
        
        Args:
            prompt: Text description of the video to generate
            negative_prompt: What to avoid in the generation
            num_frames: Number of frames to generate
            height: Video height in pixels
            width: Video width in pixels
            guidance_scale: Classifier-free guidance strength
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            progress_callback: Callback function(step, total_steps)
        
        Returns:
            List of numpy arrays representing video frames (RGB, uint8)
        """
        self.load_model()
        
        # Use config defaults if not specified
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        
        # Default negative prompt
        if negative_prompt is None:
            negative_prompt = (
                "low quality, blurry, distorted, artifacts, "
                "bad anatomy, ugly, poorly drawn"
            )
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"Generating video: {num_frames} frames at {width}x{height}")
        
        # Create callback wrapper
        def diffuser_callback(pipe, step, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step, num_inference_steps)
            return callback_kwargs
        
        # Generate video
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            callback_on_step_end=diffuser_callback if progress_callback else None,
        )
        
        # Convert to numpy frames
        frames = output.frames[0]  # Shape: (num_frames, H, W, C)
        
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        # Ensure uint8 format
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)
        
        # Convert to list of frames
        frame_list = [frames[i] for i in range(frames.shape[0])]
        
        print(f"Generated {len(frame_list)} frames")
        return frame_list
    
    def generate_to_file(
        self,
        prompt: str,
        output_path: Path,
        **kwargs,
    ) -> Path:
        """
        Generate video and save directly to file.
        
        Args:
            prompt: Text description
            output_path: Path to save the video
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            Path to the generated video file
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        frames = self.generate(prompt, **kwargs)
        
        # Export to video
        export_to_video(frames, str(output_path), fps=self.config.fps)
        
        print(f"Video saved to: {output_path}")
        return output_path


# Convenience function for quick generation
def generate_video(
    prompt: str,
    output_path: Optional[Path] = None,
    **kwargs,
) -> Path:
    """
    Quick function to generate a video from text.
    
    Args:
        prompt: Text description
        output_path: Where to save (auto-generated if not provided)
        **kwargs: Additional generation parameters
    
    Returns:
        Path to generated video
    """
    config = get_config()
    
    if output_path is None:
        from core.utils import generate_job_id
        job_id = generate_job_id()
        output_path = config.outputs_dir / f"{job_id}_raw.mp4"
    
    generator = CogVideoGenerator()
    try:
        return generator.generate_to_file(prompt, output_path, **kwargs)
    finally:
        generator.unload_model()


if __name__ == "__main__":
    # Quick test
    test_prompt = "A person standing and talking to the camera, professional lighting, studio background"
    output = generate_video(test_prompt)
    print(f"Test video generated: {output}")
