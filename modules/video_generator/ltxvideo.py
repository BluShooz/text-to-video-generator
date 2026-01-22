"""
LTX-Video Generation Module
Pro-level text-to-video synthesis using the LTX-Video DiT model.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Callable, Union

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.ltxvideo_core.pipelines.pipeline_ltx_video import LTXVideoPipeline
from modules.ltxvideo_core.schedulers.rf_scheduler import RFScheduler
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from core.config import get_config
from core.utils import get_device, get_torch_dtype, ensure_dir, clear_gpu_memory


class LTXVideoGenerator:
    """
    LTX-Video based video generator.
    Superior quality and synchronization compared to CogVideoX.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
    ):
        self.config = get_config().video
        self.device = device or get_device()
        self.dtype = get_torch_dtype(get_config().torch_dtype)
        self.pipeline: Optional[LTXVideoPipeline] = None
        self._loaded = False
        
    def load_model(self) -> None:
        """Load LTX-Video model into memory."""
        if self._loaded:
            return
            
        print(f"Loading LTX-Video model...")
        
        # Paths for weights (assumes they will be in checkpoints/ltx-video)
        model_id = "Lightricks/LTX-Video"
        
        # Load pipeline (this follows the structure in LTX-Video research repo)
        # Note: In a real implementation, we'd use the local modules we copied
        self.pipeline = LTXVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        )
        
        if self.config.enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)
            
        self._loaded = True
        print("LTX-Video model loaded successfully")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 121, # LTX-Video optimal
        height: int = 480,
        width: int = 704,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        """Generate high-quality video frames."""
        self.load_model()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        print(f"Generating LTX-Video: {num_frames} frames, {width}x{height}")
        
        # Create callback wrapper
        def ltx_callback(step, timestep, latents):
            if progress_callback:
                progress_callback(step, num_inference_steps)
        
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=ltx_callback if progress_callback else None,
        )
        
        frames = output.frames[0] # List of PIL images or numpy
        
        # Ensure numpy RGB uint8
        frame_list = []
        for frame in frames:
            f = np.array(frame)
            if f.dtype != np.uint8:
                f = (f * 255).astype(np.uint8)
            frame_list.append(f)
            
        print(f"Generated {len(frame_list)} frames with LTX-Video")
        return frame_list

    def unload_model(self) -> None:
        """Unload and clear memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        self._loaded = False
        clear_gpu_memory()


def generate_video(prompt: str, output_path: Optional[Path] = None, **kwargs) -> Path:
    """Quick function for LTX-Video generation."""
    from core.utils import generate_job_id, ensure_dir
    from diffusers.utils import export_to_video
    
    config = get_config()
    if output_path is None:
        output_path = config.outputs_dir / f"{generate_job_id()}_ltx_raw.mp4"
    ensure_dir(output_path.parent)
    
    gen = LTXVideoGenerator()
    try:
        frames = gen.generate(prompt, **kwargs)
        export_to_video(frames, str(output_path), fps=config.video.fps)
        return output_path
    finally:
        gen.unload_model()
