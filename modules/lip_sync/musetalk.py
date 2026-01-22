"""
MuseTalk 1.5 Lip-Sync Module
Professional-grade lip synchronization using latent space inpainting.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from omegaconf import OmegaConf

# Add core path for imports
import sys
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import moved musetalk core
from modules.musetalk_core.utils.utils import load_all_model, datagen
from modules.musetalk_core.utils.audio_processor import AudioProcessor
from modules.musetalk_core.utils.face_parsing import FaceParsing
from modules.musetalk_core.utils.blending import get_image
from modules.musetalk_core.utils.preprocessing import get_landmark_and_bbox

from core.config import get_config
from core.utils import get_device, clear_gpu_memory, ensure_dir


class MuseTalkSync:
    """
    MuseTalk 1.5 based lip-sync processor.
    Provides high-fidelity, real-time audio-driven lip synchronization.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        version: str = "v15",
    ):
        self.config = get_config().lip_sync
        self.device = device or get_device()
        self.version = version
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.fp = None
        self._loaded = False
        
    def load_model(self) -> None:
        """Load MuseTalk 1.5 models into memory."""
        if self._loaded:
            return
            
        print(f"Loading MuseTalk {self.version} models...")
        
        # Paths for weights
        models_dir = Path(project_root) / "checkpoints" / "musetalk"
        unet_path = models_dir / ("musetalkV15/unet.pth" if self.version == "v15" else "musetalk/pytorch_model.bin")
        unet_config = models_dir / "musetalk/musetalk.json"
        whisper_dir = models_dir / "whisper"
        
        # Load core models
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=str(unet_path),
            vae_type="sd-vae",
            unet_config=str(unet_config),
            device=self.device
        )
        
        # Memory optimizations
        if get_config().torch_dtype == "float16":
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
            
        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)
        
        # Initialize processors
        self.audio_processor = AudioProcessor(feature_extractor_path=str(whisper_dir))
        
        from transformers import WhisperModel
        self.whisper = WhisperModel.from_pretrained(str(whisper_dir))
        self.whisper = self.whisper.to(device=self.device, dtype=self.unet.model.dtype).eval()
        self.whisper.requires_grad_(False)
        
        # Face parsing for smooth blending
        if self.version == "v15":
            self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        else:
            self.fp = FaceParsing()
            
        self._loaded = True
        print(f"MuseTalk {self.version} loaded successfully")

    def sync(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        bbox_shift: int = 0,
        batch_size: int = 8,
        use_float16: bool = True,
    ) -> Path:
        """
        Synchronize video lips with audio.
        
        Args:
            video_path: Path to source video
            audio_path: Path to target audio
            output_path: Path to save result
            bbox_shift: Vertical shift for mouth crop (v1 only)
            batch_size: Inference batch size
            use_float16: Whether to use half precision
            
        Returns:
            Path to the synced video file
        """
        self.load_model()
        
        video_path = str(video_path)
        audio_path = str(audio_path)
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        # 1. Preprocess: Extract frames and detect face
        print("Preprocessing video frames and landmarks...")
        # Note: MuseTalk get_landmark_and_bbox expects image list or directory
        # We'll use cv2 to extract frames if passed a video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_list = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_list.append(frame)
        cap.release()
        
        # Temporary image list for Landmark detection
        temp_img_dir = Path(project_root) / "temp" / f"musetalk_frames_{os.getpid()}"
        ensure_dir(temp_img_dir)
        img_paths = []
        for i, frame in enumerate(frame_list):
            p = temp_img_dir / f"{i:08d}.png"
            cv2.imwrite(str(p), frame)
            img_paths.append(str(p))
            
        coord_list, _ = get_landmark_and_bbox(img_paths, bbox_shift if self.version == "v1" else 0)
        
        # 2. Process audio
        print("Processing audio features...")
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.unet.model.dtype,
            self.whisper,
            librosa_length,
            fps=fps
        )
        
        # 3. Generate latents for input crops
        print("Encoding face crops...")
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = min(y2 + 10, frame.shape[0]) # extra margin
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
            
        # 4. Inference
        print("Running MuseTalk inference...")
        # Simple cycle if video shorter than audio
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        timesteps = torch.tensor([0], device=self.device)
        
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            delay_frame=0,
            device=self.device
        )
        
        res_frame_list = []
        for whisper_batch, latent_batch in gen:
            audio_feature_batch = self.pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
            
            pred_latents = self.unet.model(
                latent_batch, 
                timesteps, 
                encoder_hidden_states=audio_feature_batch
            ).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
                
        # 5. Composite results
        print("Compositing final frames...")
        final_video_frames = []
        for i, res_frame in enumerate(res_frame_list):
            idx = i % len(frame_list)
            bbox = coord_list[idx]
            ori_frame = frame_list[idx].copy()
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = min(y2 + 10, ori_frame.shape[0])
            
            res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
            
            if self.version == "v15":
                combine_frame = get_image(ori_frame, res_frame_resized, [x1, y1, x2, y2], mode="jaw", fp=self.fp)
            else:
                combine_frame = get_image(ori_frame, res_frame_resized, [x1, y1, x2, y2], fp=self.fp)
            
            final_video_frames.append(combine_frame)
            
        # 6. Save video
        temp_vid = output_path.with_suffix(".temp.mp4")
        height, width, _ = final_video_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_vid), fourcc, fps, (width, height))
        for f in final_video_frames:
            out.write(f)
        out.release()
        
        # Merge audio with ffmpeg
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(temp_vid),
            "-i", str(audio_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-base_stream_index", "0",
            str(output_path)
        ]
        subprocess.run(cmd, check=True)
        
        # Cleanup
        if temp_vid.exists(): temp_vid.unlink()
        import shutil
        if temp_img_dir.exists(): shutil.rmtree(temp_img_dir)
        
        print(f"MuseTalk sync complete: {output_path}")
        return output_path

    def unload_model(self) -> None:
        """Free GPU memory."""
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self._loaded = False
        clear_gpu_memory()


# Convenience function
def sync_lip(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Optional[Path] = None,
    **kwargs
) -> Path:
    if output_path is None:
        from core.utils import generate_job_id
        output_path = get_config().outputs_dir / f"{generate_job_id()}_synced.mp4"
        
    syncer = MuseTalkSync()
    try:
        return syncer.sync(video_path, audio_path, output_path, **kwargs)
    finally:
        syncer.unload_model()
