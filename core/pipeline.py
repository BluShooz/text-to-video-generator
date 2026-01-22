"""
Main Pipeline Orchestration
Coordinates all AI modules for end-to-end text-to-video generation.
"""

import asyncio
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import traceback

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.config import get_config
from core.utils import (
    generate_job_id, ensure_dir, clean_temp_files, clear_gpu_memory
)
from modules.video_generator import CogVideoGenerator
from modules.tts import XTTSGenerator
from modules.lip_sync import Wav2LipSync
from modules.upscaler import RealESRGANUpscaler


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    GENERATING_VIDEO = "generating_video"
    GENERATING_AUDIO = "generating_audio"
    LIP_SYNCING = "lip_syncing"
    UPSCALING = "upscaling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationJob:
    """Represents a video generation job."""
    job_id: str
    prompt: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_step: str = ""
    output_path: Optional[Path] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Generation options
    duration: int = 6
    upscale: bool = False
    speaker_wav: Optional[str] = None
    language: str = "en"
    seed: Optional[int] = None


class TextToVideoPipeline:
    """
    Main pipeline for text-to-video generation with lip-sync.
    
    Workflow:
    1. Generate video from text prompt (CogVideoX)
    2. Generate speech from text (XTTS-v2)
    3. Synchronize lips to speech (Wav2Lip)
    4. Optional: Upscale video (Real-ESRGAN)
    """
    
    def __init__(self):
        self.config = get_config()
        self.jobs: Dict[str, GenerationJob] = {}
        
        # Lazy-loaded modules
        self._video_generator: Optional[CogVideoGenerator] = None
        self._tts_generator: Optional[XTTSGenerator] = None
        self._lip_syncer: Optional[Wav2LipSync] = None
        self._upscaler: Optional[RealESRGANUpscaler] = None
    
    @property
    def video_generator(self) -> CogVideoGenerator:
        if self._video_generator is None:
            self._video_generator = CogVideoGenerator()
        return self._video_generator
    
    @property
    def tts_generator(self) -> XTTSGenerator:
        if self._tts_generator is None:
            self._tts_generator = XTTSGenerator()
        return self._tts_generator
    
    @property
    def lip_syncer(self) -> Wav2LipSync:
        if self._lip_syncer is None:
            self._lip_syncer = Wav2LipSync()
        return self._lip_syncer
    
    @property
    def upscaler(self) -> RealESRGANUpscaler:
        if self._upscaler is None:
            self._upscaler = RealESRGANUpscaler()
        return self._upscaler
    
    def create_job(
        self,
        prompt: str,
        duration: int = 6,
        upscale: bool = False,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        seed: Optional[int] = None,
    ) -> GenerationJob:
        """Create a new generation job."""
        job = GenerationJob(
            job_id=generate_job_id(),
            prompt=prompt,
            duration=duration,
            upscale=upscale,
            speaker_wav=speaker_wav,
            language=language,
            seed=seed,
        )
        self.jobs[job.job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def _update_job(
        self,
        job: GenerationJob,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        step: Optional[str] = None,
    ) -> None:
        """Update job status and progress."""
        if status:
            job.status = status
        if progress is not None:
            job.progress = progress
        if step:
            job.current_step = step
    
    async def generate(
        self,
        job: GenerationJob,
        progress_callback: Optional[Callable[[GenerationJob], None]] = None,
    ) -> Path:
        """
        Execute the full generation pipeline.
        
        Args:
            job: The generation job to process
            progress_callback: Optional callback for progress updates
        
        Returns:
            Path to the final video
        """
        config = self.config
        ensure_dir(config.outputs_dir)
        
        # Create job-specific temp directory
        temp_dir = config.temp_dir / job.job_id
        ensure_dir(temp_dir)
        
        try:
            # ==================== STEP 1: Generate Video ====================
            self._update_job(
                job,
                status=JobStatus.GENERATING_VIDEO,
                progress=0.0,
                step="Generating video from prompt..."
            )
            if progress_callback:
                progress_callback(job)
            
            raw_video_path = temp_dir / "raw_video.mp4"
            
            def video_progress(step: int, total: int):
                job.progress = (step / total) * 25  # 0-25%
                if progress_callback:
                    progress_callback(job)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.video_generator.generate_to_file(
                    prompt=job.prompt,
                    output_path=raw_video_path,
                    num_frames=job.duration * 8,  # 8 fps
                    seed=job.seed,
                    progress_callback=video_progress,
                )
            )
            
            # Free video generator memory
            self.video_generator.unload_model()
            clear_gpu_memory()
            
            # ==================== STEP 2: Generate Audio ====================
            self._update_job(
                job,
                status=JobStatus.GENERATING_AUDIO,
                progress=25.0,
                step="Generating speech..."
            )
            if progress_callback:
                progress_callback(job)
            
            audio_path = temp_dir / "speech.wav"
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.tts_generator.synthesize(
                    text=job.prompt,
                    output_path=audio_path,
                    speaker_wav=job.speaker_wav,
                    language=job.language,
                )
            )
            
            job.progress = 50.0
            if progress_callback:
                progress_callback(job)
            
            # Free TTS memory
            self.tts_generator.unload_model()
            clear_gpu_memory()
            
            # ==================== STEP 3: Lip Sync ====================
            self._update_job(
                job,
                status=JobStatus.LIP_SYNCING,
                progress=50.0,
                step="Synchronizing lip movements..."
            )
            if progress_callback:
                progress_callback(job)
            
            synced_video_path = temp_dir / "synced_video.mp4"
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.lip_syncer.synchronize(
                    video_path=raw_video_path,
                    audio_path=audio_path,
                    output_path=synced_video_path,
                )
            )
            
            job.progress = 75.0
            if progress_callback:
                progress_callback(job)
            
            # Free lip sync memory
            self.lip_syncer.unload_model()
            clear_gpu_memory()
            
            # ==================== STEP 4: Upscale (Optional) ====================
            if job.upscale:
                self._update_job(
                    job,
                    status=JobStatus.UPSCALING,
                    progress=75.0,
                    step="Upscaling to HD..."
                )
                if progress_callback:
                    progress_callback(job)
                
                final_output = config.outputs_dir / f"{job.job_id}_final_HD.mp4"
                
                def upscale_progress(current: int, total: int):
                    job.progress = 75 + (current / total) * 25  # 75-100%
                    if progress_callback:
                        progress_callback(job)
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.upscaler.upscale_video(
                        input_path=synced_video_path,
                        output_path=final_output,
                        progress_callback=upscale_progress,
                    )
                )
                
                self.upscaler.unload_model()
                clear_gpu_memory()
            else:
                # Copy synced video to output
                import shutil
                final_output = config.outputs_dir / f"{job.job_id}_final.mp4"
                shutil.copy(str(synced_video_path), str(final_output))
            
            # ==================== Complete ====================
            self._update_job(
                job,
                status=JobStatus.COMPLETED,
                progress=100.0,
                step="Complete!"
            )
            job.output_path = final_output
            job.completed_at = datetime.now()
            
            if progress_callback:
                progress_callback(job)
            
            # Cleanup temp files
            if config.cleanup_temp_files:
                clean_temp_files(config.temp_dir, job.job_id)
            
            return final_output
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.current_step = f"Failed: {str(e)}"
            
            print(f"Generation failed: {e}")
            traceback.print_exc()
            
            if progress_callback:
                progress_callback(job)
            
            raise
    
    def generate_sync(
        self,
        prompt: str,
        **kwargs
    ) -> Path:
        """
        Synchronous generation for simple use cases.
        
        Args:
            prompt: Text prompt
            **kwargs: Additional options passed to create_job
        
        Returns:
            Path to generated video
        """
        job = self.create_job(prompt, **kwargs)
        
        # Run async in sync context
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.generate(job))
        finally:
            loop.close()


# Global pipeline instance
_pipeline: Optional[TextToVideoPipeline] = None


def get_pipeline() -> TextToVideoPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = TextToVideoPipeline()
    return _pipeline


# Convenience function
def generate_video_with_lipsync(
    prompt: str,
    upscale: bool = False,
    speaker_wav: Optional[str] = None,
    language: str = "en",
) -> Path:
    """
    Quick function to generate a lip-synced video.
    
    Args:
        prompt: Text describing what the person should say
        upscale: Whether to upscale to HD
        speaker_wav: Optional voice reference for cloning
        language: Language code
    
    Returns:
        Path to generated video
    """
    pipeline = get_pipeline()
    return pipeline.generate_sync(
        prompt=prompt,
        upscale=upscale,
        speaker_wav=speaker_wav,
        language=language,
    )


if __name__ == "__main__":
    # Quick test
    test_prompt = "Hello! Welcome to our AI-powered video demonstration. This technology is truly remarkable."
    print(f"Testing pipeline with prompt: {test_prompt}")
    
    try:
        output = generate_video_with_lipsync(test_prompt)
        print(f"Video generated: {output}")
    except Exception as e:
        print(f"Test failed: {e}")
