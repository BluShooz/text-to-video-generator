"""
Video Generation Routes
API endpoints for submitting and managing generation jobs.
"""

import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core import get_pipeline, GenerationJob, JobStatus
from api.models import (
    GenerateRequest,
    GenerateResponse,
    JobResponse,
    StatusResponse,
    DownloadInfo,
    ErrorResponse,
)

router = APIRouter(prefix="/api", tags=["generation"])


def job_to_response(job: GenerationJob, base_url: str = "") -> JobResponse:
    """Convert GenerationJob to API response."""
    output_url = None
    if job.output_path and job.output_path.exists():
        output_url = f"{base_url}/api/download/{job.job_id}"
    
    return JobResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        current_step=job.current_step,
        output_url=output_url,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def generate_video(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a new video generation request.
    
    The video will be generated in the background. Use the returned
    job_id to check status and download the result when complete.
    """
    pipeline = get_pipeline()
    
    # Create job
    job = pipeline.create_job(
        prompt=request.prompt,
        duration=request.duration,
        upscale=request.upscale,
        speaker_wav=request.speaker_wav,
        language=request.language,
        seed=request.seed,
    )
    
    # Start background generation
    async def run_generation():
        try:
            await pipeline.generate(job)
        except Exception as e:
            print(f"Background generation failed: {e}")
    
    background_tasks.add_task(
        lambda: asyncio.create_task(run_generation())
    )
    
    return GenerateResponse(
        job_id=job.job_id,
        status=job.status.value,
        message="Video generation started. Use the job_id to check status.",
    )


@router.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_job_status(job_id: str):
    """Get the status of a generation job."""
    pipeline = get_pipeline()
    job = pipeline.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return StatusResponse(job=job_to_response(job))


@router.get(
    "/download/{job_id}",
    responses={
        404: {"model": ErrorResponse},
        202: {"model": ErrorResponse},
    },
)
async def download_video(job_id: str):
    """
    Download the generated video.
    
    Returns 404 if job not found, 202 if still processing.
    """
    pipeline = get_pipeline()
    job = pipeline.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=202,
            detail=f"Job is still processing: {job.status.value}"
        )
    
    if job.output_path is None or not job.output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Output file not found"
        )
    
    return FileResponse(
        path=job.output_path,
        media_type="video/mp4",
        filename=job.output_path.name,
    )


@router.get(
    "/download/{job_id}/info",
    response_model=DownloadInfo,
    responses={404: {"model": ErrorResponse}},
)
async def get_download_info(job_id: str):
    """Get information about the generated video."""
    pipeline = get_pipeline()
    job = pipeline.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.COMPLETED or job.output_path is None:
        raise HTTPException(status_code=404, detail="Output not available")
    
    from core.utils import get_video_info
    info = get_video_info(job.output_path)
    
    return DownloadInfo(
        job_id=job.job_id,
        filename=job.output_path.name,
        size_bytes=job.output_path.stat().st_size,
        duration_seconds=info.get("duration", 0),
        resolution=f"{info.get('width', 0)}x{info.get('height', 0)}",
        download_url=f"/api/download/{job_id}",
    )


@router.get("/jobs")
async def list_jobs():
    """List all jobs."""
    pipeline = get_pipeline()
    return {
        "jobs": [
            job_to_response(job)
            for job in pipeline.jobs.values()
        ]
    }


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output."""
    pipeline = get_pipeline()
    job = pipeline.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Delete output file if exists
    if job.output_path and job.output_path.exists():
        job.output_path.unlink()
    
    # Remove from jobs dict
    del pipeline.jobs[job_id]
    
    return {"message": f"Job {job_id} deleted"}
