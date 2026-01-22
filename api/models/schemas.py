"""
API Data Models
Pydantic schemas for request/response validation.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for video generation."""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text prompt describing what the person should say"
    )
    duration: int = Field(
        default=6,
        ge=2,
        le=30,
        description="Video duration in seconds"
    )
    upscale: bool = Field(
        default=False,
        description="Whether to upscale output to HD"
    )
    pro: bool = Field(
        default=True,
        description="Whether to use SOTA Pro-Level pipeline (LTX-Video + MuseTalk + LivePortrait)"
    )
    speaker_wav: Optional[str] = Field(
        default=None,
        description="Path to voice reference audio for cloning"
    )
    language: str = Field(
        default="en",
        description="Language code (en, es, fr, de, etc.)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Hello! Welcome to our demonstration of AI-powered video generation.",
                    "duration": 6,
                    "upscale": True,
                    "pro": True,
                    "language": "en"
                }
            ]
        }
    }


class JobResponse(BaseModel):
    """Response containing job information."""
    job_id: str
    status: str
    progress: float = Field(ge=0, le=100)
    current_step: str
    output_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class GenerateResponse(BaseModel):
    """Response after submitting a generation request."""
    job_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """Response for job status query."""
    job: JobResponse


class DownloadInfo(BaseModel):
    """Response with download information."""
    job_id: str
    filename: str
    size_bytes: int
    duration_seconds: float
    resolution: str
    download_url: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None


class LanguagesResponse(BaseModel):
    """Available languages response."""
    languages: List[str]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
