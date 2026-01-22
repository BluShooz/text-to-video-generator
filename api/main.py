"""
Text-to-Video Generator API
FastAPI application with REST endpoints and static file serving.
"""

import torch
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core import get_config
from api.routes import generate_router
from api.models import HealthResponse, LanguagesResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    config = get_config()
    print(f"🚀 Text-to-Video Generator API starting...")
    print(f"   Output directory: {config.outputs_dir}")
    print(f"   GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    
    # Create required directories
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")
    torch.cuda.empty_cache()


# Create FastAPI app
app = FastAPI(
    title="Text-to-Video Generator",
    description="AI-powered text-to-video generation with lip-sync",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(generate_router)

# Static files for frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    
    # Fallback HTML if frontend not built
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text-to-Video Generator</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #6366f1; }
            .status { background: #f0fdf4; padding: 15px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h1>🎬 Text-to-Video Generator</h1>
        <div class="status">
            <p><strong>API is running!</strong></p>
            <p>Visit <a href="/docs">/docs</a> for API documentation.</p>
        </div>
    </body>
    </html>
    """)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_name = None
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory,
    )


@app.get("/api/languages", response_model=LanguagesResponse)
async def get_languages():
    """Get list of supported TTS languages."""
    # XTTS-v2 supported languages
    languages = [
        "en",  # English
        "es",  # Spanish
        "fr",  # French
        "de",  # German
        "it",  # Italian
        "pt",  # Portuguese
        "pl",  # Polish
        "tr",  # Turkish
        "ru",  # Russian
        "nl",  # Dutch
        "cs",  # Czech
        "ar",  # Arabic
        "zh-cn",  # Chinese
        "hu",  # Hungarian
        "ko",  # Korean
        "ja",  # Japanese
        "hi",  # Hindi
    ]
    return LanguagesResponse(languages=languages)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True,
    )
