<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-00d9ff?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🎬 Text-to-Video Generator</h1>

<p align="center">
  <strong>AI-powered text-to-video generation with neural lip-sync capabilities</strong>
</p>

<p align="center">
  Transform text prompts into professional talking-head videos with accurate lip synchronization.
  <br>
  Powered by CogVideoX, XTTS-v2, Wav2Lip, and Real-ESRGAN.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API</a> •
  <a href="#deployment">Deployment</a>
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎬 **Text-to-Video** | Generate video from text using CogVideoX diffusion models |
| 🎤 **Neural TTS** | High-quality speech synthesis with XTTS-v2 (17 languages) |
| 👄 **Lip Sync** | Accurate lip synchronization using Wav2Lip GAN |
| 📺 **HD Upscaling** | 4x video enhancement with Real-ESRGAN |
| 🎭 **Voice Cloning** | Clone any voice from a 6-second audio sample |
| 🌐 **REST API** | Production-ready FastAPI backend |
| 💻 **Modern UI** | Beautiful web interface with real-time progress |
| 🐳 **Docker Ready** | One-command deployment with docker-compose |

## 🖥️ Demo

> **Live Demo**: [text-to-video-generator.vercel.app](https://text-to-video-generator.vercel.app)

<p align="center">
  <em>Enter your text prompt and watch AI generate a lip-synced video!</em>
</p>

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Text Prompt                               │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│   CogVideoX       │                     │   XTTS-v2         │
│   Video Gen       │                     │   Speech Gen      │
└───────────────────┘                     └───────────────────┘
        │                                           │
        │         ┌─────────────────────┐          │
        └────────►│     Wav2Lip         │◄─────────┘
                  │     Lip Sync        │
                  └─────────────────────┘
                              │
                              ▼
                  ┌─────────────────────┐
                  │   Real-ESRGAN       │ (Optional)
                  │   Upscaling         │
                  └─────────────────────┘
                              │
                              ▼
                  ┌─────────────────────┐
                  │   Final MP4         │
                  │   Download          │
                  └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10-3.11
- NVIDIA GPU with 12GB+ VRAM (RTX 3060 or better)
- CUDA 11.8+
- FFmpeg

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/text-to-video-generator.git
cd text-to-video-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model checkpoints
python scripts/download_models.py

# Copy environment config
cp .env.example .env

# Start the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

## 📖 API Reference

### Generate Video

```bash
POST /api/generate
```

**Request Body:**

```json
{
  "prompt": "Hello! Welcome to our AI demonstration.",
  "duration": 6,
  "language": "en",
  "upscale": true
}
```

**Response:**

```json
{
  "job_id": "job_20260121_123456_abc12345",
  "status": "pending",
  "message": "Video generation started"
}
```

### Check Status

```bash
GET /api/status/{job_id}
```

**Response:**

```json
{
  "job": {
    "job_id": "job_20260121_123456_abc12345",
    "status": "generating_video",
    "progress": 35.5,
    "current_step": "Generating video frames..."
  }
}
```

### Download Video

```bash
GET /api/download/{job_id}
```

Returns the generated MP4 video file.

### Health Check

```bash
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 4090",
  "gpu_memory_gb": 24.0
}
```

## 🌍 Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `ru` | Russian |
| `es` | Spanish | `nl` | Dutch |
| `fr` | French | `cs` | Czech |
| `de` | German | `ar` | Arabic |
| `it` | Italian | `zh-cn` | Chinese |
| `pt` | Portuguese | `ko` | Korean |
| `pl` | Polish | `ja` | Japanese |
| `tr` | Turkish | `hi` | Hindi |

## 📁 Project Structure

```
text-to-video-generator/
├── api/                    # FastAPI backend
│   ├── main.py             # Application entry point
│   ├── routes/             # API endpoints
│   └── models/             # Pydantic schemas
├── core/                   # Pipeline orchestration
│   ├── config.py           # Configuration management
│   ├── pipeline.py         # Main workflow
│   └── utils.py            # Utility functions
├── modules/                # AI model wrappers
│   ├── video_generator/    # CogVideoX integration
│   ├── tts/                # XTTS-v2 integration
│   ├── lip_sync/           # Wav2Lip integration
│   └── upscaler/           # Real-ESRGAN integration
├── frontend/               # Web interface
│   ├── index.html          # Main HTML
│   ├── styles/             # CSS styles
│   └── scripts/            # JavaScript
├── scripts/                # Utility scripts
│   └── download_models.py  # Model downloader
├── outputs/                # Generated videos
├── checkpoints/            # Model weights
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
└── docker-compose.yml      # Docker orchestration
```

## ⚙️ Configuration

Configuration is managed through environment variables. Copy `.env.example` to `.env` and customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_API_PORT` | `8000` | API server port |
| `APP_TORCH_DTYPE` | `float16` | Model precision |
| `VIDEO_MODEL_ID` | `THUDM/CogVideoX-2b` | Video model |
| `VIDEO_HEIGHT` | `480` | Output height |
| `VIDEO_WIDTH` | `720` | Output width |
| `TTS_LANGUAGE` | `en` | Default language |
| `UPSCALE_SCALE` | `4` | Upscale factor |

## 🎮 GPU Requirements

| Configuration | VRAM Required | Recommended GPU |
|---------------|---------------|-----------------|
| Minimum | 12GB | RTX 3060 |
| Standard | 16GB | RTX 4070 |
| Optimal | 24GB+ | RTX 4090, A100 |

The pipeline automatically manages GPU memory by loading/unloading modules sequentially.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CogVideoX](https://github.com/THUDM/CogVideo) - Video generation model
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Text-to-speech
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) - Lip synchronization
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Video upscaling

---

<p align="center">
  Made with ❤️ by AI-powered development
</p>
