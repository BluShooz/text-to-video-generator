# Contributing to Text-to-Video Generator

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## 🚀 Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes and test them
6. Submit a Pull Request

## 💻 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/text-to-video-generator.git
cd text-to-video-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (including dev dependencies)
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8

# Download models (for testing)
python scripts/download_models.py

# Run tests
pytest tests/ -v
```

## ✏️ Making Changes

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(tts): add support for new language`
- `fix(pipeline): resolve memory leak in video generation`
- `docs(readme): update installation instructions`

## 🔄 Pull Request Process

1. **Update documentation** if you've changed functionality
2. **Add tests** for new features
3. **Run linting** with `flake8` and `black`
4. **Ensure all tests pass**
5. **Update the README** if needed
6. **Request review** from maintainers

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
```

## 🎨 Style Guidelines

### Python

- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters (Black default)
- Use docstrings for all public functions

```python
def generate_video(
    prompt: str,
    duration: int = 6,
    upscale: bool = False,
) -> Path:
    """
    Generate a video from text prompt.
    
    Args:
        prompt: Text description of the video
        duration: Video length in seconds
        upscale: Whether to upscale to HD
    
    Returns:
        Path to the generated video file
    """
    ...
```

### JavaScript

- Use modern ES6+ syntax
- Use meaningful variable names
- Comment complex logic

### CSS

- Use CSS variables for theming
- Follow BEM naming convention
- Mobile-first responsive design

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v
```

## 📝 Documentation

- Use clear, concise language
- Include code examples
- Update README for user-facing changes
- Add docstrings for new functions/classes

---

Thank you for contributing! 🙏
