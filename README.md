# Replicate Facial Landmarks Detection API

A FastAPI-based web service that uses **Replicate.ai** models to detect facial landmarks (nose, lips, and eyes) in uploaded images and returns the image with landmarks marked as colored dots.

## Why Replicate.ai?

- 🚀 **State-of-the-art models**: Access to the latest AI models without local setup
- ⚡ **High performance**: Cloud-based processing with optimized inference
- 🔄 **Always updated**: Models are maintained and updated by experts
- 💰 **Cost-effective**: Pay only for what you use
- 🛠️ **No maintenance**: No need to manage model weights or dependencies

## Features

- **Nose Detection**: Green dots mark nose landmarks
- **Eye Detection**: Blue dots for left eye, red dots for right eye  
- **Lip Detection**: Cyan dots mark lip landmarks
- **Multiple Models**: Support for different Replicate face detection models
- **RESTful API**: Simple POST endpoint for image processing
- **Docker Support**: Easy deployment with Docker

## Setup

### Prerequisites

1. Get a Replicate API token from [replicate.com](https://replicate.com)
2. Set the environment variable:

\`\`\`bash
export REPLICATE_API_TOKEN=your_token_here
\`\`\`

### Local Installation

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the server:
\`\`\`bash
python main.py
\`\`\`

### Docker Installation

1. Copy environment file:
\`\`\`bash
cp .env.example .env
# Edit .env and add your REPLICATE_API_TOKEN
\`\`\`

2. Build and run:
\`\`\`bash
docker-compose up --build
\`\`\`

## API Usage

### Endpoints

- `GET /` - API information
- `GET /health` - Health check  
- `GET /models` - List available Replicate models
- `POST /detect-landmarks` - Upload image for landmark detection
- `POST /detect-landmarks-advanced` - Advanced endpoint with model selection

### Example Usage

\`\`\`python
import requests

# Basic landmark detection
with open("face_image.jpg", "rb") as f:
    files = {"file": ("face_image.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/detect-landmarks", files=files)

# Save result
with open("result_with_landmarks.jpg", "wb") as f:
    f.write(response.content)
\`\`\`

### Advanced Usage with Model Selection

\`\`\`python
# Use specific model
with open("face_image.jpg", "rb") as f:
    files = {"file": ("face_image.jpg", f, "image/jpeg")}
    params = {"model_name": "andreasjansson/face-detection"}
    response = requests.post("http://localhost:8000/detect-landmarks-advanced", 
                           files=files, params=params)
\`\`\`

## Available Models

The API supports multiple Replicate models for face detection:

1. **andreasjansson/face-detection** - High-accuracy face detection with landmarks
2. **salesforce/blip** - BLIP model for face analysis  
3. **tencentarc/gfpgan** - Face restoration and analysis

## Landmark Colors

- **Nose**: 🟢 Green dots
- **Left Eye**: 🔵 Blue dots
- **Right Eye**: 🔴 Red dots  
- **Lips**: 🟡 Cyan dots

## Cost Optimization

- Images are processed on-demand
- No local GPU requirements
- Pay-per-use pricing model
- Efficient base64 encoding for image transfer

## Error Handling

- Invalid file type validation
- Replicate API error handling
- Fallback model support
- Comprehensive error messages
