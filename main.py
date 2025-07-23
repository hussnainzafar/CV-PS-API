from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import replicate
import requests
from PIL import Image, ImageDraw
import io
import base64
import os
from typing import Dict, Any
import uvicorn

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Replicate Facial Landmarks Detection API", version="1.0.0")

# Mount mediapipe nose-only detection endpoint
from mediapipe_nose import router as mediapipe_nose_router
app.include_router(mediapipe_nose_router)

# Initialize Replicate client
if not os.getenv("REPLICATE_API_TOKEN"):
    raise ValueError("REPLICATE_API_TOKEN environment variable is required")

def upload_image_to_replicate(image_file) -> str:
    """Upload image to a temporary hosting service and return URL"""
    # For demo purposes, we'll use base64 encoding
    # In production, you might want to use a proper image hosting service
    image = Image.open(io.BytesIO(image_file))
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def detect_landmarks_with_replicate(image_url: str) -> Dict[str, Any]:
    """Use Replicate to detect facial landmarks using chigozienri/mediapipe-face"""
    try:
        output = replicate.run(
            "chigozienri/mediapipe-face:b52b4833a810a8b8d835d6339b72536d63590918b185588be2def78a89e7ca7b",
            input={
                "images": image_url
            }
        )
        return output
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image with Replicate: {str(e)}"
        )

def draw_landmarks_on_image(image: Image.Image, landmarks_data: Dict[str, Any]) -> Image.Image:
    """Draw colored dots on detected facial landmarks"""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Parse landmarks data (format depends on the model used)
    if 'faces' in landmarks_data:
        for face in landmarks_data['faces']:
            # Draw bounding box
            if 'bbox' in face:
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
            
            # Draw landmarks if available
            if 'landmarks' in face:
                landmarks = face['landmarks']
                
                # Draw eyes (blue and red)
                if 'left_eye' in landmarks:
                    for point in landmarks['left_eye']:
                        x, y = point
                        draw.ellipse([x-3, y-3, x+3, y+3], fill="blue")
                
                if 'right_eye' in landmarks:
                    for point in landmarks['right_eye']:
                        x, y = point
                        draw.ellipse([x-3, y-3, x+3, y+3], fill="red")
                
                # Draw nose (green)
                if 'nose' in landmarks:
                    for point in landmarks['nose']:
                        x, y = point
                        draw.ellipse([x-3, y-3, x+3, y+3], fill="green")
                
                # Draw lips (cyan)
                if 'mouth' in landmarks:
                    for point in landmarks['mouth']:
                        x, y = point
                        draw.ellipse([x-3, y-3, x+3, y+3], fill="cyan")
    
    return image

@app.get("/")
async def root():
    return {
        "message": "Replicate Facial Landmarks Detection API",
        "description": "Upload an image to detect facial landmarks using Replicate.ai models",
        "endpoints": {
            "/detect-landmarks": "POST - Upload an image to detect facial landmarks",
            "/health": "GET - Health check",
            "/models": "GET - List available models"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "replicate_configured": bool(os.getenv("REPLICATE_API_TOKEN"))}

@app.get("/models")
async def list_models():
    """List available Replicate models for face detection"""
    return {
        "recommended_models": [
            {
                "name": "Face Detection with Landmarks",
                "model": "andreasjansson/face-detection",
                "description": "Detects faces and facial landmarks"
            },
            {
                "name": "BLIP Face Analysis", 
                "model": "salesforce/blip",
                "description": "General purpose vision model with face detection capabilities"
            }
        ]
    }

@app.post("/detect-landmarks")
async def detect_facial_landmarks(file: UploadFile = File(...)):
    """
    Upload an image and get back the same image with facial landmarks marked using Mediapipe (OpenCV+Mediapipe backend).
    - Detects nose, eyes, lips, and face outline
    - Returns image with colored landmark dots
    """
    import numpy as np
    import mediapipe as mp
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        np_image = np.array(image)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        results = face_mesh.process(np_image)

        if not results.multi_face_landmarks:
            raise HTTPException(status_code=404, detail="No face detected")

        draw = ImageDraw.Draw(image)
        h, w = np_image.shape[:2]
        # Draw only 5 key nose landmarks: tip, left nostril, right nostril, upper bridge, lower bridge
        NOSE_LANDMARKS = [1, 2, 98, 327, 168]  # Mediapipe FaceMesh indices
        for face_landmarks in results.multi_face_landmarks:
            for idx in NOSE_LANDMARKS:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                draw.ellipse((x-4, y-4, x+4, y+4), fill="green")

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=nose_landmarks_detected.jpg"})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-landmarks-advanced")
async def detect_facial_landmarks_advanced(
    file: UploadFile = File(...),
    model_name: str = "andreasjansson/face-detection"
):
    """
    Advanced endpoint allowing model selection
    """
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_url = upload_image_to_replicate(contents)
        
        # Use specified model
        output = replicate.run(
            model_name,
            input={"image": image_url}
        )
        
        # Process results and draw landmarks
        result_image = draw_landmarks_on_image(image.copy(), output)
        
        img_bytes = io.BytesIO()
        result_image.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        
        return StreamingResponse(
            img_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=replicate_{model_name.replace('/', '_')}_landmarks.jpg"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with model {model_name}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
