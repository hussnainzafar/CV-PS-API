import io
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
import numpy as np
import mediapipe as mp

router = APIRouter()

@router.post("/detect-nose")
async def detect_nose(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    np_image = np.array(image)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(np_image)

    if not results.multi_face_landmarks:
        raise HTTPException(status_code=404, detail="No face detected")

    draw = ImageDraw.Draw(image)
    # Nose tip and nostrils (using Mediapipe's FaceMesh landmark indices)
    NOSE_LANDMARKS = [1, 2, 98, 327, 168, 197, 195, 5, 4, 19, 94, 2]  # tip, nostrils, bridge
    h, w = np_image.shape[:2]
    for face_landmarks in results.multi_face_landmarks:
        for idx in NOSE_LANDMARKS:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            draw.ellipse((x-3, y-3, x+3, y+3), fill="green")

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=nose_detected.jpg"})
