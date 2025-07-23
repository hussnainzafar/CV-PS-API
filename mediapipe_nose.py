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
    # Use 5 key nose points: tip, left/right nostril, upper/lower bridge
    NOSE_LANDMARKS = [1, 2, 98, 327, 168]  # tip, left nostril, right nostril, upper bridge, lower bridge
    h, w = np_image.shape[:2]
    for face_landmarks in results.multi_face_landmarks:
        for idx in NOSE_LANDMARKS:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            draw.ellipse((x-4, y-4, x+4, y+4), fill="green")

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=nose_detected.jpg"})
