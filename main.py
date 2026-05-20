from fastapi import FastAPI, File, UploadFile
import shutil
import os
import uuid
import numpy as np
import cv2
from insightface.app import FaceAnalysis

app = FastAPI()

# Load model ONCE (important)
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_file(file: UploadFile):
    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return path


def get_embedding(img_path):
    img = cv2.imread(img_path)
    faces = face_app.get(img)

    if len(faces) == 0:
        return None

    return faces[0].embedding


@app.post("/verify-id")
async def verify_id(selfie: UploadFile = File(...), idcard: UploadFile = File(...)):

    selfie_path = save_file(selfie)
    id_path = save_file(idcard)

    try:
        emb1 = get_embedding(selfie_path)
        emb2 = get_embedding(id_path)

        if emb1 is None or emb2 is None:
            return {"error": "Face not detected in one of the images"}

        # cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        matched = bool(similarity > 0.5)

        return {
            "matched": matched,
            "similarity": float(similarity)
        }

    finally:
        if os.path.exists(selfie_path):
            os.remove(selfie_path)
        if os.path.exists(id_path):
            os.remove(id_path)
