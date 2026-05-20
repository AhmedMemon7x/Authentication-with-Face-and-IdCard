from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
import shutil
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_file(upload_file: UploadFile, filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return path


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/verify-id")
async def verify_id(
    selfie: UploadFile = File(...),
    idcard: UploadFile = File(...)
):

    # unique filenames (important for Render concurrency)
    uid = str(uuid.uuid4())

    selfie_path = save_file(selfie, f"{uid}_selfie.jpg")
    id_path = save_file(idcard, f"{uid}_id.jpg")

    try:
        result = DeepFace.verify(
            img1_path=selfie_path,
            img2_path=id_path,
            model_name="ArcFace",
            enforce_detection=False
        )

        return {
            "matched": result["verified"],
            "distance": float(result["distance"]),
            "threshold": float(result["threshold"])
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        # cleanup files (important for Render storage limit)
        if os.path.exists(selfie_path):
            os.remove(selfie_path)
        if os.path.exists(id_path):
            os.remove(id_path)