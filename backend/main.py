import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI(title="TOD Play MVP API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def pil_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def canny_edges(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 1.2)
    edges = cv2.Canny(gray, 80, 160)
    return edges

def simple_overlay(ref_cv, draw_cv):
    h, w = ref_cv.shape[:2]
    draw_resized = cv2.resize(draw_cv, (w, h), interpolation=cv2.INTER_AREA)
    edges_ref = canny_edges(ref_cv)
    edges_draw = canny_edges(draw_resized)
    inv = 255 - edges_draw
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    mask = (edges_ref > 0).astype(np.float32)
    heat = (dist_norm * mask)
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(ref_cv, 0.7, heat_color, 0.6, 0.0)
    return overlay

@app.post("/analyze")
async def analyze(
    ref: Optional[UploadFile] = File(default=None),
    draw: UploadFile = File(...),
    checks: str = Form("shape,value,perspective,composition"),
    locale: str = Form("ko"),
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ref_path = None
    if ref is not None:
        ref_bytes = await ref.read()
        ref_path = os.path.join(UPLOAD_DIR, f"ref_{ts}.png")
        open(ref_path, "wb").write(ref_bytes)

    draw_bytes = await draw.read()
    draw_path = os.path.join(UPLOAD_DIR, f"draw_{ts}.png")
    open(draw_path, "wb").write(draw_bytes)

    draw_img = Image.open(draw_path).convert("RGB")

    if ref_path:
        ref_img = Image.open(ref_path).convert("RGB")
        overlay_cv = simple_overlay(pil_to_cv(ref_img), pil_to_cv(draw_img))
        mode = "compare"
    else:
        dcv = pil_to_cv(draw_img)
        edges = canny_edges(dcv)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        overlay_cv = cv2.addWeighted(dcv, 0.7, edges_color, 0.6, 0.0)
        mode = "single"

    overlay_path = os.path.join(RESULT_DIR, f"overlay_{ts}.jpg")
    cv2.imwrite(overlay_path, overlay_cv)

    scores = {"shape": 78, "value": 72, "color": 70, "perspective": 75, "composition": 74}
    tips = [
        "형태: 큰 덩어리 각도부터 잡고 세부를 얹으세요.",
        "명암: 광원을 고정하고 경계의 날카로움/부드러움을 구분하세요.",
        "색감: 채도 대비로 초점 영역을 분리하세요.",
    ]

    return {
        "ok": True,
        "overlay_url": f"/result/{os.path.basename(overlay_path)}",
        "scores": scores,
        "tips": tips,
        "mode": mode,
    }

@app.get("/result/{name}")
def get_result(name: str):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    return FileResponse(path)
