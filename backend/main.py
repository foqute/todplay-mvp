# backend/main.py
import os, base64, json, traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# ======================
# 초기 로드 & 클라이언트
# ======================
load_dotenv()
client = OpenAI()

app = FastAPI(title="TOD play - GPT5 wired", version="0.1.0")

app.add_middleware(
    CORSMiddleware(
        allow_origins=["*"],     # 개발 단계 전체 허용
        allow_methods=["*"],
        allow_headers=["*"],
    )
)

# ===== In-memory context store =====
CTX: Dict[str, Dict[str, Any]] = {}

# =====================
# 기본 유틸 함수
# =====================
def to_data_url(file: UploadFile) -> str:
    raw = file.file.read()
    file.file.seek(0)
    mime = file.content_type or "image/jpeg"
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def pick_model(user_mode: str) -> str:
    # 필요하면 모드별로 분기
    return "gpt-5"

# =====================
# 프롬프트
# =====================
SYSTEM_PROMPT = """당신은 학생에게 친절하고 격려를 아끼지 않는 미술 과외 선생님입니다.
- 출력은 한국어.
- 마크다운 섹션과 번호 목록을 적극 사용.
- '빠른 요약' → '형태/비율' → '구도/원근' → '인체/손발' → '선/밸류/에지' → '채색/음영'(있을 때) →
  '오버레이 관찰 포인트' → '30초 드릴 3개' → '우선순위 Top 3' → '응원의 멘트' 순서.
- 각 항목은 1~2문장으로 핵심만. 말투는 부드럽고 동기부여가 되게.
"""

CHAT_SYSTEM = """당신은 한국어로 응답하는 그림 학습 조교입니다.
- 짧고 명확하게 답하고, 필요하면 단계별 팁을 1~3개 불릿으로 제시하세요.
- 작품 분석/연습법/도구 활용/해부학/원근/채색 등 미술 전반 질문에 답합니다.
- 세션 컨텍스트의 원작/실습 이미지를 우선 참고해 구체적으로 설명하세요.
"""

# =====================
# 유틸: metrics 파싱/라벨
# =====================
_METRIC_LABELS = {
    "shape": "형태",
    "value": "명암",
    "color": "색감",
    "edge":  "투시",
    "persp": "투시",
    "structure": "구도",
    "comp": "구도",
    # 필요시 쓸 수 있는 디테일/완성도 코드
    "detail": "디테일/완성도",
}
def parse_metrics(metrics: Optional[str]) -> List[str]:
    if not metrics:
        return []
    try:
        arr = json.loads(metrics)
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        pass
    return []

def ko_metrics_list(metrics_codes: List[str]) -> str:
    if not metrics_codes:
        return "형태 중심"
    names = [_METRIC_LABELS.get(k, k) for k in metrics_codes]
    return ", ".join(names)

# =====================
# 헬스 체크
# =====================
@app.get("/ping")
def ping():
    return {"ok": True}

# ==========================
# /analyze — 피드백 분석 (TOD 플레이)
# ==========================
@app.post("/analyze")
async def analyze(
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    mode: str = Form("AUTO"),
    metrics: Optional[str] = Form(None),
    learn: str = Form("모작"),
    hints: Optional[str] = Form(None),
    context_id: Optional[str] = Form(None),
    client_ts: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
):
    try:
        print("[ANALYZE] 요청 수신", flush=True)
        print(f"[ANALYZE] mode={mode}, learn={learn}, metrics={metrics}, context_id={context_id}", flush=True)

        model = pick_model(mode)
        left_url  = to_data_url(left)
        right_url = to_data_url(right)
        print(f"[ANALYZE] left_url_len={len(left_url)}, right_url_len={len(right_url)}", flush=True)

        metric_codes = parse_metrics(metrics)
        metric_ko = ko_metrics_list(metric_codes)
        print(f"[ANALYZE] metric_ko={metric_ko}", flush=True)

        mode_policy = (
            "FAST 모드: '형태' 또는 '형태+명암'만 간단 평가.\n"
            "FULL 모드: 형태/명암/색감/투시/구도를 모두 평가.\n"
            "체크된 항목만 다루되, 핵심만 간결히."
        )

        if learn == "창작":
            base_text = (
                "오른쪽 이미지는 '실습'입니다. 이 이미지만 단독으로 평가하세요. "
                f"평가 항목: {metric_ko}. " + mode_policy
            )
            content_images = [{"type": "input_image", "image_url": right_url}]
        elif learn == "스타일 모작":
            base_text = (
                "왼쪽은 레퍼런스(스타일), 오른쪽은 '실습'입니다. "
                f"평가 항목: {metric_ko}. " + mode_policy
            )
            content_images = [
                {"type": "input_image", "image_url": left_url},
                {"type": "input_image", "image_url": right_url},
            ]
        else:
            base_text = (
                "왼쪽은 '원본', 오른쪽은 '실습'입니다. "
                f"평가 항목: {metric_ko}. " + mode_policy
            )
            content_images = [
                {"type": "input_image", "image_url": left_url},
                {"type": "input_image", "image_url": right_url},
            ]

        # (여기서는 디테일 지침은 추가하지 않음 — 오늘의 미션 쪽에서 점수로 반영)

        if question:
            base_text += "\n\n학생 질문: " + question.strip()
        if hints:
            try:
                base_text += "\n\n추가 지침:\n" + json.dumps(
                    json.loads(hints), ensure_ascii=False, indent=2
                )
            except Exception:
                base_text += "\n\n추가 지침:\n" + str(hints)

        print(f"[ANALYZE] OpenAI 호출 시작 - model={model}", flush=True)
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": base_text}, *content_images]},
            ],
        )
        print("[ANALYZE] OpenAI 응답 수신", flush=True)

        notes_text = resp.output_text
        print(f"[ANALYZE] 응답 텍스트 길이={len(notes_text)}", flush=True)

        feedback_plain = "\n".join(
            line.replace("#", "").strip() for line in notes_text.splitlines()
        ).strip()
        cid = context_id or "default"
        CTX[cid] = {
            "left_image": left_url,
            "right_image": right_url,
            "feedback_text": feedback_plain,
            "learn": learn,
            "metric_codes": metric_codes,
            "last_mode": mode,
        }

        print(f"[ANALYZE] 컨텍스트 저장 완료 cid={cid}", flush=True)
        return {"ok": True, "notes": notes_text, "context_id": cid}
    except Exception as e:
        print("[ANALYZE][ERROR]", repr(e), flush=True)
        traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# ==========================
# /chat — 문맥 Q/A
# ==========================
class ChatIn(BaseModel):
    message: str
    mode: Optional[str] = "AUTO"
    context_id: Optional[str] = None
    feedback_text: Optional[str] = ""
    learn: Optional[str] = ""
    metrics_map: Optional[Dict[str, Any]] = None
    left_image: Optional[str] = None
    right_image: Optional[str] = None
    client_ts: Optional[int] = None

@app.post("/chat")
async def chat(inp: ChatIn):
    try:
        msg = (inp.message or "").strip()
        if not msg:
            return {"ok": False, "reply": "질문을 입력해 주세요."}
        cid = inp.context_id or "default"
        cached = CTX.get(cid, {})
        left  = inp.left_image  or cached.get("left_image")
        right = inp.right_image or cached.get("right_image")
        fb    = (inp.feedback_text or "").strip() or cached.get("feedback_text", "")
        learn = inp.learn or cached.get("learn", "")
        metric_codes = cached.get("metric_codes", [])
        if not (left and right):
            return {"ok": True, "reply": "원작/실습 이미지가 없습니다. 먼저 분석을 수행해 주세요."}

        user_blocks = []
        ctx_summary = [
            f"[세션] {cid}",
            f"[학습유형] {learn or '모작'}",
            f"[평가항목] {ko_metrics_list(metric_codes)}",
        ]
        if fb:
            ctx_summary.append(f"[피드백]\n{fb[:600]}{'...' if len(fb) > 600 else ''}")
        user_blocks.append({"type": "input_text", "text": "\n".join(ctx_summary)})
        user_blocks.append({"type": "input_image", "image_url": left})
        user_blocks.append({"type": "input_image", "image_url": right})
        user_blocks.append({"type": "input_text", "text": msg})

        model = pick_model(inp.mode)
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": CHAT_SYSTEM}]},
                {"role": "user", "content": user_blocks},
            ],
        )
        reply = resp.output_text
        CTX.setdefault(cid, {})["last_chat"] = msg
        return {"ok": True, "reply": reply}
    except Exception as e:
        return {"ok": False, "reply": "오류 발생", "error": str(e)}

# ==========================
# 🟢 오늘의 드로잉 미션 관리
# ==========================
ROOT = Path(__file__).resolve().parents[1]  # backend/.. = 프로젝트 루트
MISSION_DIR = ROOT / "missions"
MISSION_DIR.mkdir(parents=True, exist_ok=True)
MISSION_FILE = MISSION_DIR / "daily_missions.json"

# 정적 파일 제공(/missions/daily_missions.json)
app.mount("/missions", StaticFiles(directory=str(MISSION_DIR)), name="missions")

class MissionUpsert(BaseModel):
    date: str         # "YYYY-MM-DD"
    title: str = ""   # 옵션
    images: list      # dataURL 배열

def _read_json() -> Dict[str, Any]:
    if not MISSION_FILE.exists():
        return {}
    try:
        return json.loads(MISSION_FILE.read_text("utf-8"))
    except Exception:
        return {}

def _write_json(data: Dict[str, Any]):
    MISSION_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
    )

# ---- 새(권장) 경로: /api/missions/daily_missions ----
@app.post("/api/missions/daily_missions")
@app.put("/api/missions/daily_missions")
def upsert_daily_mission_api(payload: MissionUpsert = Body(...)):
    if not payload.images or not isinstance(payload.images, list):
        raise HTTPException(status_code=400, detail="images must be a list")
    data = _read_json()
    data[payload.date] = {"title": payload.title, "images": payload.images}
    _write_json(data)
    return {"ok": True, "saved": payload.date}

# ---- 레거시 경로: /missions/daily_missions (POST/PUT 둘 다 허용) ----
@app.post("/missions/daily_missions")
@app.put("/missions/daily_missions")
def upsert_daily_mission_legacy(payload: MissionUpsert = Body(...)):
    return upsert_daily_mission_api(payload)

# ---- 조회 ----
@app.get("/missions/daily_missions.json")
def read_missions():
    return _read_json()

# ==========================
# 🟢 유사도 평가 (FAST) — 오늘의 드로잉 미션용
# ==========================
from io import BytesIO
import re
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2


def _decode_data_url(data_url: str) -> Image.Image:
    # data:[mime];base64,....
    m = re.match(r"^data:.*?;base64,(.+)$", data_url)
    if not m:
        raise HTTPException(400, "Invalid dataURL")
    raw = base64.b64decode(m.group(1))
    return Image.open(BytesIO(raw)).convert("RGB")


def _to_gray_np(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)


def _phash(im: Image.Image) -> int:
    # 간단 pHash (전체 실루엣/톤)
    im_small = im.resize((32, 32), Image.LANCZOS).convert("L")
    a = np.array(im_small, dtype=np.float32)
    dct = cv2.dct(a)
    dctlow = dct[:8, :8]
    med = np.median(dctlow)
    bits = (dctlow > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return int(val)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _orb_match_score(imgA_g: np.ndarray, imgB_g: np.ndarray) -> int:
    # 엣지/디테일 위주의 특징 매칭
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgA_g, None)
    kp2, des2 = orb.detectAndCompute(imgB_g, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    # 상위 50개 매치 수/품질을 점수화
    top = matches[:50]
    score = sum(max(0, 256 - m.distance) for m in top) // max(1, len(top))
    return int(score)


class SimilarityIn(BaseModel):
    ref: str   # dataURL (원작/좌측)
    test: str  # dataURL (실습/우측)


@app.post("/similarity/compare")
def similarity_compare(inp: SimilarityIn):
    try:
        # 1) dataURL → PIL 이미지
        A = _decode_data_url(inp.ref)
        B = _decode_data_url(inp.test)

        # 2) 크기 보정(SSIM 비교를 위해 동일 크기)
        W = min(A.width, B.width)
        H = min(A.height, B.height)
        A2 = A.resize((W, H), Image.LANCZOS)
        B2 = B.resize((W, H), Image.LANCZOS)

        # 3) 그레이스케일
        gA = _to_gray_np(A2)
        gB = _to_gray_np(B2)

        #    3-1) 기본 SSIM (형태/톤 중심)
        ssim_val = float(ssim(gA, gB, data_range=255))

        #    3-2) 엣지 SSIM (윤곽선/디테일 중심)
        edgesA = cv2.Canny(gA, 50, 150)
        edgesB = cv2.Canny(gB, 50, 150)
        edge_ssim = float(ssim(edgesA, edgesB, data_range=255))

        #    3-3) pHash / ORB
        ph_a = _phash(A2)
        ph_b = _phash(B2)
        ph_dist = _hamming(ph_a, ph_b)

        orb_score = _orb_match_score(gA, gB)

        # 4) 0~100 점수 계산
        #    형태(SSIM) + 디테일(edge SSIM + ORB) 비중을 높게 설정
        sim_raw = (
            ssim_val * 60.0                          # 전체 형태/톤
            + edge_ssim * 25.0                       # 윤곽선·디테일 유사도
            + max(0.0, 1.0 - ph_dist / 64.0) * 8.0   # 전체 실루엣
            + min(1.0, orb_score / 200.0) * 7.0      # 세부 특징 매칭
        )

        # 5) 0~1 사이로 정규화 후 감마 보정
        sim01 = max(0.0, min(1.0, sim_raw / 100.0))
        sim_boosted = sim01 ** 0.7  # 0.8→0.86, 0.9→0.93, 1.0→1.0
        sim_pct = int(round(sim_boosted * 100))

        return {
            "ok": True,
            "ssim": round(ssim_val, 4),
            "edge_ssim": round(edge_ssim, 4),
            "phash_distance": ph_dist,
            "orb_score": orb_score,
            "similarity_pct": sim_pct,
        }
    except Exception as e:
        raise HTTPException(400, f"similarity error: {e}")
