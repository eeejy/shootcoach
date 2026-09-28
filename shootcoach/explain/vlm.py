"""사수용 설명 문장: 로컬 VLM(Ollama · Qwen2.5-VL) → 실패 시 템플릿 문장.

VLM은 판정을 하지 않는다. 규칙 엔진이 낸 결과(수치·원인)를 사람 말로 풀어주기만 한다.
"""
from __future__ import annotations

import base64
import json
import time

import cv2
import numpy as np
import requests

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5vl:3b"
PREFERRED_MODELS = ("qwen3:8b", "qwen2.5vl:3b")   # first one installed wins

SYSTEM_KO = (
    "당신은 해양경찰 사격 교관을 돕는 보조자입니다. 아래 '분석 결과'만 근거로 사수에게 한국어로 설명하세요. "
    "분석 결과에 없는 원인을 새로 만들지 마세요. 3문장 이내로 쓰고, 마지막에 오늘 할 교정 훈련 1가지를 제시하세요."
)


def _facts(report: dict) -> str:
    s1 = report.get("stage1", {})
    g = report.get("group", {})
    lines = [
        f"탄 수 {g.get('n')}발, 총점 {g.get('total_score')}점, 탄착군 모양: {s1.get('shape_ko')}, 쏠림 방향: {s1.get('sector_ko')}",
        f"탄착군 중심 오프셋 {g.get('offset_mm')}mm, 평균 반경 {g.get('mean_radius_mm')}mm",
    ]
    cands = s1.get("candidates", [])
    if cands:
        lines.append("표적지 기준 원인 후보: " + ", ".join(c["cause_ko"] for c in cands[:3]))
    s2 = report.get("stage2")
    if s2:
        lines.append(f"자세 영상 판정: {s2.get('final_ko')}")
        for v in s2.get("verdicts", [])[:3]:
            lines.append(f"- {v['cause_ko']}: {v['status_ko']} ({'; '.join(v['evidence'])})")
    if s1.get("zero_adjust"):
        lines.append("조준기 조정: " + s1["zero_adjust"]["text_ko"])
    return "\n".join(lines)


def template_explanation(report: dict) -> str:
    s1 = report.get("stage1", {})
    s2 = report.get("stage2")
    cands = s1.get("candidates", [])
    if not cands:
        return " ".join(s1.get("notes", [])) or "탄 수가 부족해 판단을 보류합니다."
    top = cands[0]
    if s2 and s2.get("final_cause_id"):
        top = next((c for c in cands if c["cause_id"] == s2["final_cause_id"]), top)
        head = f"자세 영상에서 '{top['cause_ko']}' 신호가 확인되었습니다."
    elif s2:
        head = f"탄착군은 {s1.get('sector_ko')} 방향이지만 자세 영상에서 뚜렷한 원인 신호는 보이지 않아 판단을 보류합니다."
    else:
        head = f"탄착군이 {s1.get('shape_ko')} 형태로, '{top['cause_ko']}' 가능성이 가장 높습니다."
    return f"{head} {top['guidance_ko']} 오늘의 훈련: {top['drill_ko']}."


def _b64(img: np.ndarray, max_side: int = 448) -> str:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(enc.tobytes()).decode()


def installed_models(url: str = OLLAMA_URL) -> list[str]:
    try:
        return [m.get("name", "") for m in requests.get(f"{url}/api/tags", timeout=2).json().get("models", [])]
    except requests.RequestException:
        return []


def pick_model(url: str = OLLAMA_URL) -> str | None:
    names = installed_models(url)
    return next((m for m in PREFERRED_MODELS if any(n.startswith(m) for n in names)), None)


def vlm_available(url: str = OLLAMA_URL, model: str | None = None) -> bool:
    names = installed_models(url)
    return any(n.startswith(model) for n in names) if model else bool(pick_model(url))


def _primary(report: dict) -> dict | None:
    s1 = report.get("stage1", {})
    cands = s1.get("candidates", [])
    if not cands:
        return None
    s2 = report.get("stage2") or {}
    if s2.get("final_cause_id"):
        return next((c for c in cands if c["cause_id"] == s2["final_cause_id"]), cands[0])
    return cands[0]


LEAK_MARKERS = ("분석 결과:", "사수에게 할 설명", "3문장 이내", "새로 만들지 마세요", "가장 유력한 원인(반드시")


def mentions_cause(text: str, cause_ko: str) -> bool:
    """True if the text names the cause: full name, its alias in parentheses, or most of its key words."""
    main = cause_ko.split(" (")[0].strip()
    alias = cause_ko[cause_ko.find("(") + 1:cause_ko.find(")")].strip() if "(" in cause_ko else ""
    if main in text or (alias and alias in text):
        return True
    words = [w for w in main.replace("·", " ").split() if len(w) >= 2]
    stems = [w[:2] for w in words]                      # Korean inflection: compare 2-char stems
    hit = sum(1 for st in stems if st in text)
    return bool(words) and hit / len(words) >= 0.6


def validate_explanation(text: str, report: dict) -> str | None:
    """Reject outputs that leak the prompt or ignore the rule engine's primary cause."""
    if not text or len(text) < 20:
        return "too short"
    if any(m in text for m in LEAK_MARKERS):
        return "prompt leak"
    p = _primary(report)
    if p and not mentions_cause(text, p["cause_ko"]):
        return f"primary cause '{p['cause_ko']}' not mentioned"
    return None


def vlm_explanation(report: dict, images: list[np.ndarray] | None = None, model: str | None = None,
                    url: str = OLLAMA_URL, timeout: float = 90.0, max_side: int = 448) -> dict:
    """Returns {"text", "source": "vlm"|"template", "latency_s", "error"?}.

    The model only phrases the rule engine's result. Output is validated and replaced by the
    template sentence if it leaks the prompt or drops the primary cause.
    """
    model = model or pick_model(url) or DEFAULT_MODEL
    if "vl" not in model:
        images = None                                   # text-only model
    p = _primary(report)
    focus = f"가장 유력한 원인(반드시 이 원인을 중심으로 설명): {p['cause_ko']}\n교정 방법: {p['guidance_ko']}\n훈련: {p['drill_ko']}" if p else ""
    user = f"분석 결과:\n{_facts(report)}\n{focus}\n\n위 내용을 사수에게 2~3문장으로 설명하고, 마지막 문장은 '오늘의 훈련:'으로 시작하세요."
    t0 = time.perf_counter()
    try:
        msg = {"role": "user", "content": user}
        if images:
            msg["images"] = [_b64(im, max_side) for im in images[:3]]
        body = {"model": model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": SYSTEM_KO}, msg],
                "options": {"temperature": 0.1, "num_predict": 200}}
        r = requests.post(f"{url}/api/chat", data=json.dumps(body), timeout=timeout)
        if r.status_code == 400 and "think" in r.text:      # models without a thinking switch
            body.pop("think")
            r = requests.post(f"{url}/api/chat", data=json.dumps(body), timeout=timeout)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "").strip()
        problem = validate_explanation(text, report)
        if problem:
            raise ValueError(f"rejected: {problem} :: {text[:80]}")
        return {"text": text, "source": "vlm", "model": model, "latency_s": round(time.perf_counter() - t0, 2)}
    except Exception as e:  # noqa: BLE001 — any failure falls back to the template
        return {"text": template_explanation(report), "source": "template", "model": model,
                "latency_s": round(time.perf_counter() - t0, 2), "error": str(e)[:200]}
