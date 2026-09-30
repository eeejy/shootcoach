"""BullsAI — AI 사격 교정 (로컬 웹앱).

    streamlit run app/streamlit_app.py
마커 없이 표적지 사진을 분석한다. AI 검출 결과는 교관이 사진을 눌러 바로 고칠 수 있다.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shootcoach.device import model_path  # noqa: E402
from shootcoach.diagnosis.calibration import ShooterProfile as CalProfile  # noqa: E402
from shootcoach.diagnosis.calibration import apply_profile  # noqa: E402
from shootcoach.diagnosis.rules import load_causes  # noqa: E402
from shootcoach.diagnosis.stage1 import Candidate, ShooterProfile, Stage1Result, ZeroAdjust, diagnose_stage1  # noqa: E402
from shootcoach.diagnosis.stage2 import diagnose_stage2  # noqa: E402
from shootcoach.explain.vlm import template_explanation, vlm_available, vlm_explanation  # noqa: E402
from shootcoach.pipeline import analyze_posture, analyze_target, count_holes, default_spec, get_photo_detector, to_json  # noqa: E402
from shootcoach.pose.features import arm_series  # noqa: E402
from shootcoach.pose.render import key_frames  # noqa: E402
from shootcoach.pose.simulate import Faults, simulate_side_view  # noqa: E402
from shootcoach.target.detect import Hole  # noqa: E402
from shootcoach.target.locate import TargetNotFound  # noqa: E402
from shootcoach.target.scoring import group_stats  # noqa: E402

from app.theme import apply_theme, brand_header, page_icon, section  # noqa: E402

st.set_page_config(page_title="BullsAI · AI 사격 교정", page_icon=page_icon(), layout="wide")
apply_theme()
SPEC = default_spec()
FEEDBACK = ROOT / "profiles" / "instructor_feedback.jsonl"


@st.cache_resource
def detector():
    return get_photo_detector()


POSE_MODELS = {"precise": "yolo11m-pose.pt", "fast": "yolo11n-pose.pt"}   # 첫 항목이 기본. 정밀 약 77ms/프레임(관절 떨림 약 절반), 빠름 약 21ms


def stage1_from_dict(d: dict) -> Stage1Result:
    return Stage1Result(d["shape"], d["shape_ko"], d["sector"], d["sector_ko"], d["handedness"],
                        [Candidate(**c) for c in d["candidates"]],
                        ZeroAdjust(**d["zero_adjust"]) if d["zero_adjust"] else None, d["notes"], d.get("components", {}))


with st.sidebar:
    brand_header()
    section("SHOOTER")
    hand = st.radio("주로 쓰는 손", ["right", "left"], format_func=lambda x: "오른손" if x == "right" else "왼손", horizontal=True)
    hand_size = st.select_slider("손 크기", ["small", "medium", "large"], "medium",
                                 format_func=lambda x: {"small": "작음", "medium": "보통", "large": "큼"}[x])
    finger = st.select_slider("손가락 길이", ["short", "normal", "long"], "normal",
                              format_func=lambda x: {"short": "짧음", "normal": "보통", "long": "김"}[x])
    fatigue = st.toggle("피로함", False)
    section("RANGE")
    c1, c2 = st.columns(2)
    distance = c1.number_input("거리 (m)", 3.0, 50.0, 10.0, 1.0)
    shots = c2.number_input("발수", 1, 30, 10, 1)
    section("PRE-CHECK")
    zero_ok = st.radio("총기 영점", ["unknown", "yes", "no"], horizontal=True,
                       format_func=lambda x: {"unknown": "모름", "yes": "확인됨", "no": "안 됨"}[x])
    grip_fit = st.radio("그립 크기", ["ok", "too_large", "too_small"], horizontal=True,
                        format_func=lambda x: {"ok": "맞음", "too_large": "손에 큼", "too_small": "손에 작음"}[x])
    section("ENGINE")
    pose_kind = st.selectbox("자세 분석 모델", list(POSE_MODELS),
                             format_func=lambda x: {"fast": "빠름", "precise": "정밀 (약 3배 느림)"}[x])
    shooter = st.text_input("사수 이름 (기록용, 선택)", "")
    use_vlm = st.checkbox("AI 설명 문장", value=False)
    if use_vlm:
        st.caption("연결됨" if vlm_available() else "Ollama 꺼짐 — 기본 문장 사용")

PROFILE = ShooterProfile(hand_size, finger, "high" if fatigue else "low", zero_ok, grip_fit)


def run_analysis(img: np.ndarray, holes: list[Hole] | None = None) -> None:
    res = analyze_target(img, SPEC, hand, distance, None, detector(), expected_shots=int(shots), holes_override=holes)
    st_ = group_stats(res.holes, SPEC)
    s1 = diagnose_stage1(st_, SPEC, hand, distance, profile=PROFILE)
    if shooter.strip():
        prof_path = ROOT / "profiles" / f"{shooter.strip()}.json"
        if prof_path.exists():
            s1 = apply_profile(s1, st_, CalProfile.load(prof_path))
    res.report["stage1"] = s1.as_dict()
    st.session_state.update(report=res.report, holes=res.holes, overlay=res.overlay, image=img)
    st.session_state.pop("stage2_seq", None)


tab1, tab2, tab3 = st.tabs(["표적지 분석", "자세 영상", "설명 문장"])

with tab1:
    demos = sorted(p.name for p in (ROOT / "samples").glob("demo_*.jpg"))
    i1, i2, i3 = st.columns([5, 4, 1.4], vertical_alignment="bottom")
    up = i1.file_uploader("표적지 사진", type=["jpg", "jpeg", "png"])
    demo = i2.selectbox("또는 데모 사진", ["(선택 안 함)"] + demos)
    if i3.button("분석", type="primary", width="stretch"):
        img = None
        if up is not None:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        elif demo != "(선택 안 함)":
            img = cv2.imread(str(ROOT / "samples" / demo))
        if img is None:
            st.warning("사진을 올리거나 데모 사진을 고르세요.")
        else:
            try:
                run_analysis(img)
                st.session_state["click"] = None
                st.session_state.pop("count_only", None)
            except TargetNotFound as e:
                st.session_state.pop("report", None)
                try:
                    st.session_state["count_only"] = count_holes(img, detector(), int(shots))
                except TargetNotFound:
                    st.error(str(e))
    rep = st.session_state.get("report")
    co = st.session_state.get("count_only")
    if co and not rep:
        st.warning("원형 표적을 찾지 못해 **탄공 개수만** 셉니다. 속사(하반신) 표적의 영역 채점은 규정을 받은 뒤 지원합니다.")
        l, r = st.columns([5, 6], gap="large")
        l.image(cv2.cvtColor(co["overlay"], cv2.COLOR_BGR2RGB), width="stretch")
        r.metric(f"탄 수 (기준 {co['expected'] or '-'})", co["n"])
    elif not rep:
        st.caption("표적지 사진을 올리거나 데모 사진을 고른 뒤 분석을 누르세요. 마커는 필요 없습니다.")
    else:
        g, s1 = rep["group"], rep["stage1"]
        left, right = st.columns([5, 6], gap="large")
        with left:
            view = st.session_state["overlay"]
            send_w = min(900, view.shape[1])
            small = cv2.resize(view, (send_w, int(view.shape[0] * send_w / view.shape[1])))
            click = streamlit_image_coordinates(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), width="stretch", key="target_view")
            if click and click != st.session_state.get("click"):
                st.session_state["click"] = click
                k = view.shape[1] / (click.get("width") or send_w) / SPEC.px_per_mm      # 표시 px → 종이 mm
                x_mm, y_mm = click["x"] * k, click["y"] * k
                holes = list(st.session_state["holes"])
                near = [i for i, h in enumerate(holes) if np.hypot(h.x_mm - x_mm, h.y_mm - y_mm) <= max(h.r_mm * 1.4, 6)]
                if near:
                    holes.pop(near[0])                                         # 누른 곳에 탄공 → 삭제
                else:
                    holes.append(Hole(x_mm, y_mm, SPEC.bullet_diameter_mm / 2, conf=1.0))   # 빈 곳 → 추가
                run_analysis(st.session_state["image"], holes)
                st.rerun()
            st.caption("사진을 누르면 고칠 수 있습니다 · 빈 곳 = 탄공 추가 · 탄공 = 삭제")
            b1, b2 = st.columns(2)
            if b1.button("AI 결과로 되돌리기", width="stretch"):
                run_analysis(st.session_state["image"])
                st.rerun()
            b2.download_button("리포트 JSON", to_json(rep), "report.json", "application/json", width="stretch")
        with right:
            m = st.columns(3)
            m[0].metric(f"탄 수 (기준 {rep.get('expected_shots') or '-'})", g["n"])
            m[1].metric("총점", g["total_score"])
            m[2].metric("반경 mm", f"{g['mean_radius_mm']:.0f}")
            st.subheader(f"{s1['shape_ko']} · {s1['sector_ko']}")
            for n in rep.get("notes", []) + s1["notes"]:
                st.info(n)
            if s1["zero_adjust"]:
                st.success(s1["zero_adjust"]["text_ko"])
            for i, c in enumerate(s1["candidates"], 1):
                with st.expander(f"{i}. {c['cause_ko']}  ·  {c['score']:.2f}", expanded=i == 1):
                    st.write(c["guidance_ko"])
                    if c.get("checklist"):
                        st.markdown("**현장 확인**")
                        for j, item in enumerate(c["checklist"]):
                            st.checkbox(item, key=f"chk_{c['cause_id']}_{j}")
                    if c.get("drill_ko"):
                        st.write(f"**교정 훈련:** {c['drill_ko']}")
            if s1["candidates"]:
                with st.expander("교관 판정 기록"):
                    opts = [c["cause_id"] for c in s1["candidates"]] + ["other"]
                    names = {c["cause_id"]: c["cause_ko"] for c in s1["candidates"]} | {"other": "목록에 없음"}
                    verdict = st.radio("실제 원인", opts, format_func=lambda k: names[k])
                    memo = st.text_input("메모", "")
                    if st.button("판정 저장"):
                        FEEDBACK.parent.mkdir(exist_ok=True)
                        with open(FEEDBACK, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), "shooter": shooter,
                                                "ai_top": s1["candidates"][0]["cause_id"],
                                                "ai_top3": [c["cause_id"] for c in s1["candidates"][:3]],
                                                "instructor": verdict, "memo": memo, "shape": s1["shape"],
                                                "center_mm": g["center_mm"], "n": g["n"]}, ensure_ascii=False) + "\n")
                        st.success("저장했습니다 (이 PC에만). AI 진단 정확도 집계에 쓰입니다.")
                    if shooter.strip():
                        causes = load_causes()
                        cid = st.selectbox("진단 세션 기록: 일부러 낸 오류", sorted(causes), format_func=lambda k: causes[k].cause_ko)
                        if st.button("이 표적지를 진단 세션으로 기록"):
                            prof_path = ROOT / "profiles" / f"{shooter.strip()}.json"
                            prof = CalProfile.load(prof_path) if prof_path.exists() else CalProfile(shooter.strip(), hand)
                            prof.add_session(cid, group_stats(st.session_state["holes"], SPEC))
                            prof.save(prof_path)
                            st.success(f"기록했습니다 → profiles/{shooter.strip()}.json")
            st.caption(f"처리 {rep['timings']['total_s'] * 1000:.0f} ms · {rep['detector']}")

with tab2:
    rep = st.session_state.get("report")
    if not rep:
        st.info("먼저 표적지 분석 탭에서 분석을 실행하세요.")
    else:
        i1, i2, i3 = st.columns([5, 4, 1.4], vertical_alignment="bottom")
        vid = i1.file_uploader("자세 영상", type=["mp4", "mov", "m4v"])
        sim = i2.selectbox("또는 데모 영상",
                           ["(선택 안 함)", "정상 자세", "격발 직전 총구 하강", "격발 직전 총구 들림 + 어깨 긴장",
                            "격발 직후 팔 내림", "조준 중 호흡 흔들림"])
        if i3.button("분석", type="primary", width="stretch", key="pose_go"):
            if vid is not None:
                with tempfile.NamedTemporaryFile(suffix=Path(vid.name).suffix, delete=False) as f:
                    f.write(vid.read())
                with st.spinner("관절 추출 중…"):
                    analyze_posture(rep, f.name, hand, pose_model=model_path(POSE_MODELS[pose_kind]))
                st.session_state["stage2_video"] = f.name
            elif sim != "(선택 안 함)":
                faults = {"정상 자세": Faults(), "격발 직전 총구 하강": Faults(dip_deg=4),
                          "격발 직전 총구 들림 + 어깨 긴장": Faults(heel_deg=4, shrug=0.02),
                          "격발 직후 팔 내림": Faults(early_drop=0.12), "조준 중 호흡 흔들림": Faults(breath_amp=0.03)}[sim]
                seq = simulate_side_view((2.5, 5.0, 7.5), faults=faults)
                s2 = diagnose_stage2(stage1_from_dict(rep["stage1"]), seq, None, hand)
                rep["stage2"] = s2.as_dict() | {"shot_source": "motion (synthetic)"}
                st.session_state["stage2_seq"] = seq
        s2 = rep.get("stage2")
        if s2:
            seq = st.session_state.get("stage2_seq")
            left, right = st.columns([6, 5], gap="large")
            with left:
                st.subheader(s2["final_ko"])
                st.caption(f"{'측면' if s2['view'] == 'side' else '후방/정면'} 촬영 · 격발 {len(s2['shot_times'])}회")
                st.dataframe(pd.DataFrame([{"원인 후보": v["cause_ko"], "판정": v["status_ko"], "신뢰도": v["confidence"],
                                            "근거": " / ".join(v["evidence"])} for v in s2["verdicts"]]),
                             hide_index=True, width="stretch")
                for x in s2.get("extra_findings", []):
                    st.warning("추가 관찰: " + x)
                if seq is not None:
                    _, pitch, _ = arm_series(seq, hand)
                    st.line_chart(pd.DataFrame({"팔뚝 각도(°)": pitch}, index=np.round(seq.t, 2)), height=200, color="#5fc2ec")
            with right:
                if seq is not None and s2["shot_times"]:
                    frames = key_frames(seq, s2["shot_times"][0])
                    cols = st.columns(3)
                    for col, im, cap in zip(cols, frames, ["0.3초 전", "격발 직전", "0.5초 후"]):
                        col.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), caption=cap, width="stretch")
                for n in s2.get("notes", []):
                    st.caption(n)

with tab3:
    rep = st.session_state.get("report")
    if not rep:
        st.info("먼저 표적지 분석 탭에서 분석을 실행하세요.")
    elif st.button("설명 문장 만들기", type="primary"):
        if use_vlm:
            with st.spinner("로컬 AI 생성 중…"):
                ex = vlm_explanation(rep, [st.session_state["overlay"]])
        else:
            ex = {"text": template_explanation(rep), "source": "template", "latency_s": 0}
        st.write(ex["text"])
        st.caption(f"출처: {'로컬 AI ' + ex.get('model', '') if ex['source'] == 'vlm' else '기본 문장'} · {ex['latency_s']}초"
                   + (f" · 오류: {ex['error']}" if ex.get("error") else ""))
