"""BullsAI — AI 사격 교정 MVP (로컬 웹앱).

    streamlit run app/streamlit_app.py
같은 와이파이의 휴대폰에서 http://<이 PC IP>:8501 로 접속해 사진을 올릴 수 있다.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shootcoach.config import load_target_spec  # noqa: E402
from shootcoach.diagnosis.calibration import ShooterProfile, apply_profile  # noqa: E402
from shootcoach.diagnosis.rules import load_causes  # noqa: E402
from shootcoach.diagnosis.stage1 import diagnose_stage1  # noqa: E402
from shootcoach.target.scoring import group_stats  # noqa: E402
from shootcoach.diagnosis.stage1 import Candidate, Stage1Result, ZeroAdjust  # noqa: E402
from shootcoach.diagnosis.stage2 import diagnose_stage2  # noqa: E402
from shootcoach.explain.vlm import template_explanation, vlm_available, vlm_explanation  # noqa: E402
from shootcoach.pipeline import analyze_posture, analyze_target, to_json  # noqa: E402
from shootcoach.pose.features import arm_series  # noqa: E402
from shootcoach.pose.render import key_frames  # noqa: E402
from shootcoach.pose.simulate import Faults, simulate_side_view  # noqa: E402
from shootcoach.target.detect import ClassicHoleDetector, get_detector  # noqa: E402
from shootcoach.target.markers import MarkerError  # noqa: E402
from shootcoach.target.template import save_printable  # noqa: E402

from app.theme import apply_theme, brand_header, hero  # noqa: E402

st.set_page_config(page_title="BullsAI · AI 사격 교정", layout="wide")
apply_theme()
SPEC = load_target_spec()


@st.cache_resource
def detector(kind: str):
    return ClassicHoleDetector() if kind == "classic" else get_detector("auto")


def stage1_from_dict(d: dict) -> Stage1Result:
    return Stage1Result(d["shape"], d["shape_ko"], d["sector"], d["sector_ko"], d["handedness"],
                        [Candidate(**c) for c in d["candidates"]],
                        ZeroAdjust(**d["zero_adjust"]) if d["zero_adjust"] else None, d["notes"])


with st.sidebar:
    brand_header()
    st.markdown("###### 분석 설정")
    hand = st.radio("주로 쓰는 손", ["right", "left"], format_func=lambda x: "오른손" if x == "right" else "왼손")
    distance = st.number_input("사격 거리 (m)", 3.0, 50.0, 15.0, 1.0)
    click = st.number_input("조준기 1클릭 값 (mm @10m, 모르면 0)", 0.0, 50.0, 0.0, 0.5)
    det_kind = st.selectbox("탄공 검출기", ["auto", "classic"],
                            format_func=lambda x: "딥러닝 (YOLO)" if x == "auto" else "전통 영상처리 (대체)")
    shooter = st.text_input("사수 프로필 (캘리브레이션, 비우면 사용 안 함)", "")
    use_vlm = st.checkbox("로컬 VLM으로 설명 문장 생성", value=False)
    if use_vlm:
        st.caption("Ollama 연결됨" if vlm_available() else "Ollama 미실행 — 템플릿 문장 사용")
    st.divider()
    st.caption("모든 분석은 이 PC 안에서만 처리됩니다. 사진·영상은 외부로 전송되지 않습니다.")

hero("한 발 더 정확하게.", "표적지의 탄착군을 읽고 자세 영상으로 원인을 좁혀 보세요. 분석부터 교정 훈련까지 한 화면에서 이어집니다.")
tab1, tab2, tab3, tab4 = st.tabs(["표적지 분석", "자세 영상", "설명 문장", "표적지 인쇄"])

with tab1:
    st.markdown('<span class="ba-step">01 / TARGET ANALYSIS</span>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        up = st.file_uploader("표적지 사진 (네 모서리 마커가 모두 보이게)", type=["jpg", "jpeg", "png"])
        demos = sorted(p.name for p in (ROOT / "samples").glob("demo_*.jpg"))
        demo = st.selectbox("또는 데모 사진", ["(선택 안 함)"] + demos)
        go = st.button("분석하기", type="primary")
    if go:
        img = None
        if up is not None:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        elif demo != "(선택 안 함)":
            img = cv2.imread(str(ROOT / "samples" / demo))
        if img is None:
            st.warning("사진을 올리거나 데모 사진을 고르세요.")
        else:
            try:
                res = analyze_target(img, SPEC, hand, distance, click or None, detector(det_kind))
                if shooter.strip():
                    prof_path = ROOT / "profiles" / f"{shooter.strip()}.json"
                    if prof_path.exists():
                        st_ = group_stats(res.holes, SPEC)
                        s1 = apply_profile(diagnose_stage1(st_, SPEC, hand, distance, click or None), st_,
                                           ShooterProfile.load(prof_path))
                        res.report["stage1"] = s1.as_dict()
                st.session_state["report"] = res.report
                st.session_state["holes"] = res.holes
                st.session_state["overlay"] = res.overlay
                st.session_state.pop("stage2_seq", None)
            except MarkerError as e:
                st.error(str(e))
    rep = st.session_state.get("report")
    if rep:
        g, s1 = rep["group"], rep["stage1"]
        with c2:
            st.image(cv2.cvtColor(st.session_state["overlay"], cv2.COLOR_BGR2RGB), caption="검출 결과 (주황: 탄공·점수, 파랑: 탄착군 중심·평균 반경, 초록: 중심 → 탄착군)")
        m = st.columns(5)
        m[0].metric("탄 수", g["n"])
        m[1].metric("총점", g["total_score"])
        m[2].metric("중심 오프셋", f"{g['offset_mm']:.0f} mm")
        m[3].metric("평균 반경", f"{g['mean_radius_mm']:.0f} mm")
        m[4].metric("처리 시간", f"{rep['timings']['total_s']*1000:.0f} ms")
        st.subheader(f"탄착군: {s1['shape_ko']} · {s1['sector_ko']}")
        for n in s1["notes"]:
            st.info(n)
        if s1["zero_adjust"]:
            st.success(s1["zero_adjust"]["text_ko"])
        for i, c in enumerate(s1["candidates"], 1):
            with st.expander(f"후보 {i}. {c['cause_ko']}  (가중치 {c['score']:.2f})", expanded=i == 1):
                st.write(c["guidance_ko"])
                st.write(f"**교정 훈련:** {c['drill_ko']}")
                st.caption(f"근거: {', '.join(c['sources'])} · 규칙 {', '.join(c['rule_ids'])}"
                           + (f" · 자세 영상: {c['observable_note']}" if c["observable_note"] else ""))
        st.download_button("리포트 JSON 받기", to_json(rep), "report.json", "application/json")
        if shooter.strip():
            with st.expander("진단 세션으로 기록 (캘리브레이션)"):
                st.caption("사수가 일부러 특정 오류를 내며 쏜 표적지라면, 그 오류가 이 사수에게서 어느 방향으로 나타나는지 기록합니다.")
                causes = load_causes()
                cid = st.selectbox("일부러 낸 오류", sorted(causes), format_func=lambda k: causes[k].cause_ko)
                if st.button("이 표적지를 기록"):
                    prof_path = ROOT / "profiles" / f"{shooter.strip()}.json"
                    prof = ShooterProfile.load(prof_path) if prof_path.exists() else ShooterProfile(shooter.strip(), hand)
                    prof.add_session(cid, group_stats(st.session_state["holes"], SPEC))
                    prof.save(prof_path)
                    st.success(f"기록했습니다 → profiles/{shooter.strip()}.json (이 PC에만 저장)")

with tab2:
    st.markdown('<span class="ba-step">02 / POSTURE CHECK</span>', unsafe_allow_html=True)
    rep = st.session_state.get("report")
    if not rep:
        st.info("먼저 표적지 분석 탭에서 분석을 실행하세요.")
    else:
        st.write("측면(사수 옆 2m, 높이 1.2m) 삼각대 촬영 영상을 올리세요. 총성이 녹음되어 있으면 격발 시점을 더 정확히 찾습니다.")
        vid = st.file_uploader("자세 영상", type=["mp4", "mov", "m4v"])
        sim = st.selectbox("또는 합성 자세 데모 (실제 영상이 없을 때)",
                           ["(선택 안 함)", "정상 자세", "격발 직전 총구 하강", "격발 직전 총구 들림 + 어깨 긴장",
                            "격발 직후 팔 내림", "조준 중 호흡 흔들림"])
        if st.button("자세 분석", type="primary"):
            if vid is not None:
                with tempfile.NamedTemporaryFile(suffix=Path(vid.name).suffix, delete=False) as f:
                    f.write(vid.read())
                with st.spinner("관절 추출 중… (이 PC에서 처리)"):
                    analyze_posture(rep, f.name, hand)
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
            st.subheader(s2["final_ko"])
            st.caption(f"촬영 방향: {'측면' if s2['view'] == 'side' else '후방/정면'} · 격발 {len(s2['shot_times'])}회 "
                       f"({s2['shot_source']})")
            st.dataframe(pd.DataFrame([{"원인 후보": v["cause_ko"], "판정": v["status_ko"], "신뢰도": v["confidence"],
                                        "근거": " / ".join(v["evidence"])} for v in s2["verdicts"]]),
                         hide_index=True, width="stretch")
            for x in s2.get("extra_findings", []):
                st.warning("추가 관찰: " + x)
            for n in s2.get("notes", []):
                st.caption(n)
            seq = st.session_state.get("stage2_seq")
            if seq is not None:
                _, pitch, _ = arm_series(seq, hand)
                st.line_chart(pd.DataFrame({"팔뚝 각도(°)": pitch}, index=np.round(seq.t, 2)), height=220)
                st.image([cv2.cvtColor(k, cv2.COLOR_BGR2RGB) for k in key_frames(seq, s2["shot_times"][0])],
                         caption=["격발 0.3초 전", "격발 직전", "격발 0.5초 후"], width=220)

with tab3:
    st.markdown('<span class="ba-step">03 / COACH NOTES</span>', unsafe_allow_html=True)
    rep = st.session_state.get("report")
    if not rep:
        st.info("먼저 표적지 분석 탭에서 분석을 실행하세요.")
    elif st.button("설명 문장 만들기", type="primary"):
        if use_vlm:
            with st.spinner("로컬 VLM 생성 중…"):
                ex = vlm_explanation(rep, [st.session_state["overlay"]])
        else:
            ex = {"text": template_explanation(rep), "source": "template", "latency_s": 0}
        st.write(ex["text"])
        st.caption(f"출처: {'로컬 VLM ' + ex.get('model', '') if ex['source'] == 'vlm' else '템플릿'} · {ex['latency_s']}초"
                   + (f" · 오류: {ex['error']}" if ex.get("error") else ""))
        st.caption("VLM은 판정하지 않고, 규칙 엔진 결과를 사람 말로 풀어주기만 합니다.")

with tab4:
    st.markdown('<span class="ba-step">04 / PRINT TARGET</span>', unsafe_allow_html=True)
    st.write("A4 연습용 표적지입니다. **배율 100%(실제 크기)**로 인쇄하세요. 실제 사격장 표적지에는 같은 마커를 스티커로 붙이고 위치를 실측해 설정 파일을 만듭니다.")
    png, pdf = save_printable(SPEC, ROOT / "samples")
    st.download_button("PDF 받기", pdf.read_bytes(), pdf.name, "application/pdf")
    st.image(str(png), width=320)
