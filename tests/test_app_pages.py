"""Both Streamlit pages must load and the main flows must run without exceptions."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    at = AppTest.from_file(str(ROOT / path), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_labeling_page_loads():
    _load("app/pages/1_라벨링.py")


def test_main_page_target_and_posture_flow():
    at = _load("app/streamlit_app.py")
    [s for s in at.selectbox if s.label == "또는 데모 사진"][0].set_value("demo_jerking_low_left.jpg")
    [b for b in at.button if b.label == "분석"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("7시 반" in s.value for s in at.subheader)
    [s for s in at.selectbox if s.label == "또는 데모 영상"][0].set_value("격발 직전 총구 하강")
    [b for b in at.button if b.key == "pose_go"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("저킹" in s.value for s in at.subheader)
