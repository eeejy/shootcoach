"""The Streamlit app must load and the main flows must run without exceptions."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_no_labeling_or_print_features():
    at = _load()
    labels = [t.label for t in at.tabs]
    assert "표적지 인쇄" not in labels and not (ROOT / "app" / "pages").exists()


def test_main_page_target_and_posture_flow():
    at = _load()
    [s for s in at.selectbox if s.label == "또는 데모 사진"][0].set_value("demo_low_left.jpg")
    [b for b in at.button if b.label == "분석"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(" · " in s.value for s in at.subheader)          # shape · direction headline
    [s for s in at.selectbox if s.label == "또는 데모 영상"][0].set_value("격발 직전 총구 하강")
    [b for b in at.button if b.key == "pose_go"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("확정" in s.value or "보류" in s.value for s in at.subheader)
