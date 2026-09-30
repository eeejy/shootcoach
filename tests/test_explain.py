from shootcoach.explain.vlm import (
    mentions_cause, template_explanation, template_explanation_posture, validate_explanation,
    validate_explanation_posture,
)

REPORT = {"stage1": {"shape_ko": "산개", "sector_ko": "7시 반", "notes": [], "candidates": [
    {"cause_id": "jerking", "cause_ko": "방아쇠 급격히 당김 (저킹)", "guidance_ko": "곧게 누르세요.", "drill_ko": "건식 사격"}]}}

STAGE1_CANDIDATES = [{"cause_id": "L1", "cause_ko": "반동 예측 (Recoil Anticipation)",
                      "guidance_ko": "반동을 예상해 총구를 누릅니다.", "drill_ko": "볼앤더미 훈련"}]
STAGE2_CONFIRMED = {"final_cause_id": "L1", "final_ko": "확정 — 반동 예측", "notes": [],
                    "verdicts": [{"cause_id": "L1", "status_ko": "확정", "evidence": ["총구 하강: 강도 1.4"]}]}
STAGE2_HOLD = {"final_cause_id": None, "final_ko": "보류", "notes": ["격발 시점을 찾지 못했습니다."], "verdicts": []}


def test_mentions_cause_alias_and_inflection():
    assert mentions_cause("방아쇠를 급격히 당겨 저킹이 발생했습니다", "방아쇠 급격히 당김 (저킹)")
    assert mentions_cause("저킹 가능성이 높습니다", "방아쇠 급격히 당김 (저킹)")
    assert not mentions_cause("총을 기울이는 습관입니다", "방아쇠 급격히 당김 (저킹)")


def test_validate_rejects_leak_and_wrong_cause():
    assert validate_explanation("분석 결과: 탄 수 8발 ... 사수에게 할 설명:", REPORT)
    assert validate_explanation("총기를 기울이는 습관이 원인입니다. 오늘의 훈련: 수평 유지", REPORT)
    assert validate_explanation("분석 결과에 따르면 저킹 가능성이 큽니다. 오늘의 훈련: 건식 사격", REPORT) is None


def test_template_mentions_primary():
    assert "저킹" in template_explanation(REPORT)


def test_posture_explanation_independent_of_target():
    """표적지 분석(REPORT) 없이도 자세 영상 결과만으로 문단을 만들 수 있다."""
    text = template_explanation_posture(STAGE2_CONFIRMED, STAGE1_CANDIDATES)
    assert "반동 예측" in text and "볼앤더미" in text


def test_posture_explanation_holds_without_confirmation():
    text = template_explanation_posture(STAGE2_HOLD, STAGE1_CANDIDATES)
    assert "보류" in text


def test_validate_posture_rejects_wrong_cause():
    assert validate_explanation_posture("총기를 기울이는 습관입니다. 오늘의 훈련: 수평 유지", STAGE2_CONFIRMED, STAGE1_CANDIDATES)
    assert validate_explanation_posture(
        "자세 영상에서 반동 예측 신호가 확인됩니다. 오늘의 훈련: 볼앤더미", STAGE2_CONFIRMED, STAGE1_CANDIDATES) is None
