from shootcoach.explain.vlm import mentions_cause, template_explanation, validate_explanation

REPORT = {"stage1": {"shape_ko": "산개", "sector_ko": "7시 반", "notes": [], "candidates": [
    {"cause_id": "jerking", "cause_ko": "방아쇠 급격히 당김 (저킹)", "guidance_ko": "곧게 누르세요.", "drill_ko": "건식 사격"}]}}


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
