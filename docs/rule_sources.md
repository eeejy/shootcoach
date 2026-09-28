# 진단 규칙 근거

규칙표(`rules/stage1_rules.csv`, `rules/causes.csv`)는 아래 세 자료를 기준으로 작성했다. **오른손잡이 기준**이며, 왼손잡이는 좌우를 뒤집어 적용한다(TSC 원문 지침).

| 약어 | 자료 | 사용한 내용 |
|---|---|---|
| **FM** | U.S. Army FM 3-23.35 *Combat Training with Pistols*, Ch.2 (공공 문서) | 힐링 → 사격하는 손 쪽 위, 방아쇠 오른쪽 면 압력 → 왼쪽, 반동 예상 → 총을 아래로 누름, BRASS 호흡, 볼앤더미 |
| **USAMU** | U.S. Army Marksmanship Unit *Pistol Marksmanship Training Guide* | 저킹 → 주로 왼쪽 아래(오른손잡이), 반동 예상 → 힐링(약 1시 방향), 조준기는 한 발이 아니라 탄착군으로 조정 |
| **TSC** | TargetShooting Canada *Pistol Group Analysis* (2003) | 위치 9종·모양(세로/가로/산개)별 원인 후보 |

링크
- FM 3-23.35 Ch.2 — https://www.globalsecurity.org/military/library/policy/army/fm/3-23-35/chap2.htm
- TC 3-23.35 (2017, 후속 교범) — https://irp.fas.org/doddir/army/tc3-23-35.pdf
- USAMU Pistol Marksmanship Training Guide — https://www.lakeis.org/documents/army%20shooting%20manual.pdf
- TargetShooting Canada, Pistol Group Analysis — https://georgia4h.org/wp-content/uploads/2018/05/grp-analysis.pdf
- 비판: Lucky Gunner — https://www.luckygunner.com/lounge/diagnostic-pistol-target-waste-time/ , Dry Fire Training Cards — https://dryfiretrainingcards.com/blog/shooting-correction-charts-the-big-lie/

## 한계 (반드시 팀장 교범 검토)

- 진단 차트는 사수마다 맞지 않을 수 있다는 비판이 있다(Lucky Gunner, Dry Fire Training Cards 등). 그래서 1단계는 **후보**만 내고, 2단계에서 자세 신호로 확정한다.
- `prior` 값은 자료에서 언급 빈도·강도를 보고 정한 **초기값**이다. 해경 교관 판정 데이터로 다시 맞춰야 한다.
- 10시 반(왼쪽 위) 방향은 원 자료에 명시가 없어 정렬 오류·그립만 낮은 가중치로 넣었다.
- K5 등 조준기 조정 가능 여부·1클릭 값은 기종별로 다르다. `click_mm_per_10m` 값을 넣어야 클릭 수를 계산한다.
