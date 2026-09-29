"""공개 탄공 데이터셋 다운로드 (Roboflow API 키 필요: .env 의 ROBOFLOW_API_KEY).

    python scripts/download_public_datasets.py
"""
import io
import os
import zipfile
from pathlib import Path

import requests

DATASETS = [  # (workspace, project, version, 저장 이름, 라이선스)
    ("project-bat-bullet-hole-detection", "bullet-hole-object-detection", 30, "butt_bullet_holes", "CC BY 4.0"),
    ("knsashootingtargets", "knsa-shooting-target", 10, "knsa", "CC BY 4.0"),
    ("project-bat-bullet-hole-detection", "carbine-target-bullet-hole-detection", 1, "carbine", "CC BY 4.0"),
    ("kanat-hefkh", "kanat2.0-yolov11-bullets-targets-and-boards", 5, "kanat", "Public Domain"),
]


def load_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    env = Path(__file__).resolve().parents[1] / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY 가 없습니다 (.env 또는 환경변수).")
    return key


def main():
    key = load_key()
    for ws, proj, ver, name, lic in DATASETS:
        dst = Path("data/public") / name
        if (dst / "data.yaml").exists():
            print("있음:", name)
            continue
        j = requests.get(f"https://api.roboflow.com/{ws}/{proj}/{ver}/yolov8", params={"api_key": key}, timeout=120).json()
        link = j["export"]["link"]
        print(f"받는 중: {name} ({lic})")
        data = requests.get(link, timeout=1200).content
        zipfile.ZipFile(io.BytesIO(data)).extractall(dst)
    print("완료 → python scripts/build_public_dataset.py")


if __name__ == "__main__":
    main()
