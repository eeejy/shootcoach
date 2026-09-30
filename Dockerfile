# BullsAI — AI 사격 교정 웹앱 (Streamlit) 컨테이너 이미지
# opencv-contrib-python + ultralytics(torch) 실행을 위한 시스템 라이브러리를 포함한다.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV / Ultralytics 런타임에 필요한 최소 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치해 레이어 캐시를 활용 (torch는 CPU 휠로 설치해 이미지 경량화)
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 애플리케이션 소스
COPY . .

# 컨테이너에서는 localhost 바인딩이 외부 접근을 막으므로, 로컬 전용 config의
# address 라인만 제거한다(테마 등 나머지 설정은 유지). 실제 바인딩은 아래 CMD가 지정.
RUN sed -i '/^address = "localhost"/d' .streamlit/config.toml || true

# Streamlit 기본 포트
EXPOSE 8501

# 컨테이너 헬스체크 (Streamlit 내장 엔드포인트)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status==200 else 1)" || exit 1

# 컨테이너에서는 모든 인터페이스에 바인딩해야 외부에서 접근 가능하다.
# XSRF/CORS는 리버스 프록시 뒤에서 동작하도록 완화한다.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
