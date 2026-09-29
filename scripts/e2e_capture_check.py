"""촬영 가이드 E2E 확인: 크롬 가짜 카메라에 표적지 영상을 넣고 셔터 → 분석 결과까지 자동 확인.

    python app/capture_server.py --https &      # 먼저 서버 실행
    python scripts/e2e_capture_check.py out/fake_cam.y4m
(y4m 만들기: ffmpeg -loop 1 -i samples/demo_jerking_low_left.jpg -t 6 -vf "scale=-2:1000,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p" -r 10 out/fake_cam.y4m)
"""
import asyncio
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright


async def main(y4m: str, url: str = "https://127.0.0.1:8600/"):
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={Path(y4m).resolve()}"])
        ctx = await b.new_context(ignore_https_errors=True, permissions=["camera"], viewport={"width": 390, "height": 844})
        page = await ctx.new_page()
        await page.goto(url)
        t0 = time.time()
        await page.wait_for_function("() => !document.getElementById('shutter').disabled", timeout=60000)
        print(f"셔터 활성화: {time.time() - t0:.1f}s · 상태: {await page.inner_text('#status')}")
        print("사진 모드 버튼 숨김:", not await page.is_visible("#photoRow"))
        await page.screenshot(path="out/capture_live_guide.png")
        await page.click("#shutter")
        await page.wait_for_selector("#result .card", timeout=60000)
        print("결과:", (await page.inner_text("#result"))[:260].replace("\n", " | "))
        await page.screenshot(path="out/capture_result.png", full_page=True)
        await b.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "out/fake_cam.y4m"))
