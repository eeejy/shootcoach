"""웹앱 화면 캡처 (디자인 확인용): python scripts/shot_ui.py out/ui [--mobile]"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def main(out: str, mobile: bool):
    Path(out).mkdir(parents=True, exist_ok=True)
    vp = {"width": 390, "height": 2400} if mobile else {"width": 1440, "height": 2000}
    tag = "m" if mobile else "d"
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True)
        page = await b.new_page(viewport=vp, device_scale_factor=1)
        await page.goto("http://localhost:8501/")
        await page.wait_for_selector("text=분석하기", timeout=60000)
        await asyncio.sleep(2)
        await page.screenshot(path=f"{out}/{tag}1_home.png", full_page=True)
        await page.get_by_text("또는 데모 사진").first.wait_for()
        sel = page.locator('[data-testid="stSelectbox"]').filter(has_text="데모 사진").first
        await sel.click()
        await page.get_by_role("option", name="demo_jerking_low_left.jpg").click()
        await page.get_by_role("button", name="분석하기").click()
        await page.wait_for_selector("text=탄착군:", timeout=60000)
        await asyncio.sleep(2)
        await page.screenshot(path=f"{out}/{tag}2_result.png", full_page=True)
        await page.get_by_role("tab").nth(1).click()
        await asyncio.sleep(1)
        sim = page.locator('[data-testid="stSelectbox"]').filter(has_text="합성").first
        await sim.click()
        await page.get_by_role("option", name="격발 직전 총구 하강").click()
        await page.get_by_role("button", name="자세 분석").click()
        await page.wait_for_selector("text=확정", timeout=60000)
        await asyncio.sleep(2)
        await page.screenshot(path=f"{out}/{tag}3_pose.png", full_page=True)
        await b.close()


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "out/ui", "--mobile" in sys.argv))
