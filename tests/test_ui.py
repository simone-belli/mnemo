"""Browser test of the page (Playwright + Chromium) against the app in fake mode.

The CDN libraries are served from tests/node_modules (`cd tests && npm install`), so this
runs offline. Skipped when Playwright or the local libraries are missing.
"""
import re
import socket
import threading
import time
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")

MODULES = Path(__file__).parent / "node_modules"
LOCAL = {
    "marked@12.0.2/marked.min.js": MODULES / "marked/marked.min.js",
    "katex@0.16.11/dist/katex.min.js": MODULES / "katex/dist/katex.min.js",
    "katex@0.16.11/dist/katex.min.css": MODULES / "katex/dist/katex.min.css",
}
if not all(p.exists() for p in LOCAL.values()):
    pytest.skip("run `npm install` in tests/ for the local CDN copies", allow_module_level=True)


@pytest.fixture
def server(svc):
    from app.main import create_app
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(svc), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(5)


@pytest.fixture
def page(server):
    with pw.sync_playwright() as p:
        exe = Path("/opt/pw-browsers/chromium")
        browser = p.chromium.launch(executable_path=str(exe) if exe.is_file() else None,
                                    args=["--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 390, "height": 844})   # iPhone-sized

        def route(r):
            url = r.request.url
            for key, path in LOCAL.items():
                if url.endswith(key):
                    ctype = "text/css" if key.endswith(".css") else "application/javascript"
                    return r.fulfill(body=path.read_bytes(), content_type=ctype)
            return r.fulfill(body="", content_type="text/css")      # fonts etc.: not needed offline

        ctx.route(re.compile(r"^https://"), route)
        pg = ctx.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(server)
        yield pg
        assert not errors, errors
        browser.close()


def test_first_run_onboarding_then_quiz(page, svc):
    page.wait_for_selector("#onboarding:not([hidden])")
    assert page.locator(".rating").count() == 6
    page.fill("#ratings input >> nth=1", "5")
    page.click("#rate-go")
    page.wait_for_selector("#quiz-options button")
    with svc.db() as conn:
        right = conn.execute("SELECT answer FROM quiz_items WHERE score IS NULL").fetchone()["answer"]
    page.click(f"#quiz-options button >> nth={right}")
    page.wait_for_selector("#quiz-next:not([hidden])")
    assert "Right" in page.text_content("#quiz-result")
    page.click("#quiz-next")
    page.wait_for_function("document.querySelector('#quiz-bar').style.width !== '0%'")
    page.click("#quiz-stop")
    assert page.is_hidden("#onboarding")


def test_recall_renders_and_highlights(page):
    page.wait_for_selector("#onboarding:not([hidden])")
    page.click("#rate-skip")
    page.click("#quiz-stop")
    page.click(".modes [data-mode=both]")
    page.fill("#q", "remind me about statistical mechanics")
    page.click("#go")
    page.wait_for_selector("#card:not([hidden])")
    assert page.locator("#crumbs .chip").count() == 1
    # Markdown and KaTeX rendered in the written recap
    assert page.locator("#recap .katex").count() >= 1
    assert page.locator("#recap em, #recap strong").count() >= 1
    # The script plays and the current word is lit, moving forward
    page.wait_for_selector("#script .w.now", timeout=5000)
    first = int(page.get_attribute("#script .w.now", "data-i"))
    page.wait_for_timeout(1200)
    later = int(page.get_attribute("#script .w.now", "data-i"))
    assert later > first
    assert page.locator("#script .w.read").count() >= later
    # Next-step chips
    assert 2 <= page.locator("#chips button").count() <= 3
    # Feedback: a button is acknowledged
    page.click("#feedback [data-signal=knew_that]")
    page.wait_for_selector("#feedback [data-signal=knew_that].done")
    # Lost me asks for a note, then reports the diagnosis
    page.click("#feedback [data-signal=lost_me]")
    page.fill("#lost-note", "partition function")
    page.click("#lost button")
    page.wait_for_function("document.querySelector('#fb-result').textContent.includes('partition function')")


def test_modes_are_remembered(page):
    page.wait_for_selector("#onboarding:not([hidden])")
    page.click("#rate-skip")
    page.click("#quiz-stop")
    page.click(".modes [data-mode=read]")
    page.fill("#q", "remind me about statistical mechanics")
    page.click("#go")
    page.wait_for_selector("#card:not([hidden])")
    assert page.is_visible("#recap") and page.is_hidden("#script")
    assert page.evaluate("mnemo.current.audio_url") is None          # Read mode: no audio generated
    page.click(".modes [data-mode=listen]")
    page.wait_for_function("mnemo.current.audio_url !== null")
    assert page.is_hidden("#recap") and page.is_visible("#script")
    page.reload()
    assert page.get_attribute(".modes [data-mode=listen]", "aria-checked") == "true"


def test_chip_runs_a_recall(page):
    page.wait_for_selector("#onboarding:not([hidden])")
    page.click("#rate-skip")
    page.click("#quiz-stop")
    page.fill("#q", "tell me about art")
    page.click("#go")
    page.wait_for_selector("#chips button")
    first_path = page.text_content("#crumbs")
    page.click("#chips button >> nth=0")
    page.wait_for_function(f"document.querySelector('#crumbs').textContent !== {first_path!r}")
    assert len(page.locator("#crumbs span:not(.sep):not(.chip)").all()) >= 3


def test_theme_switch_cycles_and_persists(page):
    page.wait_for_selector("#onboarding:not([hidden])")
    bg = lambda: page.evaluate("getComputedStyle(document.body).backgroundColor")
    light = bg()
    assert page.text_content("#theme") == "Theme: auto"
    page.click("#theme")
    assert page.text_content("#theme") == "Theme: light" and bg() == light
    page.click("#theme")
    assert page.text_content("#theme") == "Theme: dark" and bg() != light
    page.reload()
    page.wait_for_selector("#onboarding:not([hidden])")
    assert page.text_content("#theme") == "Theme: dark" and bg() != light
    page.click("#theme")
    assert page.text_content("#theme") == "Theme: auto" and bg() == light
