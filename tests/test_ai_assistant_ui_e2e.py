"""
E2E browser tests for Checkpoint 5C: AI Strategy Assistant Frontend UI.
Tests the interactive drawer, open/close, suggested prompts, query submission,
loading/error states, key numbers, and evidence-chip navigation.
"""

import re
import time
import socket
import subprocess
import sys
import urllib.request
import pytest
from playwright.sync_api import sync_playwright, expect
from src.race.ai.orchestrator import set_global_llm_provider
from src.race.ai.provider import MockLLMProvider


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_assistant_server():
    """
    Spawns uvicorn server with MockLLMProvider configured so the UI
    can submit real HTTP queries and receive grounded deterministic answers.
    """
    port = find_free_port()
    # Configure mock provider globally for server subprocess via env or startup script
    cmd = [
        sys.executable,
        "-c",
        (
            "from src.race.ai.orchestrator import set_global_llm_provider; "
            "from src.race.ai.provider import MockLLMProvider; "
            "set_global_llm_provider(MockLLMProvider()); "
            "import uvicorn; from src.race.api.app import app; "
            f"uvicorn.run(app, host='127.0.0.1', port={port}, log_level='warning')"
        ),
    ]
    proc = subprocess.Popen(cmd)

    max_wait = 45.0
    start = time.time()
    ready = False
    while time.time() - start < max_wait:
        if proc.poll() is not None:
            break
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/races")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.2)

    if not ready:
        proc.kill()
        pytest.fail(f"Assistant live server failed to start on port {port}")

    yield f"http://127.0.0.1:{port}"

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_assistant_drawer_open_close_and_accessibility(live_assistant_server):
    """Verify assistant drawer toggle in header, open/close buttons, and Escape key dismissal."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(live_assistant_server, wait_until="networkidle")

        # 1. Assistant button is visible in header bar
        toggle_btn = page.locator("#btn-toggle-assistant")
        expect(toggle_btn).to_be_visible()
        expect(toggle_btn).to_contain_text("AI Assistant")

        # Drawer container is initially not rendered/visible
        drawer = page.locator("#assistant-drawer-container")
        expect(drawer).not_to_be_visible()

        # 2. Click toggle button to open drawer
        toggle_btn.click()
        expect(drawer).to_be_visible()
        expect(page.locator(".assistant-header-title h3")).to_contain_text("AI Strategy Assistant")

        # 3. Model identifier and status pill are visible
        expect(page.locator(".model-id")).to_be_visible()
        status_pill = page.locator("#assistant-status-pill")
        expect(status_pill).to_be_visible()
        expect(status_pill).to_contain_text("Online")

        # 4. Suggested prompts appear in empty state
        suggested_cards = page.locator(".suggested-prompt-card")
        expect(suggested_cards.first).to_be_visible()
        assert suggested_cards.count() >= 5

        # 5. Click close button (X) to dismiss
        close_btn = page.locator("#btn-close-assistant")
        expect(close_btn).to_be_visible()
        close_btn.click()
        expect(drawer).not_to_be_visible()

        # 6. Reopen and test Escape key dismissal
        toggle_btn.click()
        expect(drawer).to_be_visible()
        page.keyboard.press("Escape")
        expect(drawer).not_to_be_visible()

        browser.close()


def test_assistant_query_submission_loading_and_answer(live_assistant_server):
    """Verify submitting a query renders loading state, answer text, key numbers, and evidence chips."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(live_assistant_server, wait_until="networkidle")

        # Open drawer
        page.locator("#btn-toggle-assistant").click()
        expect(page.locator("#assistant-drawer-container")).to_be_visible()

        # 1. Click the first suggested prompt ("Who is leading the race as of now?")
        first_suggested = page.locator("#suggested-prompt-0")
        expect(first_suggested).to_be_visible()
        first_suggested.click()

        # Verify textarea populated
        prompt_input = page.locator("#assistant-prompt-input")
        expect(prompt_input).to_have_value("Who is leading the race as of now?")

        # 2. Submit via button
        submit_btn = page.locator("#assistant-submit-btn")
        expect(submit_btn).to_be_enabled()
        submit_btn.click()

        # 3. Verify user message appears in feed
        user_msg = page.locator(".user-turn .user-message-bubble")
        expect(user_msg.first).to_be_visible()
        expect(user_msg.first).to_contain_text("Who is leading the race as of now?")

        # 4. Wait for assistant response card
        bot_card = page.locator(".bot-turn .bot-message-card")
        expect(bot_card.first).to_be_visible(timeout=10000)

        # 5. Verify answer text and confidence pill
        expect(page.locator(".bot-answer-text").first).to_be_visible()
        expect(page.locator(".confidence-pill").first).to_be_visible()

        # 6. Verify key metrics grid rendered
        key_numbers = page.locator(".key-number-card")
        expect(key_numbers.first).to_be_visible()

        # 7. Verify evidence chip rendered
        evidence_chips = page.locator(".evidence-chip")
        expect(evidence_chips.first).to_be_visible()

        browser.close()


def test_assistant_evidence_chip_navigation(live_assistant_server):
    """Verify that clicking an evidence chip deep-links to the corresponding tab view."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(live_assistant_server, wait_until="networkidle")

        # Open drawer
        page.locator("#btn-toggle-assistant").click()
        expect(page.locator("#assistant-drawer-container")).to_be_visible()

        # Ask a comparison question: "Compare lap times between VER and LEC."
        page.locator("#suggested-prompt-1").click()
        page.locator("#assistant-submit-btn").click()

        # Wait for bot response with comparison evidence chip
        bot_card = page.locator(".bot-turn .bot-message-card")
        expect(bot_card.first).to_be_visible(timeout=10000)

        # Locate the evidence chip
        comp_chip = page.locator(".evidence-chip-comparison").first
        expect(comp_chip).to_be_visible()

        # Click the evidence chip
        comp_chip.click()

        # Verify that active view switched to Driver Comparison
        comparison_tab = page.locator("#tab-driver-comparison")
        expect(comparison_tab).to_have_class(re.compile(r"\bactive\b"))
        expect(page.locator("#driver-comparison-config-bar")).to_be_visible()

        browser.close()


def test_assistant_clear_conversation(live_assistant_server):
    """Verify clearing conversation restores empty state and suggested prompts."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(live_assistant_server, wait_until="networkidle")

        page.locator("#btn-toggle-assistant").click()
        expect(page.locator("#assistant-drawer-container")).to_be_visible()

        # Submit a query
        page.locator("#suggested-prompt-0").click()
        page.locator("#assistant-submit-btn").click()

        expect(page.locator(".user-turn").first).to_be_visible()

        # Clear conversation button should now be visible in header
        clear_btn = page.locator("#btn-clear-chat")
        expect(clear_btn).to_be_visible()
        clear_btn.click()

        # Turns removed and suggested prompts returned
        expect(page.locator(".user-turn")).not_to_be_visible()
        expect(page.locator(".suggested-prompt-card").first).to_be_visible()

        browser.close()


@pytest.fixture(scope="module")
def live_no_key_server():
    """Spawns uvicorn server without any API key or mock provider to test unavailable state."""
    port = find_free_port()
    cmd = [
        sys.executable,
        "-c",
        (
            "import os; os.environ['GEMINI_API_KEY'] = ''; "
            "from src.race import config; config.load_dotenv_file = lambda *a, **k: None; "
            "import uvicorn; from src.race.api.app import app; "
            f"uvicorn.run(app, host='127.0.0.1', port={port}, log_level='warning')"
        ),
    ]
    proc = subprocess.Popen(cmd)

    max_wait = 45.0
    start = time.time()
    ready = False
    while time.time() - start < max_wait:
        if proc.poll() is not None:
            break
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/races")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.2)

    if not ready:
        proc.kill()
        pytest.fail(f"No-key server failed to start on port {port}")

    yield f"http://127.0.0.1:{port}"

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_assistant_missing_api_key_banner(live_no_key_server):
    """Verify missing API key shows amber status and explanatory banner in UI."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(live_no_key_server, wait_until="networkidle")

        # Open assistant
        page.locator("#btn-toggle-assistant").click()
        expect(page.locator("#assistant-drawer-container")).to_be_visible()

        # Check status pill indicates missing key
        status_pill = page.locator("#assistant-status-pill")
        expect(status_pill).to_be_visible()
        expect(status_pill).to_contain_text("API Key Missing")

        # Check explanatory banner is present
        banner = page.locator("#assistant-key-notice")
        expect(banner).to_be_visible()
        expect(banner).to_contain_text("GEMINI_API_KEY Not Configured")

        browser.close()


def test_assistant_responsive_layout(live_assistant_server):
    """Verify assistant drawer displays responsively on mobile viewports."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Mobile viewport width: 400px
        page = browser.new_page(viewport={"width": 400, "height": 800})
        page.goto(live_assistant_server, wait_until="networkidle")

        page.locator("#btn-toggle-assistant").click()
        drawer = page.locator("#assistant-drawer-container")
        expect(drawer).to_be_visible()

        box = drawer.bounding_box()
        assert box is not None
        # On small viewport, drawer width expands to full viewport width (400px)
        assert box["width"] >= 380, f"Expected drawer width ~400px, got {box['width']}"

        browser.close()


@pytest.fixture(scope="module")
def live_quota_exhausted_server():
    """Spawns uvicorn server with quota cooldown active to verify UI quota exhaustion presentation."""
    port = find_free_port()
    cmd = [
        sys.executable,
        "-c",
        (
            "import time, os; "
            "from src.race.ai.provider import GeminiLLMProvider, NvidiaLLMProvider; "
            "GeminiLLMProvider._global_quota_cooldown_until = time.time() + 3600.0; "
            "GeminiLLMProvider._global_quota_retry_str = '16h 15m'; "
            "NvidiaLLMProvider._global_quota_cooldown_until = time.time() + 3600.0; "
            "NvidiaLLMProvider._global_quota_retry_str = '16h 15m'; "
            "import uvicorn; from src.race.api.app import app; "
            f"uvicorn.run(app, host='127.0.0.1', port={port}, log_level='warning')"
        ),
    ]
    proc = subprocess.Popen(cmd)

    max_wait = 45.0
    start = time.time()
    ready = False
    while time.time() - start < max_wait:
        if proc.poll() is not None:
            break
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/races")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.2)

    if not ready:
        proc.kill()
        pytest.fail(f"Quota exhausted server failed to start on port {port}")

    yield f"http://127.0.0.1:{port}"

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_assistant_quota_exhausted_ui(live_quota_exhausted_server):
    """Verify quota exhausted state displays orange status pill, informative banner, and authoritative model."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(live_quota_exhausted_server, wait_until="networkidle")

        # Open assistant
        page.locator("#btn-toggle-assistant").click()
        expect(page.locator("#assistant-drawer-container")).to_be_visible()

        # Check status pill indicates quota exhausted
        status_pill = page.locator("#assistant-status-pill")
        expect(status_pill).to_be_visible()
        expect(status_pill).to_contain_text("Quota Exhausted")

        # Check explanatory banner is present with retry info
        banner = page.locator("#assistant-key-notice")
        expect(banner).to_be_visible()
        expect(banner).to_contain_text("API Quota Exhausted")
        expect(banner).to_contain_text("16h 15m")

        # Model identifier must NOT display fallback gemini-2.5-flash
        model_id = page.locator(".model-id")
        expect(model_id).to_be_visible()
        expect(model_id).not_to_contain_text("gemini-2.5-flash")

        browser.close()

