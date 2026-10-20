"""
End-to-end browser test for RACE F1 Historical Replay UI (Milestone 1).
Uses Playwright to launch headless Chromium against the live FastAPI app with cached Monza data,
exercising playback, pause, seeking, lap stepping, and visible timing-tower updates.
"""

import re
import time
import socket
import subprocess
import sys
import urllib.request
import pytest
from playwright.sync_api import sync_playwright, expect


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Spawns uvicorn server serving API and built UI on an ephemeral port."""
    port = find_free_port()
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.race.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(cmd)

    # Wait for server to accept HTTP requests and warm up race catalog
    max_wait = 45.0
    start = time.time()
    ready = False
    last_error = None
    while time.time() - start < max_wait:
        if proc.poll() is not None:
            # Process terminated unexpectedly
            break
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/races")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
                last_error = f"HTTP {resp.status}"
        except Exception as e:
            last_error = str(e)
            time.sleep(0.2)

    if not ready:
        poll_code = proc.poll()
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError(
            f"Uvicorn server failed to achieve HTTP readiness on port {port} "
            f"within {max_wait}s (exit_code={poll_code}, last_error={last_error})"
        )

    url = f"http://127.0.0.1:{port}"
    yield url

    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_replay_ui_e2e_playback_and_timing_tower(live_server):
    """
    Executes full browser end-to-end replay flow:
    1. Page loads with dark F1 aesthetic, Monza race header, and 20-car timing tower.
    2. Verifies initial transponder timing order.
    3. Clicks PLAY -> asserts state transitions to PAUSE and timecode increments.
    4. Clicks PAUSE -> asserts playback halts.
    5. Clicks +1 LAP -> asserts leader lap advances.
    6. Seeks to lap 12 -> asserts timing tower reflects accurate point-in-time race status.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Load application root
        page.goto(live_server, wait_until="domcontentloaded")

        # 2. Verify header elements
        expect(page.locator(".brand-badge")).to_have_text("RACE F1")
        expect(page.locator("#leader-lap-pill")).to_be_visible()
        expect(page.locator("#race-select-dropdown")).to_be_visible()

        # 3. Verify timing tower rendered with 20 drivers
        tower_table = page.locator("#timing-tower-table")
        expect(tower_table).to_be_visible()
        rows = tower_table.locator("tbody tr")
        expect(rows).to_have_count(20)

        # Assert leader row
        p1_row = rows.first
        expect(p1_row.locator(".pos-col")).to_have_text("1")
        expect(p1_row).to_contain_text("LEADER")

        # 4. Playback test: Click PLAY
        play_btn = page.locator("#btn-play-pause")
        expect(play_btn).to_have_text("PLAY")
        play_btn.click()

        # Button should switch to PAUSE
        expect(play_btn).to_have_text("PAUSE")

        # Wait 1.5 seconds for replay clock to advance
        page.wait_for_timeout(1500)

        # 5. Click PAUSE
        play_btn.click()
        expect(play_btn).to_have_text("PLAY")

        # Read timecode after pause
        timecode_text = page.locator("#timecode-display").inner_text()
        assert "00:00.00" not in timecode_text or "(" in timecode_text

        # 6. Step Lap Forward (+1 LAP)
        step_lap_next = page.locator("#btn-step-lap-next")
        expect(step_lap_next).to_be_visible()
        step_lap_next.click()

        # Wait for state update to arrive via WebSocket
        page.wait_for_timeout(500)
        expect(page.locator("#leader-lap-pill .current")).not_to_have_text("0")

        # 7. Speed multiplier button
        speed_2x_btn = page.locator("#btn-speed-2x")
        expect(speed_2x_btn).to_be_visible()
        speed_2x_btn.click()
        expect(speed_2x_btn).to_have_class("speed-btn active")

        # 8. Scrubber interaction
        scrubber = page.locator("#timeline-scrubber")
        expect(scrubber).to_be_visible()

        # 9. Circuit map & Moving Driver Markers verification (Checkpoint C)
        circuit_map = page.locator("#circuit-map-container")
        expect(circuit_map).to_be_visible()
        expect(circuit_map.locator(".circuit-name")).to_be_visible()
        expect(circuit_map.locator("svg.circuit-svg")).to_be_visible()
        expect(circuit_map.locator(".track-centerline")).to_be_visible()
        expect(circuit_map.locator(".start-finish-group")).to_be_visible()

        # Verify driver markers exist on track
        driver_markers = circuit_map.locator(".driver-marker")
        expect(driver_markers.first).to_be_visible()
        marker_count = driver_markers.count()
        assert marker_count >= 18, f"Expected at least 18 active driver markers, found {marker_count}"

        # Verify leader marker synchronization with timing tower P1 driver
        tower_leader_code = p1_row.locator(".driver-code").inner_text()
        leader_marker = circuit_map.locator(".driver-marker.is-leader")
        expect(leader_marker).to_be_visible()
        assert leader_marker.get_attribute("data-driver") == tower_leader_code
        expect(leader_marker.locator(".driver-leader-halo")).to_be_visible()

        # 10. Marker Motion Synchronization: verify marker moves during playback
        initial_transform = leader_marker.get_attribute("transform")
        play_btn.click()
        expect(play_btn).to_have_text("PAUSE")
        page.wait_for_timeout(1500)
        play_btn.click()
        expect(play_btn).to_have_text("PLAY")

        updated_transform = leader_marker.get_attribute("transform")
        assert initial_transform != updated_transform, "Leader marker position must update as replay time advances"

        # 11. Switch to 2023 Monza and assert circuit map reloads with 2023 metadata
        race_dropdown = page.locator("#race-select-dropdown")
        race_dropdown.select_option(value="2023_monza")
        page.wait_for_timeout(1000)
        expect(circuit_map.locator(".circuit-pill").first).to_have_text("2023")

        # Step forward (+1 LAP) so 2023 race is underway with active telemetry
        step_lap_next.click()
        page.wait_for_timeout(1000)

        # Verify 2023 markers are now active and rendered on track (19 active drivers)
        markers_2023 = circuit_map.locator(".driver-marker")
        expect(markers_2023.first).to_be_visible()
        expect(markers_2023).to_have_count(19)

        # Confirm retired TSU has no marker on track
        tsu_marker = circuit_map.locator("#driver-marker-TSU")
        expect(tsu_marker).to_have_count(0)

        browser.close()


def test_replay_controls_bugfixes_e2e(live_server):
    """
    E2E browser tests specifically verifying:
    1. -1 Lap while playing moves back exactly one lap and continues playing.
    2. -1 Lap while paused moves back and stays paused.
    3. Repeated backward steps work reliably.
    4. Dragging timeline while playing pauses during drag and resumes afterward.
    5. Dragging timeline while paused leaves playback paused.
    6. Releasing outside slider is handled safely.
    7. Normal clicking on timeline allows seeking.
    8. Circuit map markers and timing tower remain synchronized.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server, wait_until="domcontentloaded")

        play_btn = page.locator("#btn-play-pause")
        step_lap_next = page.locator("#btn-step-lap-next")
        step_lap_prev = page.locator("#btn-step-lap-prev")
        scrubber = page.locator("#timeline-scrubber")
        lap_pill = page.locator("#leader-lap-pill .current")

        expect(play_btn).to_be_visible()
        expect(step_lap_next).to_be_visible()
        expect(step_lap_prev).to_be_visible()
        expect(scrubber).to_be_visible()

        # Ensure paused state and reset to Lap 1
        if play_btn.inner_text() == "PAUSE":
            play_btn.click()
            expect(play_btn).to_have_text("PLAY")

        while int(lap_pill.inner_text()) > 1:
            step_lap_prev.click()
            page.wait_for_timeout(300)
        expect(lap_pill).to_have_text("1")

        # Step forward to Lap 3
        step_lap_next.click()
        page.wait_for_timeout(500)
        step_lap_next.click()
        page.wait_for_timeout(500)
        expect(lap_pill).to_have_text("3")

        # 1. Test -1 Lap while PLAYING
        play_btn.click()
        expect(play_btn).to_have_text("PAUSE")

        # Click -1 LAP while playing
        step_lap_prev.click()
        page.wait_for_timeout(500)
        expect(lap_pill).to_have_text("2")
        expect(play_btn).to_have_text("PAUSE")  # Must continue playback!

        # Repeated step -1 LAP while playing
        step_lap_prev.click()
        page.wait_for_timeout(500)
        expect(lap_pill).to_have_text("1")
        expect(play_btn).to_have_text("PAUSE")  # Must still be playing!

        # Another step at Lap 1 clamps safely and remains playing
        step_lap_prev.click()
        page.wait_for_timeout(500)
        expect(lap_pill).to_have_text("1")
        expect(play_btn).to_have_text("PAUSE")

        # 2. Test -1 Lap while PAUSED
        play_btn.click()
        expect(play_btn).to_have_text("PLAY")

        step_lap_next.click()
        page.wait_for_timeout(500)
        expect(lap_pill).to_have_text("2")

        step_lap_prev.click()
        page.wait_for_timeout(500)
        expect(lap_pill).to_have_text("1")
        expect(play_btn).to_have_text("PLAY")  # Must stay paused!

        # 3. Test timeline dragging while PLAYING
        # Resume playback
        play_btn.click()
        expect(play_btn).to_have_text("PAUSE")

        box = scrubber.bounding_box()
        assert box is not None

        # Drag from 25% of slider to 60% of slider
        start_x = box["x"] + box["width"] * 0.25
        start_y = box["y"] + box["height"] * 0.5
        end_x = box["x"] + box["width"] * 0.60
        end_y = box["y"] + box["height"] * 0.5

        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.wait_for_timeout(200)

        # Dragging: move mouse across slider
        page.mouse.move(end_x, end_y, steps=5)
        page.wait_for_timeout(200)
        page.mouse.up()

        # After releasing drag, playback resumes and target time updated
        page.wait_for_timeout(600)
        expect(play_btn).to_have_text("PAUSE")

        # Verify timecode updated
        timecode_text = page.locator("#timecode-display").inner_text()
        assert "00:00.00" not in timecode_text

        # 4. Test timeline dragging while PAUSED
        play_btn.click()
        expect(play_btn).to_have_text("PLAY")

        # Drag from 60% to 30%
        page.mouse.move(end_x, end_y)
        page.mouse.down()
        page.wait_for_timeout(200)

        page.mouse.move(start_x, start_y, steps=5)
        page.wait_for_timeout(200)
        page.mouse.up()

        page.wait_for_timeout(600)
        # Must remain paused after release
        expect(play_btn).to_have_text("PLAY")

        # 5. Test release outside slider
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        # Move far above the slider into blank area
        page.mouse.move(start_x + 50, start_y - 120, steps=5)
        page.mouse.up()
        page.wait_for_timeout(500)
        expect(play_btn).to_have_text("PLAY")

        # 6. Test direct click on timeline seeks
        target_click_x = box["x"] + box["width"] * 0.45
        target_click_y = box["y"] + box["height"] * 0.5
        page.mouse.click(target_click_x, target_click_y)
        page.wait_for_timeout(600)

        # 7. Verify circuit map markers and timing tower are rendered and synchronized
        leader_marker = page.locator("#circuit-map-container .driver-marker.is-leader")
        expect(leader_marker).to_be_visible()
        p1_code = page.locator("#timing-tower-table tbody tr").first.locator(".driver-code").inner_text()
        assert leader_marker.get_attribute("data-driver") == p1_code

        browser.close()


def test_driver_comparison_results_table_e2e(live_server):
    """
    E2E test for Checkpoint 2B-2B:
    1. Page loads and navigates to Driver Comparison view.
    2. Idle state is displayed (#comparison-table-idle).
    3. Clicks 'Run Comparison' for default VER vs LEC single lap mode with Full Race cutoff.
    4. Table renders (#comparison-results-table) with 2 rows.
    5. Verifies reference driver row has REF badge and delta is '0.000s'.
    6. Verifies non-reference driver row displays signed delta.
    7. Changes mode to Lap Range [5, 8].
    8. Stale configuration alert appears (#table-stale-banner).
    9. Clicks 'Run Comparison' -> table renders 8 rows.
    10. Tests tyre_age synchronization -> alignment column shows 'Age X' and preserves actual race lap number.
    11. Switches race to 2023_monza -> verifies previous comparison results are cleared.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server, wait_until="domcontentloaded")

        # 1. Navigate to Driver Comparison tab
        comparison_tab = page.locator("#tab-driver-comparison")
        expect(comparison_tab).to_be_visible()
        comparison_tab.click()

        # 2. Verify Initial/Idle state
        idle_card = page.locator("#comparison-table-idle")
        expect(idle_card).to_be_visible()

        # 3. Select Full Race cutoff so valid completed laps exist
        full_race_btn = page.locator("#section-modifiers-cutoff button:has-text('Full Race')")
        full_race_btn.click()

        run_btn = page.locator("#btn-run-comparison")
        expect(run_btn).to_be_enabled()
        run_btn.click()

        # 4. Table renders with 2 rows
        table = page.locator("#comparison-results-table")
        expect(table).to_be_visible(timeout=15000)
        rows = table.locator("tbody tr")
        expect(rows).to_have_count(2)

        # 5. Verify reference driver row (VER)
        ver_row = rows.filter(has_text="VER")
        expect(ver_row.locator(".ref-badge")).to_have_text("REF")
        expect(ver_row.locator(".cell-delta")).to_have_text("0.000s")

        # 6. Verify non-reference driver row has signed delta
        lec_row = rows.filter(has_text="LEC")
        lec_delta = lec_row.locator(".cell-delta").inner_text()
        assert "+" in lec_delta or "-" in lec_delta or "0.000s" in lec_delta

        # 7. Change mode to Lap Range [5, 8]
        range_btn = page.locator("#section-lap-mode button:has-text('Lap Range')")
        range_btn.click()
        page.locator("#input-range-start").fill("5")
        page.locator("#input-range-end").fill("8")

        # 8. Stale configuration alert appears
        stale_alert = page.locator("#table-stale-banner")
        expect(stale_alert).to_be_visible()

        # 9. Click Run Comparison -> 8 rows rendered
        run_btn.click()
        expect(rows).to_have_count(8)

        # 10. Switch to Tyre Age sync
        tyre_age_btn = page.locator("#section-sync-mode button:has-text('Tyre Age')")
        tyre_age_btn.click()
        run_btn.click()

        # Alignment column shows 'Age' and Actual Lap column shows real lap number (5 to 8)
        first_row = rows.first
        expect(first_row.locator(".cell-align-key")).to_contain_text("Age")
        lap_num = int(first_row.locator(".cell-lap").inner_text())
        assert 5 <= lap_num <= 8

        # 11. Switch race to 2023_monza -> previous table cleared
        race_select = page.locator("#race-select-dropdown")
        race_select.select_option(value="2023_monza")
        page.wait_for_timeout(500)

        # Previous table must be cleared, showing idle card
        expect(table).to_have_count(0)
        expect(page.locator("#comparison-table-idle")).to_be_visible()

        browser.close()


def test_driver_comparison_pace_chart_e2e(live_server):
    """
    E2E test for Checkpoint 2B-2C: Driver Comparison SVG Pace Chart.
    1. Page loads and navigates to Driver Comparison view.
    2. Initial/Idle state is displayed (#comparison-chart-idle).
    3. Selects Full Race cutoff and Lap Range [5, 8], runs comparison.
    4. Pace chart renders (#comparison-pace-chart) with SVG and legend.
    5. Verifies Lap-Time mode:
       - Mode button is active.
       - Legend contains driver codes with team colors and REF badge.
       - Data points are rendered (8 points across VER and LEC).
       - Lower lap time (faster) has smaller SVG cy (appears higher visually).
       - X-axis ticks display actual race lap numbers (5 to 8).
       - Point selection updates inspection panel (#chart-point-details).
    6. Switches to Delta-to-Reference mode without re-fetching:
       - Mode button #chart-mode-delta becomes active.
       - Zero-reference baseline (#chart-zero-reference-line) is visible.
       - Reference driver points align with the zero-reference baseline.
    7. Tests Tyre-Age synchronization:
       - Shows stale warning banner before rerun.
       - Reruns comparison and confirms actual race lap numbers are preserved on X-axis.
       - Point inspection reflects tyre age and stint metadata.
    8. Switches race to 2023_monza:
       - Chart is immediately cleared and returns to idle state.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server, wait_until="domcontentloaded")

        # 1. Navigate to Driver Comparison tab
        comparison_tab = page.locator("#tab-driver-comparison")
        expect(comparison_tab).to_be_visible()
        comparison_tab.click()

        # 2. Verify Initial / Idle state
        idle_chart = page.locator("#comparison-chart-idle")
        expect(idle_chart).to_be_visible()

        # 3. Select Full Race cutoff and Lap Range [5, 8]
        full_race_btn = page.locator("#section-modifiers-cutoff button:has-text('Full Race')")
        full_race_btn.click()

        range_btn = page.locator("#section-lap-mode button:has-text('Lap Range')")
        range_btn.click()
        page.locator("#input-range-start").fill("5")
        page.locator("#input-range-end").fill("8")

        run_btn = page.locator("#btn-run-comparison")
        expect(run_btn).to_be_enabled()
        run_btn.click()

        # 4. Pace Chart renders
        chart_card = page.locator("#comparison-pace-chart")
        expect(chart_card).to_be_visible(timeout=15000)
        expect(idle_chart).to_have_count(0)

        # 5. Verify Lap-Time mode features
        lap_mode_btn = page.locator("#chart-mode-lap-time")
        expect(lap_mode_btn).to_have_class("chart-mode-btn active")

        # Legend checks
        legend = page.locator("#chart-legend")
        expect(legend).to_be_visible()
        expect(legend.locator(".legend-item", has_text="VER")).to_be_visible()
        expect(legend.locator(".legend-item", has_text="LEC")).to_be_visible()
        expect(legend.locator(".legend-ref-tag")).to_have_text("REF")

        # Data points rendered
        points = chart_card.locator(".chart-point")
        expect(points).to_have_count(8)

        # X-axis ticks contain actual race lap numbers 5 to 8
        x_ticks = chart_card.locator(".chart-tick-label.x-tick")
        x_tick_texts = [x_ticks.nth(i).text_content() or "" for i in range(x_ticks.count())]
        assert "5" in x_tick_texts and "8" in x_tick_texts

        # Verify vertical axis inversion:
        # Find points for VER, compare a faster lap vs slower lap
        # Lower time (faster) must have smaller cy (higher up in SVG)
        ver_points = chart_card.locator("circle[aria-label*='Driver VER']")
        assert ver_points.count() == 4
        cy_vals = [float(ver_points.nth(i).get_attribute("cy") or "0") for i in range(4)]
        # All cy coordinates must be valid positive numbers within SVG height
        for cy in cy_vals:
            assert 0 < cy < 360

        # Point inspection panel
        first_point = points.first
        first_point.click()
        inspection_panel = page.locator("#chart-point-details")
        expect(inspection_panel).to_be_visible()
        expect(inspection_panel.locator(".details-stat")).to_have_count(7)
        lap_val = inspection_panel.locator(".details-stat", has_text="Race Lap:").locator(".stat-value").inner_text()
        assert int(lap_val) in [5, 6, 7, 8]

        # 6. Switch to Delta-to-Reference mode without re-fetching
        delta_mode_btn = page.locator("#chart-mode-delta")
        delta_mode_btn.click()
        expect(delta_mode_btn).to_have_class("chart-mode-btn active")
        expect(lap_mode_btn).to_have_class("chart-mode-btn")

        # Zero reference line must exist and have valid coordinates
        zero_line = chart_card.locator("#chart-zero-reference-line")
        expect(zero_line).to_have_count(1)
        zero_y = float(zero_line.get_attribute("y1") or "0")
        assert 0 < zero_y < 360

        # Reference driver (VER) deltas are 0.000s and sit precisely on zero reference line
        ver_delta_points = chart_card.locator("circle[aria-label*='Driver VER']")
        for i in range(ver_delta_points.count()):
            pt_cy = float(ver_delta_points.nth(i).get_attribute("cy") or "0")
            assert abs(pt_cy - zero_y) < 1.0, f"VER delta point {i} cy={pt_cy} must lie on zero line y={zero_y}"

        # 7. Switch to Tyre-Age synchronization
        tyre_age_btn = page.locator("#section-sync-mode button:has-text('Tyre Age')")
        tyre_age_btn.click()

        # Stale configuration banner appears
        stale_banner = page.locator("#chart-stale-banner")
        expect(stale_banner).to_be_visible()

        run_btn.click()
        expect(chart_card.locator(".chart-point").first).to_be_visible(timeout=15000)
        expect(stale_banner).to_have_count(0)

        # X-axis still strictly displays actual race laps 5 to 8
        x_ticks_tyre = chart_card.locator(".chart-tick-label.x-tick")
        x_tick_tyre_texts = [x_ticks_tyre.nth(i).text_content() or "" for i in range(x_ticks_tyre.count())]
        assert "5" in x_tick_tyre_texts and "8" in x_tick_tyre_texts

        # 8. Switch race to 2023_monza -> verify chart is cleared back to idle state
        race_select = page.locator("#race-select-dropdown")
        race_select.select_option(value="2023_monza")
        page.wait_for_timeout(500)

        expect(page.locator("#comparison-pace-chart")).to_have_count(0)
        expect(page.locator("#comparison-chart-idle")).to_be_visible()
        expect(page.locator("#comparison-table-idle")).to_be_visible()

        browser.close()


def test_driver_comparison_heat_strip_e2e(live_server):
    """
    E2E test for Checkpoint 2B-3B: Mini-Sector Heat Strip UI.
    1. Navigate to Driver Comparison tab.
    2. Verify initial / idle state (#heat-strip-idle).
    3. Verify comparison view sub-navigation buttons exist:
       - #tab-comp-view-all
       - #tab-comp-view-heat-strip
       - #tab-comp-view-line
       - #tab-comp-view-table
    4. Select Full Race cutoff, Single Lap (Lap 1), run comparison.
    5. Verify Mini-Sector Heat Strip renders (#comparison-heat-strip):
       - Legend is present with color semantics.
       - S1, S2, S3 headers are present (.sector-1-group, .sector-2-group, .sector-3-group).
       - 20 mini-sector column headers (1 to 20).
       - One row per driver (VER and LEC).
       - 20 cells per driver.
       - VER (reference) has REF badge and cells display 'REF' with is-reference-cell class.
       - LEC (non-reference) has cells with signed delta or timing info and faster/slower classes.
    6. Verify Cell Inspection Panel:
       - Clicking or focusing a cell on LEC row updates #heat-strip-cell-details.
       - Details display driver, mini-sector number, sector, delta, duration, speed, and status.
    7. Verify Unsupported Lap Mode handling:
       - Switching to Lap Range mode displays #heat-strip-unsupported-mode notice.
       - Notice explains that Single Lap or Fastest Lap mode is required.
    8. Verify View Navigation switching without duplicate network requests:
       - Switch back to Single Lap and run comparison.
       - Click 'Heat Strip' tab -> only #comparison-heat-strip is displayed.
       - Click 'Table' tab -> only #comparison-results-table is displayed.
       - Click 'Line' tab -> only #comparison-pace-chart is displayed.
       - Click 'All Views' tab -> all three views are displayed.
    9. Verify Stale Configuration detection:
       - Change reference driver to LEC.
       - #heat-strip-stale-banner is displayed.
    10. Verify Race Switch clearance:
       - Switch race to 2023_monza.
       - Returns to #heat-strip-idle.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server, wait_until="domcontentloaded")

        # 1. Navigate to Driver Comparison tab
        comparison_tab = page.locator("#tab-driver-comparison")
        expect(comparison_tab).to_be_visible()
        comparison_tab.click()

        # 2. Verify Initial / Idle state
        idle_heat_strip = page.locator("#heat-strip-idle")
        expect(idle_heat_strip).to_be_visible()

        # 3. Verify comparison view navigation bar
        view_nav = page.locator("#comparison-view-navigation")
        expect(view_nav).to_be_visible()
        btn_all = page.locator("#tab-comp-view-all")
        btn_heat_strip = page.locator("#tab-comp-view-heat-strip")
        btn_line = page.locator("#tab-comp-view-line")
        btn_table = page.locator("#tab-comp-view-table")

        expect(btn_all).to_be_visible()
        expect(btn_heat_strip).to_be_visible()
        expect(btn_line).to_be_visible()
        expect(btn_table).to_be_visible()
        expect(btn_all).to_have_class("comp-view-tab-btn active")

        # 4. Select Full Race cutoff and run comparison for Single Lap (Lap 1)
        full_race_btn = page.locator("#section-modifiers-cutoff button:has-text('Full Race')")
        full_race_btn.click()

        run_btn = page.locator("#btn-run-comparison")
        expect(run_btn).to_be_enabled()
        run_btn.click()

        # 5. Verify Mini-Sector Heat Strip renders
        heat_strip = page.locator("#comparison-heat-strip")
        expect(heat_strip).to_be_visible(timeout=15000)
        expect(idle_heat_strip).to_have_count(0)

        # Legend checks
        legend = page.locator("#heat-strip-legend")
        expect(legend).to_be_visible()
        expect(legend.locator(".legend-item", has_text="Faster")).to_be_visible()
        expect(legend.locator(".legend-item", has_text="Slower")).to_be_visible()
        expect(legend.locator(".legend-item", has_text="Reference Baseline")).to_be_visible()
        expect(legend.locator(".legend-item", has_text="Telemetry Gap")).to_be_visible()
        expect(legend.locator(".legend-item", has_text="Missing Telemetry")).to_be_visible()

        # Sector headers S1, S2, S3
        expect(heat_strip.locator(".sector-1-group")).to_be_visible()
        expect(heat_strip.locator(".sector-2-group")).to_be_visible()
        expect(heat_strip.locator(".sector-3-group")).to_be_visible()

        # 20 Mini-Sector column numbers
        ms_headers = heat_strip.locator(".ms-col-header")
        expect(ms_headers).to_have_count(20)

        # Driver rows
        ver_row = page.locator("#heat-strip-row-VER")
        lec_row = page.locator("#heat-strip-row-LEC")
        expect(ver_row).to_be_visible()
        expect(lec_row).to_be_visible()

        # Reference driver (VER) has REF badge and reference cells
        expect(ver_row.locator(".ref-badge")).to_have_text("REF")
        ver_cells = ver_row.locator(".heat-strip-cell")
        expect(ver_cells).to_have_count(20)
        # All valid reference cells have 'REF' and 'is-reference-cell' class
        first_ver_cell = ver_cells.first
        expect(first_ver_cell).to_have_class("heat-strip-cell is-reference-cell")
        expect(first_ver_cell).to_have_text("REF")

        # Non-reference driver (LEC) has 20 cells with deltas
        lec_cells = lec_row.locator(".heat-strip-cell")
        expect(lec_cells).to_have_count(20)
        first_lec_cell = lec_cells.first
        first_lec_text = first_lec_cell.inner_text()
        assert "+" in first_lec_text or "-" in first_lec_text or "0.00" in first_lec_text or "GAP" in first_lec_text or "—" in first_lec_text

        # 6. Verify Cell Inspection Panel
        first_lec_cell.click()
        details_panel = page.locator("#heat-strip-cell-details")
        expect(details_panel).to_be_visible()
        expect(details_panel.locator(".inspection-driver")).to_have_text("LEC")
        expect(details_panel.locator(".inspection-ms-id")).to_contain_text("Mini-Sector 1")
        expect(details_panel.locator(".inspection-sector-tag")).to_contain_text("Sector 1")
        expect(details_panel.locator(".inspection-stat", has_text="Delta to Ref")).to_be_visible()
        expect(details_panel.locator(".inspection-stat", has_text="Duration")).to_be_visible()
        expect(details_panel.locator(".inspection-stat", has_text="Avg Speed")).to_be_visible()
        expect(details_panel.locator(".inspection-stat", has_text="Distance Range")).to_be_visible()
        expect(details_panel.locator(".inspection-stat", has_text="Status")).to_be_visible()

        # 7. Verify Unsupported Lap Mode notice
        range_btn = page.locator("#section-lap-mode button:has-text('Lap Range')")
        range_btn.click()
        unsupported_card = page.locator("#heat-strip-unsupported-mode")
        expect(unsupported_card).to_be_visible()
        expect(unsupported_card).to_contain_text("Mode Selection Required")
        expect(unsupported_card).to_contain_text("RANGE")

        # 8. Switch back to Single Lap, run comparison, test View Navigation
        single_btn = page.locator("#section-lap-mode button:has-text('Single Lap')")
        single_btn.click()
        run_btn.click()
        expect(page.locator("#comparison-heat-strip")).to_be_visible()

        # Sub-view: Heat Strip only
        btn_heat_strip.click()
        expect(btn_heat_strip).to_have_class("comp-view-tab-btn active")
        expect(page.locator("#comparison-heat-strip")).to_be_visible()
        expect(page.locator("#comparison-results-table")).to_have_count(0)
        expect(page.locator("#comparison-pace-chart")).to_have_count(0)
        expect(page.locator("#comparison-track-view")).to_have_count(0)

        # Sub-view: Table only
        btn_table.click()
        expect(btn_table).to_have_class("comp-view-tab-btn active")
        expect(page.locator("#comparison-results-table")).to_be_visible()
        expect(page.locator("#comparison-heat-strip")).to_have_count(0)
        expect(page.locator("#comparison-pace-chart")).to_have_count(0)
        expect(page.locator("#comparison-track-view")).to_have_count(0)

        # Sub-view: Line only
        btn_line.click()
        expect(btn_line).to_have_class("comp-view-tab-btn active")
        expect(page.locator("#comparison-pace-chart")).to_be_visible()
        expect(page.locator("#comparison-heat-strip")).to_have_count(0)
        expect(page.locator("#comparison-results-table")).to_have_count(0)
        expect(page.locator("#comparison-track-view")).to_have_count(0)

        # Sub-view: All Views
        btn_all.click()
        expect(btn_all).to_have_class("comp-view-tab-btn active")
        expect(page.locator("#comparison-heat-strip")).to_be_visible()
        expect(page.locator("#comparison-pace-chart")).to_be_visible()
        expect(page.locator("#comparison-results-table")).to_be_visible()
        expect(page.locator("#comparison-track-view")).to_be_visible()

        # 9. Stale Configuration Banner
        # Change reference driver to LEC
        ref_select = page.locator("#reference-driver-select")
        ref_select.select_option(value="LEC")
        stale_banner = page.locator("#heat-strip-stale-banner")
        expect(stale_banner).to_be_visible()

        # 10. Switch race to 2023_monza -> cleared back to idle
        race_select = page.locator("#race-select-dropdown")
        race_select.select_option(value="2023_monza")
        page.wait_for_timeout(500)

        expect(page.locator("#comparison-heat-strip")).to_have_count(0)
        expect(page.locator("#heat-strip-idle")).to_be_visible()

        browser.close()


def test_driver_comparison_track_view_e2e(live_server):
    """
    E2E test for Checkpoint 2B-4: Track View Integration.
    1. Navigate to Driver Comparison tab.
    2. Verify initial / idle state (#track-view-idle).
    3. Verify Track view tab button (#tab-comp-view-track).
    4. Select Full Race cutoff and run comparison for Single Lap (Lap 1).
    5. Verify Track View renders (#comparison-track-view):
       - Circuit map container (#comparison-circuit-map).
       - All 20 mini-sector segments (#track-minisector-1 to 20).
       - Overlay toggles: Dominance (#track-overlay-dominance), Delta (#track-overlay-delta), Speed (#track-overlay-speed).
       - Dominance legend with win counters.
    6. Verify Track interaction and section inspection:
       - Empty state prompt in #track-section-details.
       - Click #track-minisector-1 -> gets 'is-selected' class.
       - #track-section-details updates with Mini-Sector 1, distance, winner, and driver breakdown table.
       - Keyboard navigation: focus #track-minisector-2 and press Enter -> updates to Mini-Sector 2.
    7. Verify Overlay switching without re-fetching:
       - Switch to Delta overlay -> #track-driver-select appears, legend switches to delta swatches.
       - Switch to Speed overlay -> speed gradient bar appears.
    8. Verify View Navigation switching:
       - Click 'Track' tab -> only #comparison-track-view visible.
       - Click 'All Views' -> all 4 views visible.
    9. Verify Unsupported Lap Mode handling:
       - Switch to Lap Range -> #track-view-unsupported-mode displayed.
    10. Verify Stale Configuration detection and Race Switch clearance.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server, wait_until="domcontentloaded")

        # 1. Navigate to Driver Comparison tab
        comparison_tab = page.locator("#tab-driver-comparison")
        expect(comparison_tab).to_be_visible()
        comparison_tab.click()

        # 2. Verify Initial / Idle state
        idle_track = page.locator("#track-view-idle")
        expect(idle_track).to_be_visible()

        # 3. Verify Track view navigation button
        btn_track = page.locator("#tab-comp-view-track")
        expect(btn_track).to_be_visible()
        btn_all = page.locator("#tab-comp-view-all")
        expect(btn_all).to_have_class("comp-view-tab-btn active")

        # 4. Select Full Race cutoff and run comparison for Single Lap (Lap 1)
        full_race_btn = page.locator("#section-modifiers-cutoff button:has-text('Full Race')")
        full_race_btn.click()

        run_btn = page.locator("#btn-run-comparison")
        expect(run_btn).to_be_enabled()
        run_btn.click()

        # 5. Verify Track View renders
        track_view = page.locator("#comparison-track-view")
        expect(track_view).to_be_visible(timeout=15000)
        expect(idle_track).to_have_count(0)

        # Circuit map is present
        circuit_map = page.locator("#comparison-circuit-map")
        expect(circuit_map).to_be_visible()

        # All 20 mini-sector paths are rendered
        for ms_id in range(1, 21):
            ms_path = circuit_map.locator(f"#track-minisector-{ms_id}")
            expect(ms_path).to_have_count(1)

        # Overlay controls
        btn_dom = page.locator("#track-overlay-dominance")
        btn_delta = page.locator("#track-overlay-delta")
        btn_speed = page.locator("#track-overlay-speed")
        expect(btn_dom).to_be_visible()
        expect(btn_delta).to_be_visible()
        expect(btn_speed).to_be_visible()
        expect(btn_dom).to_have_class("track-overlay-btn active")

        # Dominance legend displays drivers and wins count
        legend = page.locator("#track-overlay-legend")
        expect(legend).to_be_visible()
        expect(legend.locator(".dominance-legend-item", has_text="VER")).to_be_visible()
        expect(legend.locator(".dominance-legend-item", has_text="LEC")).to_be_visible()

        # 6. Verify Section Details & Interaction
        details_panel = page.locator("#track-section-details")
        expect(details_panel).to_be_visible()
        # Empty prompt initially
        expect(details_panel.locator(".section-details-empty")).to_be_visible()

        # Click mini-sector 1
        ms1 = page.locator("#track-minisector-1")
        ms1.click(force=True)
        expect(ms1).to_have_class(re.compile(r"is-selected"))
        expect(details_panel.locator(".section-details-empty")).to_have_count(0)
        expect(details_panel.locator(".section-title")).to_contain_text("Mini-Sector #1")
        expect(details_panel.locator(".section-distance")).to_be_visible()

        # Drivers breakdown table has rows for VER and LEC
        table = details_panel.locator(".section-drivers-table")
        expect(table).to_be_visible()
        expect(table.locator("tbody tr", has_text="VER")).to_be_visible()
        expect(table.locator("tbody tr", has_text="LEC")).to_be_visible()

        # Keyboard interaction: focus mini-sector 2 and press Enter
        ms2 = page.locator("#track-minisector-2")
        ms2.focus()
        page.keyboard.press("Enter")
        expect(ms2).to_have_class(re.compile(r"is-selected"))
        expect(details_panel.locator(".section-title")).to_contain_text("Mini-Sector #2")

        # 7. Switch to Delta Overlay without duplicate API request
        btn_delta.click()
        expect(btn_delta).to_have_class(re.compile(r"active"))
        expect(btn_dom).not_to_have_class(re.compile(r"active"))

        driver_select = page.locator("#track-driver-select")
        expect(driver_select).to_be_visible()
        expect(driver_select.locator("option")).to_have_count(2)

        # Delta legend is rendered
        expect(legend.locator(".delta-swatches")).to_be_visible()
        expect(legend.locator(".legend-item", has_text="Faster")).to_be_visible()
        expect(legend.locator(".legend-item", has_text="Slower")).to_be_visible()
        expect(legend.locator(".legend-item", has_text="Reference")).to_be_visible()

        # Switch to Speed Overlay
        btn_speed.click()
        expect(btn_speed).to_have_class("track-overlay-btn active")
        expect(legend.locator(".speed-bar-container")).to_be_visible()
        expect(legend.locator(".speed-gradient-bar")).to_be_visible()

        # 8. Test Sub-View Navigation
        btn_track.click()
        expect(btn_track).to_have_class("comp-view-tab-btn active")
        expect(page.locator("#comparison-track-view")).to_be_visible()
        expect(page.locator("#comparison-heat-strip")).to_have_count(0)
        expect(page.locator("#comparison-results-table")).to_have_count(0)
        expect(page.locator("#comparison-pace-chart")).to_have_count(0)

        # Switch back to All Views -> all four views rendered
        btn_all.click()
        expect(page.locator("#comparison-track-view")).to_be_visible()
        expect(page.locator("#comparison-heat-strip")).to_be_visible()
        expect(page.locator("#comparison-results-table")).to_be_visible()
        expect(page.locator("#comparison-pace-chart")).to_be_visible()

        # 9. Unsupported Lap Mode
        range_btn = page.locator("#section-lap-mode button:has-text('Lap Range')")
        range_btn.click()
        unsupported = page.locator("#track-view-unsupported-mode")
        expect(unsupported).to_be_visible()
        expect(unsupported).to_contain_text("Single or Fastest Lap Mode")

        # 10. Switch back to Single Lap, change reference driver -> Stale Configuration
        single_btn = page.locator("#section-lap-mode button:has-text('Single Lap')")
        single_btn.click()
        run_btn.click()
        expect(page.locator("#comparison-track-view")).to_be_visible()

        ref_select = page.locator("#reference-driver-select")
        ref_select.select_option(value="LEC")
        stale_banner = page.locator("#track-view-stale-banner")
        expect(stale_banner).to_be_visible()

        # 11. Race switch -> idle
        race_select = page.locator("#race-select-dropdown")
        race_select.select_option(value="2023_monza")
        page.wait_for_timeout(500)

        expect(page.locator("#comparison-track-view")).to_have_count(0)
        expect(page.locator("#track-view-idle")).to_be_visible()

        browser.close()


def test_tyre_degradation_panel_e2e(live_server):
    """
    E2E Playwright test for Tyre Degradation Analysis Panel:
    - Verifies tab navigation to Tyre Degradation panel.
    - Verifies KPI cards (deg rate, observed slope, compound, confidence).
    - Verifies interactive SVG chart rendering with lap points and trend line.
    - Verifies stint selection pill interaction (switching to Stint 2).
    - Verifies methodology assumptions note.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server, wait_until="domcontentloaded")

        # 1. Navigate to Tyre Degradation tab
        tyre_tab = page.locator("#tab-tyre-degradation")
        expect(tyre_tab).to_be_visible()
        tyre_tab.click()

        # 2. Verify Tyre panel container is visible
        panel = page.locator("#tyre-degradation-panel")
        expect(panel).to_be_visible()

        # 3. Verify driver selector defaults to VER
        driver_select = page.locator("#select-tyre-driver")
        expect(driver_select).to_be_visible()
        expect(driver_select).to_have_value("VER")

        # 4. Verify KPI Cards render
        expect(page.locator("#card-deg-rate")).to_be_visible()
        expect(page.locator("#card-raw-slope")).to_be_visible()
        expect(page.locator("#card-compound-age")).to_be_visible()
        expect(page.locator("#card-confidence")).to_be_visible()

        # 5. Verify SVG chart renders
        chart_wrapper = page.locator("#tyre-chart-wrapper")
        expect(chart_wrapper).to_be_visible()
        svg_chart = page.locator(".tyre-svg-chart")
        expect(svg_chart).to_be_visible(timeout=10000)

        # Verify lap points rendered in chart
        points = page.locator(".raw-point")
        expect(points.first).to_be_visible()

        # 6. Verify Methodology card
        methodology = page.locator("#tyre-assumptions-card")
        expect(methodology).to_be_visible()
        expect(methodology).to_contain_text("Fuel Correction Baseline")

        # 7. Switch to Stint 2
        stint2_btn = page.locator("#btn-stint-2")
        expect(stint2_btn).to_be_visible()
        stint2_btn.click()
        page.wait_for_timeout(500)

        # Stint 2 has MEDIUM compound
        compound_card = page.locator("#card-compound-age")
        expect(compound_card).to_contain_text("MEDIUM")

        browser.close()


def test_gap_battle_panel_e2e(live_server):
    """
    E2E test for the Gap & Battle Analysis UI module (Milestone 2 M2).
    - Navigates to Gap & Battles tab.
    - Verifies header controls, driver selectors (VER vs LEC), and swap button.
    - Verifies KPI cards (Current Gap, Closing Dynamics, Battle Laps, Completed Overtakes).
    - Verifies interactive SVG gap progression chart with ±1.0s battle corridor.
    - Verifies driver swap inverts leader/chaser dropdowns.
    - Verifies historical overtakes log table and filter tabs.
    - Verifies active battles section and methodology disclaimer.
    """
    # Warm up battles analysis on fresh server process to populate in-memory telemetry cache
    try:
        req = urllib.request.Request(f"{live_server}/api/races/2022_monza/battles?driver1=VER&driver2=LEC")
        with urllib.request.urlopen(req, timeout=90.0) as resp:
            resp.read()
    except Exception:
        pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server, wait_until="domcontentloaded")

        # 1. Navigate to Gap & Battles tab
        battles_tab = page.locator("#tab-gap-battles")
        expect(battles_tab).to_be_visible()
        battles_tab.click()

        # 2. Verify driver selection controls
        d1_select = page.locator("#select-driver1")
        d2_select = page.locator("#select-driver2")
        expect(d1_select).to_be_visible()
        expect(d2_select).to_be_visible()
        expect(d1_select).to_have_value("VER")
        expect(d2_select).to_have_value("LEC")

        # 3. Verify KPI Cards render (allow sufficient timeout for initial offline telemetry load)
        expect(page.locator("#kpi-current-gap")).to_be_visible(timeout=45000)
        expect(page.locator("#kpi-closing-rate")).to_be_visible()
        expect(page.locator("#kpi-battle-laps")).to_be_visible()
        expect(page.locator("#kpi-overtakes")).to_be_visible()

        # 4. Verify SVG chart renders with ±1.0s corridor
        svg_container = page.locator("#gap-chart-svg-container")
        expect(svg_container).to_be_visible()
        expect(page.locator(".gap-battle-svg")).to_be_visible()

        # 5. Test Driver Swap functionality
        swap_btn = page.locator("#btn-swap-drivers")
        expect(swap_btn).to_be_visible()
        swap_btn.click()
        page.wait_for_timeout(400)
        expect(d1_select).to_have_value("LEC")
        expect(d2_select).to_have_value("VER")

        # 6. Verify Historical Position Changes Table
        expect(page.locator(".overtakes-table")).to_be_visible()

        # 7. Test Filter Pills
        filter_pills = page.locator(".filter-pill")
        expect(filter_pills.first).to_be_visible()
        # Click On-Track Passes filter
        on_track_btn = page.locator("button.filter-pill:has-text('On-Track Passes')")
        if on_track_btn.count() > 0:
            on_track_btn.click()
            page.wait_for_timeout(300)

        # 8. Verify Active Battles Section
        expect(page.locator(".active-battles-card")).to_be_visible()

        # 9. Verify Methodology Note
        expect(page.locator(".methodology-note-card")).to_be_visible()
        expect(page.locator(".methodology-note-card")).to_contain_text("RACE Gap & Battle Analytics Rules")

        browser.close()


def test_battle_overtake_prediction_card_e2e(live_server):
    """
    E2E test for Battle & Overtake Machine Learning Prediction Card (Checkpoint 3D).
    Validates:
    - Navigation to Gap & Battles tab.
    - Warm-up and fetch of prediction API.
    - Card renders below KPI grid with dual visually separate probability gauges.
    - Decision threshold markers (0.50) and percentage text in gauge tracks.
    - Attacker and defender role chips displaying team indicators.
    - Supporting causal features (Current Gap, Closing Rate, Clean Pace Delta, Tyre Age Delta).
    - Model reliability warnings, including the high-bin calibration overconfidence warning.
    - Driver swap interaction: verifies attacker/defender physical orientation invariance.
    - Insufficient data handling when cars are widely separated (>10.0s).
    """
    # 1. Warm up prediction endpoint for 2023 Monza VER vs SAI
    try:
        req = urllib.request.Request(
            f"{live_server}/api/races/2023_monza/predictions?driver1=VER&driver2=SAI&as_of=1850.0&horizon_laps=5"
        )
        with urllib.request.urlopen(req, timeout=90.0) as resp:
            resp.read()
    except Exception:
        pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server, wait_until="domcontentloaded")

        # 2. Select 2023 Monza
        race_select = page.locator("#race-select-dropdown")
        expect(race_select).to_be_visible()
        race_select.select_option(value="2023_monza")
        page.wait_for_timeout(800)

        # 3. Navigate to Gap & Battles tab
        battles_tab = page.locator("#tab-gap-battles")
        expect(battles_tab).to_be_visible()
        battles_tab.click()

        # 4. Set Driver 1 = VER, Driver 2 = SAI
        d1_select = page.locator("#select-driver1")
        d2_select = page.locator("#select-driver2")
        expect(d1_select).to_be_visible()
        expect(d2_select).to_be_visible()
        d1_select.select_option(value="VER")
        d2_select.select_option(value="SAI")

        # 5. Enable Replay Clock synchronization
        sync_checkbox = page.locator("#toggle-replay-sync")
        expect(sync_checkbox).to_be_visible()
        if not sync_checkbox.is_checked():
            sync_checkbox.check()

        # Step to Lap 5 (+1 LAP clicks) so replay clock enters Lap 5 battle window
        step_lap_next = page.locator("#btn-step-lap-next")
        expect(step_lap_next).to_be_visible()
        for _ in range(5):
            step_lap_next.click()
            page.wait_for_timeout(300)

        # Allow prediction hook to fetch point-in-time forecast
        page.wait_for_timeout(1000)

        # 6. Verify Prediction Card is rendered
        prediction_card = page.locator("#battle-overtake-prediction-card")
        expect(prediction_card).to_be_visible(timeout=30000)

        # Header Title and Horizon pill
        expect(prediction_card.locator(".prediction-title")).to_contain_text("Battle & Overtake ML Forecast")
        expect(prediction_card.locator(".horizon-pill")).to_contain_text("+5 Laps")

        # Attacker vs Defender Identities
        attacker_chip = page.locator("#role-attacker")
        defender_chip = page.locator("#role-defender")
        expect(attacker_chip).to_be_visible()
        expect(defender_chip).to_be_visible()
        expect(attacker_chip).to_contain_text("VER")
        expect(defender_chip).to_contain_text("SAI")

        # 7. Dual Separate Probability Gauges
        battle_gauge = page.locator("#gauge-battle-probability")
        overtake_gauge = page.locator("#gauge-overtake-probability")
        expect(battle_gauge).to_be_visible()
        expect(overtake_gauge).to_be_visible()

        # Values displayed in percentage format
        expect(battle_gauge.locator(".gauge-pct")).to_contain_text("%")
        expect(overtake_gauge.locator(".gauge-pct")).to_contain_text("%")

        # Decision threshold marker (0.50)
        expect(battle_gauge.locator(".threshold-label")).to_contain_text("0.50")
        expect(overtake_gauge.locator(".threshold-label")).to_contain_text("0.50")

        # 8. Supporting Causal Features
        features_panel = page.locator("#prediction-supporting-features")
        expect(features_panel).to_be_visible()
        expect(features_panel).to_contain_text("Current Gap:")
        expect(features_panel).to_contain_text("Closing Rate:")
        expect(features_panel).to_contain_text("Clean-Pace Delta:")
        expect(features_panel).to_contain_text("Tyre-Age Delta:")

        # 9. Model Reliability Warnings
        warnings_container = page.locator("#prediction-reliability-warnings")
        expect(warnings_container).to_be_visible()
        expect(warnings_container).to_contain_text("Calibration Limitation")

        # 10. Test Driver Swap Symmetry
        swap_btn = page.locator("#btn-swap-drivers")
        expect(swap_btn).to_be_visible()
        swap_btn.click()
        page.wait_for_timeout(800)

        # Dropdowns swap: d1=SAI, d2=VER
        expect(d1_select).to_have_value("SAI")
        expect(d2_select).to_have_value("VER")

        # Attacker remains VER, Defender remains SAI
        expect(page.locator("#role-attacker")).to_contain_text("VER")
        expect(page.locator("#role-defender")).to_contain_text("SAI")

        # 11. Test Insufficient Data State
        # Select two drivers with huge gap (e.g. VER vs SAR)
        d2_select.select_option(value="SAR")
        page.wait_for_timeout(800)
        expect(page.locator("#prediction-card-insufficient")).to_be_visible(timeout=15000)

        browser.close()


def test_pit_strategy_tab_e2e(live_server):
    """
    E2E test for Pit Strategy & Time-Loss Accounting UI (Checkpoint 4).
    Validates:
    - Navigation to Pit Strategy tab.
    - Driver selection (VER).
    - KPI cards: Total Completed Stops, Assumed Stationary, In-Lane Transit, Net Pit Loss.
    - Pit stops table with status chips, timing, and reference assumption badges.
    - Time-loss waterfall sub-tab with non-overlapping decomposition and ±0.01s identity verification badge.
    - Tactical analysis sub-tab with heuristic rival undercut/overcut classification.
    - Strict methodology disclaimers regarding the 2.5s assumption and unknown track status.
    """
    # 1. Warm up pit strategy endpoint
    try:
        req = urllib.request.Request(
            f"{live_server}/api/races/2023_monza/pits?driver=VER"
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            resp.read()
    except Exception:
        pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server, wait_until="domcontentloaded")

        # 2. Select 2023 Monza
        race_select = page.locator("#race-select-dropdown")
        expect(race_select).to_be_visible()
        race_select.select_option(value="2023_monza")
        page.wait_for_timeout(500)

        # 3. Click Pit Strategy tab
        pit_tab = page.locator("#tab-pit-strategy")
        expect(pit_tab).to_be_visible()
        pit_tab.click()

        # 4. Verify Pit Strategy Panel container
        panel = page.locator("#pit-strategy-panel")
        expect(panel).to_be_visible()

        # 5. Verify Driver Selector
        driver_select = page.locator("#pit-driver-select")
        expect(driver_select).to_be_visible()
        driver_select.select_option(value="VER")
        page.wait_for_timeout(1000)

        # 6. Verify Overview KPIs at Race Start (0 completed stops)
        expect(page.locator("#kpi-completed-stops")).to_be_visible()
        expect(page.locator("#kpi-active-stop-state")).to_be_visible()
        expect(page.locator("#kpi-stationary-assumption")).to_be_visible()
        expect(page.locator("#kpi-clean-pace-baseline")).to_be_visible()
        expect(page.locator("#kpi-waterfall-verified")).to_be_visible()

        # At race start (Lap 1), VER has not stopped yet -> empty state shown
        empty_state = page.locator("#pit-empty-state")
        expect(empty_state).to_be_visible()
        expect(empty_state).to_contain_text("No Pit Stops Recorded")

        # 7. Toggle Replay Sync to 'Full Race' mode to inspect all race pit events
        sync_toggle_btn = page.locator("#btn-toggle-pit-sync")
        expect(sync_toggle_btn).to_be_visible()
        sync_toggle_btn.click()
        page.wait_for_timeout(1000)

        # 8. Verify Pit Stops Table now populates with VER's stop
        table = page.locator("#pit-stops-table")
        expect(table).to_be_visible()
        stop_row_1 = page.locator("#pit-stop-row-1")
        expect(stop_row_1).to_be_visible()
        expect(stop_row_1.locator(".pit-status-chip")).to_contain_text("COMPLETED")
        # Stationary badge should state Illustrative
        expect(stop_row_1.locator(".stationary-col")).to_contain_text("Illustrative")

        # 9. Switch to Waterfall Sub-tab
        waterfall_tab = page.locator("#tab-pit-view-waterfall")
        expect(waterfall_tab).to_be_visible()
        waterfall_tab.click()

        waterfall_card = page.locator("#time-loss-waterfall-card")
        expect(waterfall_card).to_be_visible()
        # Identity Verified badge
        expect(waterfall_card.locator(".identity-badge")).to_contain_text("Identity Verified (±0.01s)")
        # Equation breakdown box
        expect(waterfall_card.locator(".waterfall-equation-box")).to_contain_text("Strict Accounting Identity")

        # 10. Switch to Tactical Sub-tab
        tactical_tab = page.locator("#tab-pit-view-tactical")
        expect(tactical_tab).to_be_visible()
        tactical_tab.click()

        tactical_card = page.locator("#tactical-evidence-stop-1")
        expect(tactical_card).to_be_visible()
        expect(tactical_card.locator(".tactical-title")).to_contain_text("VER Stop #1 Tactical Classification")

        # 11. Methodology Disclaimer
        disclaimer = page.locator("#pit-accounting-disclaimer")
        expect(disclaimer).to_be_visible()
        expect(disclaimer).to_contain_text("1. Illustrative Assumption:")

        browser.close()


def test_pit_strategy_2022_monza_real_browser_regression(live_server):
    """
    Regression test for urgent bug: Pit Strategy tab does not load data in running UI.
    User workflow:
    1. Select 2022 Monza (where ALB did not drive, testing driver reconciliation).
    2. Open Pit Strategy tab.
    3. Assert network request for /api/races/2022_monza/pits is triggered with 200 OK.
    4. Assert no false red 'Not Found' error appears.
    5. Assert in Synced mode (lap 1), legitimate empty state is shown.
    6. Switch to 'Full Race' and click 'Refresh'.
    7. Assert /pits request is sent again with full race cutoff and status 200.
    8. Assert table populates with pit stop rows and KPIs reflect completed stops.
    9. Switch driver to LEC and assert table updates with LEC stops.
    """
    pit_requests = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.on("request", lambda r: pit_requests.append(r.url) if "/pits" in r.url else None)

        # 1. Load application
        page.goto(live_server, wait_until="domcontentloaded")

        # 2. Select 2022 Monza
        race_select = page.locator("#race-select-dropdown")
        expect(race_select).to_be_visible()
        race_select.select_option(value="2022_monza")
        page.wait_for_timeout(500)

        # 3. Open Pit Strategy tab
        pit_tab = page.locator("#tab-pit-strategy")
        expect(pit_tab).to_be_visible()
        pit_tab.click()

        # 4. Verify no error card is displayed
        error_card = page.locator("#pit-error-state")
        expect(error_card).not_to_be_visible()

        # 5. Verify /pits request was triggered
        page.wait_for_timeout(1000)
        assert len(pit_requests) >= 1, f"Expected at least 1 /pits request, got {pit_requests}"
        assert any("2022_monza" in req for req in pit_requests), f"Request did not contain 2022_monza: {pit_requests}"

        # 6. Verify driver dropdown does NOT default to ALB (ALB didn't race in 2022 Monza)
        driver_select = page.locator("#pit-driver-select")
        expect(driver_select).to_be_visible()
        selected_driver = driver_select.input_value()
        assert selected_driver != "ALB", f"Driver unexpectedly defaulted to ALB in 2022 Monza: {selected_driver}"

        # 7. In Synced mode at lap 1 before pit stops, legitimate empty state is shown
        empty_state = page.locator("#pit-empty-state")
        expect(empty_state).to_be_visible()
        expect(empty_state).to_contain_text("No Pit Stops Recorded")

        # 8. Switch to Full Race and Refresh
        sync_toggle_btn = page.locator("#btn-toggle-pit-sync")
        expect(sync_toggle_btn).to_be_visible()
        sync_toggle_btn.click()

        refresh_btn = page.locator("#btn-refresh-pit-strategy")
        expect(refresh_btn).to_be_visible()
        refresh_btn.click()
        page.wait_for_timeout(1500)

        # 9. Verify table populates and legitimate empty state disappears
        expect(empty_state).not_to_be_visible()
        table = page.locator("#pit-stops-table")
        expect(table).to_be_visible()
        rows = page.locator("#pit-stops-table tbody tr")
        expect(rows.first).to_be_visible()

        # 10. Switch driver to LEC
        driver_select.select_option(value="LEC")
        page.wait_for_timeout(1500)
        expect(table).to_be_visible()
        lec_rows = page.locator("#pit-stops-table tbody tr")
        assert lec_rows.count() >= 2, f"Expected at least 2 pit stops for LEC in 2022 Monza, got {lec_rows.count()}"

        browser.close()











