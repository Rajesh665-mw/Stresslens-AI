"""
Streamlit dashboard for the On-Device Stress Detector.
Command-center redesign — FRONTEND ONLY. Backend calls/contracts unchanged.

Run with (from the project root):
    streamlit run ui/dashboard.py

WHAT'S REAL VS. WHAT'S VISUAL (read this before a demo):
- The live webcam feed + MediaPipe Face Mesh overlay, and the microphone
  waveform/spectrum, run entirely client-side in the browser (getUserMedia +
  MediaPipe/Web Audio API loaded from a CDN). They are genuinely live, not
  fake loops — but they are a separate camera/mic handle from Person A's
  Python capture pipeline. Most OSes only let one process hold a webcam/mic
  at a time, so if Person A's pipeline already has the device open, the
  browser panel shows a graceful "camera/mic unavailable" state instead of
  the preview. Worth testing on the demo machine beforehand.
- Every number on the Physiological / Voice / Stress panels (blink rate,
  tension scores, pitch, stress %, etc.) still comes from the real
  `FeatureData` / `SLMResponse` objects produced by the existing pipeline —
  nothing there is fabricated.
- "Session Analytics" only reports things this frontend can honestly derive
  without touching the backend: reading count, session duration, and time
  between readings (all computed from the real history timestamps), plus
  the host machine's CPU/RAM if `psutil` is installed. There is no true
  camera/audio FPS or per-stage model latency counter in the backend today,
  so rather than invent one, those fields are simply left out.
- Sparklines on the metric cards are built from a short rolling log kept in
  this session's `st.session_state` (not persisted, not sent anywhere) —
  it resets if the tab is refreshed.
"""
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Make the project root importable when Streamlit runs this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_store import store
from backend.pipeline import start_background_pipeline, run_inference, get_latest_features
from utils.config import APP_TITLE, UI_REFRESH_SECONDS, STALE_DATA_SECONDS, POLL_INTERVAL_SECONDS, FRAME_FILE


try:
    import psutil
except ImportError:
    psutil = None

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

start_background_pipeline()

COMPONENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def load_asset(*parts: str) -> str:
    with open(os.path.join(*parts), "r", encoding="utf-8") as f:
        return f.read()


# ============================================================================
# THEME
# ============================================================================
st.markdown(f"<style>{load_asset(ASSETS_DIR, 'theme.css')}</style>", unsafe_allow_html=True)


# ============================================================================
# HELPERS
# ============================================================================
def render_html(html_str: str):
    """Renders HTML safely in Streamlit without markdown line breaks or 4-space code block escaping."""
    if not html_str:
        return
    clean_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    st.markdown("".join(clean_lines), unsafe_allow_html=True)


def safe(v, fmt="{:.2f}", default="—"):
    return fmt.format(v) if isinstance(v, (int, float)) else default


def metric_bar(label, icon, value, color="#38bdf8", max_value=1.0, fmt="{:.0%}"):
    if value is None:
        pct_width, val_label = 0, "—"
    else:
        pct_width = max(0.0, min(1.0, value / max_value)) * 100
        val_label = fmt.format(value)
    return (
        f'<div class="mrow">'
        f'<div class="mrow-top"><span class="mrow-label">{icon} {label}</span><span class="mrow-val">{val_label}</span></div>'
        f'<div class="mbar-track"><div class="mbar-fill" style="width:{pct_width:.1f}%; background-image:linear-gradient(90deg, {color}99, {color}, {color}99);"></div></div>'
        f'</div>'
    )


def stat_chip(label, value, suffix=""):
    val = f"{value}{suffix}" if value is not None else "—"
    return f'<div class="chip"><div class="chip-label">{label}</div><div class="chip-val">{val}</div></div>'


def badge(label, kind="neutral", icon="", pulse=False):
    dot = f'<span class="dot{" dot-pulse" if pulse else ""}"></span>' if icon == "" else ""
    return f'<span class="badge badge-{kind}">{icon}{dot} {label}</span>'


def sparkline_svg(values, color="#38bdf8", w=90, h=26):
    vals = [v for v in values if isinstance(v, (int, float))]
    if len(vals) < 2:
        return f'<svg width="{w}" height="{h}"></svg>'
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    step = w / (len(vals) - 1)
    pts = " ".join(f"{i * step:.1f},{h - ((v - lo) / rng) * (h - 4) - 2:.1f}" for i, v in enumerate(vals))
    last_x, last_y = (len(vals) - 1) * step, h - ((vals[-1] - lo) / rng) * (h - 4) - 2
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.4" fill="{color}"/>'
        f'</svg>'
    )


def trend_arrow(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if len(vals) < 2:
        return '<span class="metric-trend trend-flat">–</span>'
    diff = vals[-1] - vals[-2]
    if abs(diff) < 1e-6:
        return '<span class="metric-trend trend-flat">▶ flat</span>'
    cls = "trend-up" if diff > 0 else "trend-down"
    arrow = "▲" if diff > 0 else "▼"
    return f'<span class="metric-trend {cls}">{arrow} {abs(diff):.2f}</span>'


def metric_card(icon, name, value_label, spark_values, color, unit=""):
    return (
        f'<div class="metric-card">'
        f'<div class="metric-card-top">'
        f'<div>'
        f'<div class="metric-icon">{icon}</div>'
        f'<div class="metric-name">{name}</div>'
        f'<div class="metric-value">{value_label}{unit}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'{sparkline_svg(spark_values, color)}'
        f'<div style="margin-top:4px;">{trend_arrow(spark_values)}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def embed_component(filename: str, height: int, replacements: Optional[dict] = None):
    html_str = load_asset(COMPONENTS_DIR, filename)
    for token, val in (replacements or {}).items():
        html_str = html_str.replace(token, val)
    components.html(html_str, height=height, scrolling=False)



# ============================================================================
# LIVE STATE
# ============================================================================
state = store.get_state()
features = state["features"]
response = state["response"]
last_updated = state["last_updated"]
history = state["history"]

seconds_since_update = (time.time() - last_updated) if last_updated else None
feed_active = seconds_since_update is not None and seconds_since_update < STALE_DATA_SECONDS

# ---- session-local rolling metric log (frontend-only, resets on refresh) ----
if "_metric_log" not in st.session_state:
    st.session_state["_metric_log"] = []
if "_session_start" not in st.session_state:
    st.session_state["_session_start"] = time.time()

is_new_reading = False
if features is not None and response is not None:
    last_ts = st.session_state.get("_last_logged_ts")
    if last_ts != features.timestamp:
        is_new_reading = True
        st.session_state["_last_logged_ts"] = features.timestamp
        st.session_state["_metric_log"].append(
            {
                "timestamp": features.timestamp,
                "stress_score": features.stress_score,
                "stress_level": response.stress_level,
                "blink_rate": features.blink_rate,
                "eye_openness": features.eye_openness,
                "mouth_tension": features.mouth_tension,
                "brow_tension": features.brow_tension,
                "head_movement": features.head_movement,
                "face_confidence": features.face_confidence,
                "voice_pitch": features.voice_pitch,
                "voice_energy": features.voice_energy,
                "speech_rate": features.speech_rate,
                "pitch_variance": features.pitch_variance,
            }
        )
        st.session_state["_metric_log"] = st.session_state["_metric_log"][-200:]

mlog = st.session_state["_metric_log"]


def series(field, n=20):
    return [row[field] for row in mlog[-n:]]


# ============================================================================
# HEADER
# ============================================================================
col_title, col_badges = st.columns([3, 1.4])
with col_title:
    render_html(
        """
        <div class="hero-title">🧠 On-Device Stress Detector</div>
        <div class="hero-sub">Local SLM interpretation of physiological stress signals — no cloud, no data leaves this machine.</div>
        """
    )
with col_badges:
    ts_label = datetime.fromtimestamp(last_updated).strftime("%H:%M:%S") if last_updated else "--:--:--"
    render_html(
        f"""
        <div style="text-align:right; margin-top:6px;">
            {badge("Running Completely Offline", "offline", "🔒")}
            <div style="margin-top:8px; color:#7d8aa3; font-size:0.75rem; font-family:'JetBrains Mono',monospace;">
                LAST SYNC&nbsp;·&nbsp;{ts_label}
            </div>
        </div>
        """
    )

# ---------------- animated pipeline banner ----------------
render_html('<div class="glass" style="padding:4px 6px;">')
embed_component("pipeline_flow.html", height=90, replacements={"__BURST__": "true" if is_new_reading else "false"})
render_html("</div>")

# ============================================================================
# MAIN CONTENT
# ============================================================================
if features is None or response is None:
    render_html(
        """
        <div class="glass" style="text-align:center; padding:40px 20px;">
            <div style="font-size:2rem;">⏳</div>
            <div style="font-weight:700; margin-top:8px;">Waiting for live data…</div>
            <div class="panel-sub" style="margin-top:6px;">
                Make sure Person A's pipeline (or <code>python simulate_person_a.py</code>) is writing to
                <code>data/features.json</code>.
            </div>
        </div>
        """
    )
else:
    # ---------------- HERO ROW: camera | gauge | AI assistant ----------------
    col_cam, col_gauge, col_ai = st.columns([1.25, 0.95, 1.1], gap="medium")

    with col_cam:
        render_html('<div class="glass glass-hero">')
        frame_fresh = os.path.exists(FRAME_FILE) and (time.time() - os.path.getmtime(FRAME_FILE) < 4.0)
        if frame_fresh:
            render_html(
                '<div class="panel-title" style="margin-bottom:8px;"><h4>📷 Live Face Mesh View</h4>'
                f'{badge("Live OpenCV + MediaPipe", "active")}</div>'
            )
            # Read bytes first to avoid PermissionError race with atomic writer
            try:
                with open(FRAME_FILE, "rb") as _f:
                    _frame_bytes = _f.read()
                if _frame_bytes:
                    st.image(_frame_bytes, use_container_width=True)
            except (PermissionError, OSError):
                st.info("📷 Acquiring frame...")
        else:
            embed_component("camera_panel.html", height=330)
        render_html("</div>")


    with col_gauge:
        prev_score = mlog[-2]["stress_score"] if len(mlog) >= 2 else None
        delta_cfg = {"reference": prev_score} if prev_score is not None else None
        gauge_color = {"Low": "#4ade80", "Moderate": "#fbbf24", "High": "#f87171"}.get(response.stress_level, "#38bdf8")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta" if delta_cfg else "gauge+number",
                value=features.stress_score,
                number={"suffix": "%", "font": {"size": 42, "color": "#f1f5f9", "family": "Inter"}},
                delta=delta_cfg,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#7d8aa3", "size": 10}},
                    "bar": {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(74,222,128,0.14)"},
                        {"range": [40, 70], "color": "rgba(251,191,36,0.14)"},
                        {"range": [70, 100], "color": "rgba(248,113,113,0.14)"},
                    ],
                    "threshold": {"line": {"color": "#f1f5f9", "width": 3}, "thickness": 0.8, "value": features.stress_score},
                },
            )
        )
        fig.update_layout(
            height=250, margin=dict(l=16, r=16, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf7", "family": "Inter"},
            transition={"duration": 700, "easing": "cubic-in-out"},
        )

        level_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(response.stress_level, "⚪")
        trend = "—"
        if prev_score is not None:
            diff = features.stress_score - prev_score
            trend = f'{"▲" if diff > 0 else "▼" if diff < 0 else "▶"} {abs(diff):.0f} pts'

        render_html('<div class="glass" style="padding-bottom:8px;">')
        render_html(
            f'<div class="panel-title"><h4>🧭 Stress Gauge</h4>'
            f'<div style="display:flex; gap:6px; align-items:center;">'
            f'{badge("15s Cycle", "neutral", icon="⏱️ ")}'
            f'{badge(response.stress_level, {"Low": "active", "Moderate": "amber", "High": "inactive"}.get(response.stress_level, "neutral"), icon=level_emoji + " ")}</div></div>'
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        render_html(
            f"""
            <div class="chip-grid">
                {stat_chip("Risk Level", response.stress_level)}
                {stat_chip("Trend", trend)}
            </div>
            <div class="chip-grid" style="margin-top:8px;">
                {stat_chip("Model Confidence", response.confidence)}
                {stat_chip("Fusion Confidence", safe(features.confidence, "{:.0%}"))}
            </div>
            """
        )
        render_html("</div>")

    with col_ai:
        render_html('<div class="glass">')
        render_html(f'<div class="panel-title"><h4>🤖 Local AI Assistant</h4>{badge("Phi-3 · Local", "violet")}</div>')

        if response.is_fallback:
            render_html('<div class="badge badge-amber" style="margin-bottom:10px;">⚠️ Fallback response — model output could not be fully parsed</div>')

        reasoning_text = response.reasoning or ""
        prev_reasoning = st.session_state.get("_last_reasoning")
        just_changed = prev_reasoning != reasoning_text
        if just_changed:
            st.session_state["_last_reasoning"] = reasoning_text
            st.session_state["_reasoning_nonce"] = st.session_state.get("_reasoning_nonce", 0) + 1

        render_html('<div class="panel-sub" style="margin-top:0;">🧩 Reasoning</div>')

        safe_json_text = json.dumps(reasoning_text)
        if just_changed:
            components.html(
                f"""
                <div id="scan" style="font:600 0.78rem 'JetBrains Mono',monospace; color:#7dd3fc; margin-bottom:6px; min-height:16px;"></div>
                <div id="tw" style="font-family:'Inter',sans-serif; font-size:0.88rem; line-height:1.6; color:#dbeafe; min-height:70px;"></div>
                <script>
                const text = {safe_json_text};
                const scanEl = document.getElementById("scan");
                const el = document.getElementById("tw");
                const steps = ["🔎 Scanning facial micro-expressions…", "🎧 Analyzing speech…", "🧠 Running Phi-3 inference…", "✍️ Generating reasoning…"];
                let s = 0;
                function showStep() {{
                    if (s < steps.length) {{
                        scanEl.textContent = steps[s];
                        s++;
                        setTimeout(showStep, 260);
                    }} else {{
                        scanEl.textContent = "";
                        typeText();
                    }}
                }}
                let i = 0;
                function typeText() {{
                    if (i <= text.length) {{
                        el.innerHTML = text.slice(0, i).replace(/\\n/g, "<br>") + '<span class="tw-cursor"></span>';
                        i++;
                        setTimeout(typeText, 14);
                    }} else {{
                        el.innerHTML = text.replace(/\\n/g, "<br>");
                    }}
                }}
                showStep();
                </script>
                """,
                height=120,
            )
        else:
            render_html(f'<div style="font-size:0.88rem; line-height:1.6; color:#dbeafe;">{json.loads(safe_json_text)}</div>')

        render_html('<div class="panel-sub">💡 Suggested Intervention</div>')
        render_html(f'<div class="intervention-card">{response.intervention}</div>')

        render_html("<div style='height:12px;'></div>")
        if st.button("🔄  Generate New Intervention", use_container_width=True):
            latest = get_latest_features()
            if latest:
                with st.status("AI is thinking…", expanded=True) as status:
                    st.write("🔎 Scanning facial micro-expressions…")
                    time.sleep(0.3)
                    st.write("🎧 Analyzing speech…")
                    time.sleep(0.2)
                    st.write("🧠 Running Phi-3 inference… ██████████")
                    run_inference(latest)
                    st.write("✍️ Generating reasoning…")
                    time.sleep(0.15)
                    status.update(label="Done", state="complete", expanded=False)
                st.rerun()
            else:
                st.error("No feature reading available yet to re-run.")
        render_html("</div>")

    # ---------------- SECOND ROW: voice | physiological | session analytics ----------------
    col_voice, col_phys, col_session = st.columns([1, 1.3, 0.9], gap="medium")

    with col_voice:
        render_html('<div class="glass">')
        speech_state = "Silence" if features.silence_detected else "Speaking"
        speech_kind = "neutral" if features.silence_detected else "active"
        render_html(
            f'<div class="panel-title"><h4>🎙️ Voice Analysis</h4>{badge(speech_state, speech_kind, pulse=not features.silence_detected)}</div>'
        )
        embed_component("audio_panel.html", height=180)
        render_html(
            f"""
            {metric_bar("Voice Energy", "🔊", features.voice_energy, "#a78bfa")}
            <div class="chip-grid">
                {stat_chip("Pitch", safe(features.voice_pitch, "{:.0f}"), " Hz")}
                {stat_chip("Speech Rate", safe(features.speech_rate, "{:.1f}"))}
            </div>
            <div class="chip-grid" style="margin-top:8px;">
                {stat_chip("Pitch Variance", safe(features.pitch_variance, "{:.2f}"))}
                {stat_chip("Fusion Confidence", safe(features.confidence, "{:.0%}"))}
            </div>
            """
        )
        render_html("</div>")

    with col_phys:
        render_html('<div class="glass">')
        render_html(f'<div class="panel-title"><h4>🧬 Physiological Metrics</h4>{badge("MediaPipe features", "neutral")}</div>')
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            render_html(metric_card("👁️", "Eye Openness", safe(features.eye_openness, "{:.0%}"), series("eye_openness"), "#38bdf8"))
            render_html(metric_card("🤨", "Brow Tension", safe(features.brow_tension, "{:.0%}"), series("brow_tension"), "#fbbf24"))
        with r1c2:
            render_html(metric_card("😬", "Mouth Tension", safe(features.mouth_tension, "{:.0%}"), series("mouth_tension"), "#f472b6"))
            render_html(metric_card("🎯", "Face Confidence", safe(features.face_confidence, "{:.0%}"), series("face_confidence"), "#2dd4bf"))
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            render_html(metric_card("👀", "Blink Rate", safe(features.blink_rate, "{:.1f}"), series("blink_rate"), "#a78bfa", " /min"))
        with r3c2:
            render_html(metric_card("🧍", "Head Movement", safe(features.head_movement, "{:.2f}"), series("head_movement"), "#fb7185"))
        render_html("</div>")

    with col_session:
        render_html('<div class="glass">')
        render_html('<div class="panel-title"><h4>📊 Session Analytics</h4></div>')
        session_seconds = time.time() - st.session_state["_session_start"]
        mins, secs = divmod(int(session_seconds), 60)
        avg_interval = "—"
        if len(mlog) >= 2:
            try:
                t0 = pd.to_datetime(mlog[0]["timestamp"])
                t1 = pd.to_datetime(mlog[-1]["timestamp"])
                span = (t1 - t0).total_seconds()
                if span > 0:
                    avg_interval = f"{span / (len(mlog) - 1):.1f}s"
            except Exception:
                avg_interval = "—"

        cpu_label, ram_label = "—", "—"
        if psutil is not None:
            try:
                cpu_label = f"{psutil.cpu_percent(interval=0.05):.0f}%"
                ram_label = f"{psutil.virtual_memory().percent:.0f}%"
            except Exception:
                pass

        render_html(
            f"""
            <div class="chip-grid">
                {stat_chip("Readings Logged", len(mlog))}
                {stat_chip("Session Time", f"{mins}m {secs}s")}
            </div>
            <div class="chip-grid" style="margin-top:8px;">
                {stat_chip("Avg Reading Interval", avg_interval)}
                {stat_chip("Poll Interval (cfg)", f"{POLL_INTERVAL_SECONDS}s")}
            </div>
            <div class="chip-grid" style="margin-top:8px;">
                {stat_chip("Host CPU", cpu_label)}
                {stat_chip("Host RAM", ram_label)}
            </div>
            <div class="panel-sub" style="margin-top:12px; margin-bottom:0;">
                No per-stage FPS/latency counters exist in the current pipeline, so they're
                left out here rather than guessed.
            </div>
            """
        )
        render_html("</div>")

    # ---------------- TIMELINE ----------------
    render_html('<div class="glass">')
    render_html(f'<div class="panel-title"><h4>📈 Timeline</h4>{badge(f"{len(mlog)} points this session", "neutral")}</div>')

    metric_options = {
        "Stress Score": ("stress_score", "#38bdf8", False),
        "Blink Rate": ("blink_rate", "#a78bfa", True),
        "Speech Rate": ("speech_rate", "#f472b6", True),
        "Voice Energy": ("voice_energy", "#2dd4bf", True),
        "Mouth Tension": ("mouth_tension", "#fb7185", True),
        "Head Movement": ("head_movement", "#fbbf24", True),
        "Face Confidence": ("face_confidence", "#67e8f9", True),
    }
    chosen = st.multiselect(
        "Metrics to plot", list(metric_options.keys()), default=["Stress Score"],
        label_visibility="collapsed",
    )

    if len(mlog) >= 2 and chosen:
        df = pd.DataFrame(mlog)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        tfig = go.Figure()
        for label in chosen:
            field, color, normalize = metric_options[label]
            if field not in df.columns:
                continue
            y = df[field].astype(float)
            if normalize:
                lo, hi = y.min(), y.max()
                y_plot = ((y - lo) / (hi - lo) * 100) if hi > lo else y * 0
                hover = label + " (normalized): %{customdata:.2f}<extra></extra>"
                customdata = df[field]
            else:
                y_plot = y
                hover = label + ": %{y:.0f}%<extra></extra>"
                customdata = None
            tfig.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=y_plot, mode="lines+markers", name=label,
                    line=dict(color=color, width=2.4, shape="spline"),
                    marker=dict(size=5),
                    customdata=customdata,
                    hovertemplate=hover,
                )
            )
        tfig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#a7b3c6", "family": "Inter", "size": 11},
            xaxis=dict(showgrid=False, showline=False, zeroline=False),
            yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                       title="Value (metrics other than Stress Score are normalized 0–100 for comparison)"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
            transition={"duration": 400},
        )
        st.plotly_chart(tfig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    else:
        render_html('<div class="panel-sub">Not enough session data yet — this builds up as new readings arrive.</div>')
    render_html("</div>")

    with st.expander("🔧 Raw feature snapshot (debug)"):
        st.json(features.__dict__)

# ---------------- auto refresh loop ----------------
time.sleep(UI_REFRESH_SECONDS)
st.rerun()
