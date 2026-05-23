import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st


API_URL = "http://localhost:8000"
ALERT_LOG_PATH = Path("logs") / "alerts.log"

DEFAULT_SENSOR_VALUES = {
    "setting1": 0.0,
    "setting2": 0.0,
    "T24": 642.0,
    "T30": 1588.0,
    "P15": 21.6,
    "Nf": 2388.0,
    "Nc": 9050.0,
    "NRf": 2388.0,
    "NRc": 8130.0,
    "BPR": 8.4,
    "htBleed": 392.0,
    "W31": 39.0,
}


st.set_page_config(
    page_title="Jet Engine Predictive Maintenance",
    page_icon="🛩️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    .subtitle {
        color: #5b6472;
        font-size: 1.05rem;
        margin-top: -0.6rem;
        margin-bottom: 1.25rem;
    }
    .result-box {
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        margin: 0.5rem 0 1rem 0;
        color: white;
    }
    .safe-box {
        background: #0f8a4b;
        border: 1px solid #0b6f3b;
    }
    .danger-box {
        background: #c62828;
        border: 1px solid #9f1f1f;
        animation: pulse-danger 1.2s infinite;
    }
    .result-box h2 {
        margin: 0 0 0.35rem 0;
        letter-spacing: 0;
    }
    .result-box p {
        margin: 0.15rem 0;
        font-size: 1rem;
    }
    @keyframes pulse-danger {
        0% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0.45); }
        70% { box-shadow: 0 0 0 12px rgba(198, 40, 40, 0); }
        100% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "reading_counts" not in st.session_state:
    st.session_state.reading_counts = {}

if "last_response" not in st.session_state:
    st.session_state.last_response = None


def call_prediction_api(engine_id: int, sensor_readings: dict[str, float]) -> tuple[int, dict]:
    payload = json.dumps(
        {
            "engine_id": engine_id,
            "sensor_readings": sensor_readings,
        }
    ).encode("utf-8")

    request = Request(
        f"{API_URL}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))
    except URLError as exc:
        return 503, {"detail": f"Could not connect to FastAPI server: {exc.reason}"}
    except TimeoutError:
        return 504, {"detail": "FastAPI server timed out."}


def parse_alert_log() -> pd.DataFrame:
    if not ALERT_LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "engine_id", "confidence"])

    rows = []
    for line in ALERT_LOG_PATH.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3:
            continue

        timestamp = parts[0]
        engine_id = parts[1].replace("engine_id=", "")
        confidence = parts[2].replace("confidence=", "")
        rows.append(
            {
                "timestamp": timestamp,
                "engine_id": int(engine_id),
                "confidence": float(confidence),
            }
        )

    return pd.DataFrame(rows, columns=["timestamp", "engine_id", "confidence"])


st.title("🛩️ Jet Engine Predictive Maintenance")
st.markdown(
    '<div class="subtitle">Real-time failure detection powered by NASA C-MAPSS data</div>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Sensor Input")
    engine_id = st.number_input("engine_id", min_value=1, max_value=100, value=20, step=1)

    ps30 = st.slider("Ps30 (Static Pressure)", 0.0, 100.0, 47.5, 0.1)
    t50 = st.slider("T50 (Outlet Temp)", 0.0, 1000.0, 900.0, 1.0)
    phi = st.slider("Phi", 0.0, 100.0, 52.0, 0.1)
    w32 = st.slider("W32", 0.0, 100.0, 23.3, 0.1)
    p30 = st.slider("P30", 0.0, 1000.0, 554.0, 0.1)

    current_count = st.session_state.reading_counts.get(int(engine_id), 0)
    st.metric("Readings sent", current_count)

    run_prediction = st.button("🔍 Run Prediction", type="primary", use_container_width=True)


sensor_readings = {
    **DEFAULT_SENSOR_VALUES,
    "Ps30": ps30,
    "T50": t50,
    "Phi": phi,
    "W32": w32,
    "P30": p30,
}

if run_prediction:
    engine_key = int(engine_id)
    status_code, response_body = call_prediction_api(engine_key, sensor_readings)
    st.session_state.reading_counts[engine_key] = (
        st.session_state.reading_counts.get(engine_key, 0) + 1
    )
    st.session_state.last_response = {
        "status_code": status_code,
        "body": response_body,
        "received_at": time.strftime("%H:%M:%S"),
    }


left_col, right_col = st.columns([1.35, 1])

with left_col:
    st.subheader("Prediction Result")
    st.info("Note: Send 10 readings per engine to warm up the rolling window")

    last_response = st.session_state.last_response
    if last_response is None:
        st.success("Ready for prediction. Send sensor readings from the sidebar.")
    else:
        body = last_response["body"]
        status_code = last_response["status_code"]

        if status_code == 200:
            confidence_pct = body["confidence"] * 100

            if body["prediction"] == "SAFE":
                st.markdown(
                    f"""
                    <div class="result-box safe-box">
                        <h2>✅ ENGINE SAFE</h2>
                        <p>Confidence: {confidence_pct:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.success("No alert triggered for this reading.")
            else:
                alert_message = body.get("alert", {}).get("message", "")
                st.markdown(
                    f"""
                    <div class="result-box danger-box">
                        <h2>🚨 DANGER DETECTED</h2>
                        <p>Confidence: {confidence_pct:.2f}%</p>
                        <p>{alert_message}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.error(alert_message)

            metric_cols = st.columns(3)
            metric_cols[0].metric("Engine", body["engine_id"])
            metric_cols[1].metric("Prediction", body["prediction"])
            metric_cols[2].metric("Confidence", f"{confidence_pct:.2f}%")
        else:
            detail = body.get("detail", "Prediction request failed.")
            st.error(detail)

with right_col:
    st.subheader("Current Sensor Snapshot")
    snapshot_cols = st.columns(2)
    snapshot_cols[0].metric("Ps30", f"{ps30:.1f}")
    snapshot_cols[1].metric("T50", f"{t50:.1f}")
    snapshot_cols[0].metric("Phi", f"{phi:.1f}")
    snapshot_cols[1].metric("W32", f"{w32:.1f}")
    snapshot_cols[0].metric("P30", f"{p30:.1f}")


st.divider()
st.subheader("Alert Log Viewer")
if st.button("🔄 Refresh Alerts"):
    pass

alerts_df = parse_alert_log()

if alerts_df.empty:
    st.success("No alerts triggered yet ✅")
else:
    recent_alerts = alerts_df.tail(10).sort_index(ascending=False).reset_index(drop=True)
    recent_alerts["confidence"] = recent_alerts["confidence"].map(
        lambda value: f"{value * 100:.2f}%"
    )
    st.dataframe(recent_alerts, use_container_width=True, hide_index=True)
