import os
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from pathlib import Path
from ml.predict import predict_stress
from ml.anomaly_detector import detect_anomaly


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

DATA_PATH = BASE_DIR / "data" / "processed" / "stress_features.csv"
TEST_PATH = BASE_DIR / "results" / "test_predictions.csv"

st.set_page_config(
    page_title="StressSense AI",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# THEME
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #DCEDEA;
    color: #18313D;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #F4F8F8;
    border-right: 1px solid #C7DEDB;
}

h1, h2, h3 {
    color: #18313D !important;
    font-weight: 780 !important;
}

div[data-testid="stMetric"] {
    background: #F9FCFB;
    border: 1px solid #C7DEDB;
    border-radius: 15px;
    padding: 1rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #F9FCFB;
    border: 1px solid #C7DEDB;
    border-radius: 15px;
}

.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 650;
}

.alertbox {
    background: #FFF0F0;
    border: 3px solid #D92D20;
    border-radius: 18px;
    padding: 22px;
    color: #7A271A;
}

.alerttitle {
    font-size: 27px;
    font-weight: 850;
    color: #B42318;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — ONLY TWO PAGES
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 StressSense AI")
    st.caption("Wearable Stress Intelligence")

    page = st.radio(
        "Navigation",
        ["Model Results", "Live Monitoring"],
    )

    st.divider()
    st.caption("PyTorch · FastAPI · MQTT · Streamlit")


# ============================================================
# HELPERS
# ============================================================

def api_get(path):
    try:
        response = requests.get(
            f"{API_BASE_URL}{path}",
            timeout=3,
        )

        if response.ok:
            return response.json()

        return None

    except requests.RequestException:
        return None


def read_csv(path):
    try:
        if path.exists():
            return pd.read_csv(path)

        return None

    except Exception:
        return None


def find_column(df, candidates):
    if df is None:
        return None

    lookup = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def percent(value):
    try:
        value = float(value)

        if 0 <= value <= 1:
            return value * 100

        return value

    except (TypeError, ValueError):
        return 0.0


def show_alarm(title, detail=""):
    st.markdown(
        f"""
        <div class="alertbox">
            <div class="alerttitle">🚨 {title}</div>
            <div style="font-size:17px;margin-top:10px">
                {detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MANUAL PREDICTION — EIGHT INPUTS
# ============================================================

def build_manual_features(
    heart_rate,
    eda,
    temperature,
    ibi,
    rmssd,
    acc_x,
    acc_y,
    acc_z,
):
    # The trained model expects 20 features.
    # Window statistics below are estimates because the form
    # receives individual measurements, not a sensor time window.

    hr_std = 5.0
    eda_std = 0.3
    temp_std = 0.3
    ibi_std = 0.08

    acceleration_magnitude = (
        acc_x**2 + acc_y**2 + acc_z**2
    ) ** 0.5

    return {
        "hr_mean": heart_rate,
        "hr_std": hr_std,
        "hr_min": max(0.0, heart_rate - hr_std),
        "hr_max": heart_rate + hr_std,

        "eda_mean": eda,
        "eda_std": eda_std,
        "eda_min": max(0.0, eda - eda_std),
        "eda_max": eda + eda_std,

        "temp_mean": temperature,
        "temp_std": temp_std,
        "temp_min": temperature - temp_std,
        "temp_max": temperature + temp_std,

        "ibi_mean": ibi,
        "ibi_std": ibi_std,
        "rmssd": rmssd,

        "acc_x_mean": acc_x,
        "acc_y_mean": acc_y,
        "acc_z_mean": acc_z,

        "acc_magnitude_mean": acceleration_magnitude,
        "acc_magnitude_std": 0.15,
    }


def manual_prediction():
    st.title("Manual Stress Prediction")

    st.caption(
        "Enter eight main wearable measurements to estimate "
        "stress and check configured anomaly thresholds."
    )

    with st.container(border=True):

        st.subheader("1. Cardiovascular Signals")

        c1, c2, c3, c4 = st.columns(4)

        heart_rate = c1.number_input(
            "Heart Rate (BPM)",
            min_value=30.0,
            max_value=220.0,
            value=75.0,
            step=1.0,
        )

        ibi = c2.number_input(
            "IBI (seconds)",
            min_value=0.2,
            max_value=3.0,
            value=0.85,
            step=0.01,
        )

        rmssd = c3.number_input(
            "RMSSD / HRV (seconds)",
            min_value=0.001,
            max_value=2.0,
            value=0.12,
            step=0.01,
            format="%.3f",
        )

        eda = c4.number_input(
            "EDA (µS)",
            min_value=0.0,
            max_value=30.0,
            value=1.2,
            step=0.1,
        )

        st.subheader("2. Temperature and Movement")

        c1, c2, c3, c4 = st.columns(4)

        temperature = c1.number_input(
            "Skin Temperature (°C)",
            min_value=20.0,
            max_value=45.0,
            value=33.5,
            step=0.1,
        )

        acc_x = c2.number_input(
            "Acceleration X",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
        )

        acc_y = c3.number_input(
            "Acceleration Y",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
        )

        acc_z = c4.number_input(
            "Acceleration Z",
            min_value=-20.0,
            max_value=20.0,
            value=1.0,
            step=0.1,
        )

        submitted = st.button(
            "🧠 Predict Stress Level",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            features = build_manual_features(
                heart_rate,
                eda,
                temperature,
                ibi,
                rmssd,
                acc_x,
                acc_y,
                acc_z,
            )

            stress_result = predict_stress(features)
            anomaly_result = detect_anomaly(features)

            result = {
                **stress_result,
                **anomaly_result,
            }

            st.subheader("Prediction Result")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Prediction",
                str(result.get("prediction", "Unknown")).upper(),
            )

            c2.metric(
                "Confidence",
                f"{percent(result.get('confidence')):.1f}%",
            )

            c3.metric(
                "Stress Probability",
                f"{percent(result.get('stress_probability')):.1f}%",
            )

            c4.metric(
                "Rest Probability",
                f"{percent(result.get('rest_probability')):.1f}%",
            )

            anomaly = str(
                result.get("anomaly_status", "NORMAL")
            ).upper()

            risk = str(
                result.get("risk_level", "LOW")
            ).upper()

            alerts = result.get("triggered_alerts") or []

            if anomaly != "NORMAL" or risk not in ("LOW", "NORMAL"):

                show_alarm(
                    "PHYSIOLOGICAL ANOMALY DETECTED",
                    f"Status: {anomaly} | Risk Level: {risk}",
                )

                for alert in alerts:
                    st.error(f"⚠️ {alert}")

            else:
                st.success(
                    "No configured physiological anomaly detected."
                )

            st.caption(
                "Educational output only, not a medical diagnosis. "
                "Features estimated from individual readings may differ "
                "from real window-based sensor features."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ============================================================
# TRAINED MODEL RESULTS
# ============================================================

def model_results():
    # Manual prediction intentionally appears first.
    manual_prediction()

    st.divider()

    st.title("Trained Model Results")

    st.caption(
        "These metrics are calculated from the saved held-out "
        "participant predictions, not training accuracy."
    )

    test_df = read_csv(TEST_PATH)

    if test_df is None:
        st.warning(
            "Test predictions not found. Run `python -m ml.train` first."
        )
        return

    actual_col = find_column(
        test_df,
        ["actual_label", "actual", "actual_class", "y_true"],
    )

    predicted_col = find_column(
        test_df,
        ["predicted_label", "predicted", "predicted_class", "y_pred"],
    )

    if actual_col is None or predicted_col is None:
        st.warning(
            "The prediction CSV is missing actual_label or "
            "predicted_label. Run training again."
        )
        return

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    actual = pd.to_numeric(
        test_df[actual_col],
        errors="coerce",
    )

    predicted = pd.to_numeric(
        test_df[predicted_col],
        errors="coerce",
    )

    valid = actual.notna() & predicted.notna()

    actual = actual[valid].astype(int)
    predicted = predicted[valid].astype(int)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Held-out Accuracy",
        f"{accuracy_score(actual, predicted) * 100:.2f}%",
    )

    c2.metric(
        "Precision",
        f"{precision_score(actual, predicted, zero_division=0) * 100:.2f}%",
    )

    c3.metric(
        "Recall",
        f"{recall_score(actual, predicted, zero_division=0) * 100:.2f}%",
    )

    c4.metric(
        "F1 Score",
        f"{f1_score(actual, predicted, zero_division=0) * 100:.2f}%",
    )

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted Rest", "Predicted Stress"],
            y=["Actual Rest", "Actual Stress"],
            text=cm,
            texttemplate="%{text}",
            colorscale=[
                [0, "#EAF5F3"],
                [0.5, "#8FC6C1"],
                [1, "#287C9E"],
            ],
            showscale=False,
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        height=350,
        paper_bgcolor="#F9FCFB",
        plot_bgcolor="#F9FCFB",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    dataset = read_csv(DATA_PATH)

    if dataset is not None:

        st.subheader("Dataset Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric("Samples", len(dataset))

        subject_col = find_column(
            dataset,
            ["subject_id", "subject", "participant"],
        )

        c2.metric(
            "Participants",
            dataset[subject_col].nunique()
            if subject_col else "—",
        )

        label_col = find_column(
            dataset,
            ["stress_label", "stress_level", "label", "target"],
        )

        if label_col:

            labels = dataset[label_col]

            if pd.api.types.is_numeric_dtype(labels):
                count = int(
                    (pd.to_numeric(labels, errors="coerce") == 1).sum()
                )
            else:
                count = int(
                    labels.astype(str)
                    .str.lower()
                    .str.contains("stress")
                    .sum()
                )

            c3.metric("Stress-labelled Samples", count)

    st.info(
        "The held-out participants are excluded from model selection. "
        "An accuracy above 80% is a goal to test, not a guaranteed result."
    )


# ============================================================
# LIVE MONITORING — AUTO REFRESH EVERY 3 SECONDS
# ============================================================

@st.fragment(run_every=3)
def live_monitoring():

    st.title("Live Monitoring")

    st.caption(
        "Automatically refreshes every 3 seconds while this page is open."
    )

    health = api_get("/health")

    if not health:
        st.error(
            "API is offline. Start it with "
            "`uvicorn api.main:app --reload`."
        )
        return

    st.success("● API connected · Automatic refresh active")

    latest = api_get("/latest")
    response = api_get("/history?limit=100")

    if not latest:
        st.info(
            "Waiting for the first prediction from the MQTT stream."
        )
        return

    history = (response or {}).get("data", [])
    hist = pd.DataFrame(history)

    stress_probability = percent(
        latest.get("stress_probability", 0)
    )

    rest_probability = percent(
        latest.get("rest_probability", 0)
    )

    confidence = percent(
        latest.get("confidence", 0)
    )

    anomaly = str(
        latest.get("anomaly_status") or "NORMAL"
    )

    risk = str(
        latest.get("risk_level") or "LOW"
    )

    raw_alerts = latest.get("triggered_alerts") or ""

    if isinstance(raw_alerts, str):
        alerts = [
            item.strip()
            for item in raw_alerts.split("|")
            if item.strip()
        ]
    else:
        alerts = raw_alerts

    abnormal = (
        anomaly.upper() != "NORMAL"
        or risk.upper() not in ("LOW", "NORMAL")
    )

    if abnormal:

        record_id = latest.get("id")

        if (
            record_id is not None
            and st.session_state.get("last_alert_id") != record_id
        ):
            st.session_state["last_alert_id"] = record_id
            st.toast("🚨 ANOMALY DETECTED", icon="🚨")

        show_alarm(
            "ANOMALY DETECTED",
            f"Status: {anomaly} | Risk Level: {risk}",
        )

        for alert in alerts:
            st.error(f"⚠️ {alert}")

    else:
        st.success(
            "✓ No configured physiological anomaly in the latest reading."
        )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Prediction",
        str(latest.get("prediction", "Unknown")).upper(),
    )

    c2.metric("Confidence", f"{confidence:.1f}%")
    c3.metric("Stress Probability", f"{stress_probability:.1f}%")
    c4.metric("Rest Probability", f"{rest_probability:.1f}%")

    if hist.empty:
        st.info("Charts will populate after multiple predictions are saved.")
        return

    # --------------------------------------------------------
    # TIME AXIS
    # --------------------------------------------------------

    time_col = (
        "timestamp"
        if "timestamp" in hist.columns
        and hist["timestamp"].notna().any()
        else "created_at"
    )

    if time_col in hist.columns:
        hist["_time"] = pd.to_datetime(
            hist[time_col],
            errors="coerce",
        )
    else:
        hist["_time"] = pd.NaT

    hist = hist.sort_values(
        "_time",
        na_position="last",
    )

    if hist["_time"].notna().any():
        x_values = hist["_time"]
    else:
        x_values = list(range(len(hist)))

    # --------------------------------------------------------
    # STRESS PROBABILITY GRAPH
    # --------------------------------------------------------

    if "stress_probability" in hist.columns:

        hist["_stress_pct"] = pd.to_numeric(
            hist["stress_probability"],
            errors="coerce",
        ).map(percent)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=hist["_stress_pct"],
                mode="lines+markers",
                name="Stress Probability",
                line=dict(
                    color="#287C9E",
                    width=3,
                ),
                fill="tozeroy",
            )
        )

        fig.add_hline(
            y=50,
            line_dash="dash",
            annotation_text="50% reference",
        )

        fig.update_yaxes(
            title="Stress Probability (%)",
            range=[0, 100],
        )

        fig.update_layout(
            title="Stress Probability Over Time",
            height=350,
            paper_bgcolor="#F9FCFB",
            plot_bgcolor="#F9FCFB",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # PREDICTION TIMELINE
    # --------------------------------------------------------

    if "prediction" in hist.columns:

        hist["_stress_flag"] = (
            hist["prediction"]
            .astype(str)
            .str.lower()
            .eq("stress")
            .astype(int)
        )

        bar_colors = [
            "#D92D20" if value == 1 else "#287C9E"
            for value in hist["_stress_flag"]
        ]

        fig2 = go.Figure(
            go.Bar(
                x=x_values,
                y=hist["_stress_flag"],
                marker_color=bar_colors,
                name="Prediction",
            )
        )

        fig2.update_yaxes(
            title="Predicted State",
            range=[0, 1.2],
            tickvals=[0, 1],
            ticktext=["Rest", "Stress"],
        )

        fig2.update_layout(
            title="Prediction Timeline",
            height=280,
            paper_bgcolor="#F9FCFB",
            plot_bgcolor="#F9FCFB",
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # HISTORY TABLE
    # --------------------------------------------------------

    st.subheader("Recent Monitoring History")

    display_columns = [
        column
        for column in [
            "created_at",
            "timestamp",
            "device_id",
            "prediction",
            "confidence",
            "stress_probability",
            "anomaly_status",
            "risk_level",
        ]
        if column in hist.columns
    ]

    if display_columns:
        display_df = hist[display_columns].copy()

        st.dataframe(
            display_df.iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # ANOMALY THRESHOLDS
    # --------------------------------------------------------

    st.subheader("Configured Anomaly Thresholds")

    thresholds = pd.DataFrame([
        {
            "Signal": "Heart Rate",
            "Configured Range": "45–110 BPM",
        },
        {
            "Signal": "EDA",
            "Configured Range": "0–20 µS",
        },
        {
            "Signal": "Skin Temperature",
            "Configured Range": "25–40 °C",
        },
        {
            "Signal": "IBI",
            "Configured Range": "0.30–2.00 seconds",
        },
        {
            "Signal": "RMSSD",
            "Configured Range": "0.005–2.00 seconds",
        },
    ])

    st.dataframe(
        thresholds,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Latest reading: {latest.get('timestamp') or latest.get('created_at') or 'Unavailable'} "
        f"| Device: {latest.get('device_id') or 'Unknown'}"
    )

    st.caption(
        "Anomaly alerts use configured project thresholds and are not medical diagnoses."
    )


# ============================================================
# PAGE ROUTING
# ============================================================

if page == "Model Results":
    model_results()
else:
    live_monitoring()