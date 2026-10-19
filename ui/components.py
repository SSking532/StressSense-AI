import streamlit as st


def render_header(device_id, timestamp):

    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("🧠 StressSense AI")
        st.caption(
            "Real-time wearable stress monitoring "
            "powered by IoT, PyTorch & FastAPI"
        )

    with col2:
        st.success("● SYSTEM LIVE")
        st.caption(f"Device: {device_id}")
        st.caption(f"Last sync: {timestamp}")


def render_stress_card(stress_level, confidence):

    level = stress_level.lower()

    if level == "low":

        icon = "✓"
        status = "LOW"
        description = (
            "Current biometric readings indicate "
            "a lower-stress pattern."
        )

        status_function = st.success

    elif level == "moderate":

        icon = "!"
        status = "MODERATE"
        description = (
            "Current biometric readings indicate "
            "a moderate stress-related pattern."
        )

        status_function = st.warning

    else:

        icon = "!"
        status = "HIGH"
        description = (
            "Current biometric readings indicate "
            "an elevated stress-related pattern."
        )

        status_function = st.error

    # Main AI assessment card
    with st.container(border=True):

        st.markdown("###  AI Stress Assessment")

        status_function(
            f"{icon}  CURRENT STRESS LEVEL: {status}"
        )

        st.caption(description)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

        with col2:
            st.metric(
                "Classification",
                stress_level
            )

        st.divider()

        # Confidence indicator
        st.caption("AI classification confidence")

        st.progress(
            min(max(confidence / 100, 0.0), 1.0)
        )

        st.caption(
            "Classification generated from the latest "
            "wearable reading."
        )


def render_metric_card(
    icon,
    label,
    value,
    unit,
    accent=None
):

    with st.container(border=True):

        # Header
        header_col1, header_col2 = st.columns(
            [1, 4]
        )

        with header_col1:
            st.markdown(
                f"### {icon}"
            )

        with header_col2:
            st.markdown(
                f"#### {label}"
            )

        # Main reading
        st.metric(
            label="LIVE READING",
            value=f"{value} {unit}"
        )

        # Supporting description
        if accent:
            st.caption(accent)

        # Live indicator
        st.caption("● Live wearable data")

def render_biometrics_panel(latest):

    st.markdown(
        "### 📡 Live Biometrics"
    )

    st.caption(
        "Latest physiological readings received "
        "from the wearable device"
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (
            col1,
            "❤️",
            "Heart Rate",
            f"{latest['heart_rate']:.1f}",
            "BPM",
            "Cardiac activity"
        ),
        (
            col2,
            "◉",
            "Heart Rate Variability",
            f"{latest['hrv']:.1f}",
            "ms",
            "Beat-to-beat variation"
        ),
        (
            col3,
            "🫁",
            "Respiration",
            f"{latest['respiration_rate']:.1f}",
            "/min",
            "Breathing rate"
        ),
        (
            col4,
            "☾",
            "Sleep",
            f"{latest['sleep_hours']:.1f}",
            "hrs",
            "Recorded sleep"
        )
    ]

    for (
        column,
        icon,
        label,
        value,
        unit,
        description
    ) in metrics:

        with column:

            with st.container(border=True):

                st.markdown(
                    f"#### {icon} {label}"
                )

                st.metric(
                    label="CURRENT",
                    value=f"{value} {unit}"
                )

                st.caption(
                    description
                )

                st.caption(
                    "● Live"
                )
                
def render_section_title(
    title,
    description=None
):

    st.subheader(title)

    if description:
        st.caption(description)


def render_footer(timestamp):

    st.divider()

    st.caption(
        f"Last reading: {timestamp} • StressSense AI"
    )