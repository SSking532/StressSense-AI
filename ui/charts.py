import plotly.graph_objects as go


# ========================================
# COLORS
# ========================================

COLORS = {
    "primary": "#287C9E",
    "deep_blue": "#1F5F7A",
    "coral": "#F28C8C",
    "apricot": "#F2B866",
    "mint": "#45B89C",
    "lavender": "#8C83D9",
    "text": "#18313D",
    "secondary": "#687F89",
    "grid": "#E0E8ED",
    "background": "#FFFFFF"
}


# ========================================
# COMMON LAYOUT
# ========================================

def base_layout(title):

    return {
        "title": {
            "text": title,
            "font": {
                "size": 16,
                "color": COLORS["text"]
            }
        },

        "paper_bgcolor": COLORS["background"],

        "plot_bgcolor": COLORS["background"],

        "font": {
            "color": COLORS["text"]
        },

        "margin": {
            "l": 20,
            "r": 20,
            "t": 55,
            "b": 20
        },

        "hovermode": "x unified",

        "xaxis": {
            "showgrid": False,
            "color": COLORS["secondary"]
        },

        "yaxis": {
            "showgrid": True,
            "gridcolor": COLORS["grid"],
            "color": COLORS["secondary"]
        }
    }


# ========================================
# HEART RATE CHART
# ========================================

def heart_rate_chart(df):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["heart_rate"],
            mode="lines+markers",
            name="Heart Rate",

            line=dict(
                color=COLORS["coral"],
                width=3
            ),

            marker=dict(
                size=6
            ),

            hovertemplate=(
                "<b>%{y:.1f} BPM</b>"
                "<br>%{x}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        **base_layout("Heart Rate Trend")
    )

    fig.update_yaxes(
        title_text="BPM"
    )

    return fig


# ========================================
# STRESS LEVEL CHART
# ========================================

def stress_level_chart(df):

    level_map = {
        "Low": 0,
        "Moderate": 1,
        "High": 2
    }

    plot_df = df.copy()

    plot_df["stress_numeric"] = (
        plot_df["stress_level"]
        .map(level_map)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["stress_numeric"],

            mode="lines+markers",

            name="Stress Level",

            line=dict(
                color=COLORS["lavender"],
                width=3
            ),

            marker=dict(
                size=7
            ),

            customdata=plot_df["stress_level"],

            hovertemplate=(
                "<b>%{customdata}</b>"
                "<br>%{x}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        **base_layout("Stress Level Trend")
    )

    fig.update_yaxes(
        title_text="Stress Level",

        tickmode="array",

        tickvals=[
            0,
            1,
            2
        ],

        ticktext=[
            "Low",
            "Moderate",
            "High"
        ],

        range=[
            -0.2,
            2.2
        ]
    )

    return fig


# ========================================
# PROBABILITY CHART
# ========================================

def probability_chart(probabilities):

    labels = list(
        probabilities.keys()
    )

    values = [
        float(value)
        for value in probabilities.values()
    ]

    colors = []

    for label in labels:

        if label == "Low":

            colors.append(
                COLORS["mint"]
            )

        elif label == "Moderate":

            colors.append(
                COLORS["apricot"]
            )

        else:

            colors.append(
                COLORS["coral"]
            )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,

            orientation="h",

            marker=dict(
                color=colors
            ),

            text=[
                f"{value:.2f}%"
                for value in values
            ],

            textposition="outside",

            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Model Confidence: %{x:.2f}%"
                "<extra></extra>"
            )
        )
    )

    # ------------------------------------
    # CUSTOM LAYOUT
    # ------------------------------------

    layout = base_layout(
        "AI Classification Confidence"
    )

    layout["xaxis"] = {
        "title": "Model Confidence (%)",
        "range": [0, 105],
        "showgrid": True,
        "gridcolor": COLORS["grid"],
        "color": COLORS["secondary"]
    }

    layout["yaxis"] = {
        "showgrid": False,
        "color": COLORS["secondary"]
    }

    layout["showlegend"] = False

    layout["height"] = 300

    fig.update_layout(
        **layout
    )

    return fig