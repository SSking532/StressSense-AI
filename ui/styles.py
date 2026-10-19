import streamlit as st


def apply_styles():

    st.markdown(
        """
        <style>

        /* ========================================
           GLOBAL PAGE
        ======================================== */

        .stApp {
            background: #DCEDEA;
            color: #18313D;
        }

        .main .block-container {
            max-width: 1400px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }


        /* ========================================
           TYPOGRAPHY
        ======================================== */

        h1 {
            color: #18313D !important;
            font-size: 2.2rem !important;
            font-weight: 750 !important;
            letter-spacing: -0.035em;
        }

        h2 {
            color: #18313D !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

        h3,
        h4 {
            color: #18313D !important;
        }

        .stCaption {
            color: #58737A !important;
        }


        /* ========================================
           METRIC CARDS
        ======================================== */

        div[data-testid="stMetric"] {

            background: #F9FCFB;

            border: 1px solid #C7DEDB;

            border-radius: 18px;

            padding: 1.15rem 1.25rem;

            box-shadow:
                0 7px 24px rgba(
                    31,
                    95,
                    122,
                    0.07
                );

            transition:
                transform 0.2s ease,
                box-shadow 0.2s ease;
        }


        div[data-testid="stMetric"]:hover {

            transform:
                translateY(-3px);

            box-shadow:
                0 12px 30px rgba(
                    31,
                    95,
                    122,
                    0.12
                );
        }


        div[data-testid="stMetricLabel"] {

            color: #58737A !important;

            font-weight: 600 !important;
        }


        div[data-testid="stMetricValue"] {

            color: #287C9E !important;

            font-weight: 750 !important;
        }


        /* ========================================
           GENERAL CONTAINERS
        ======================================== */

        div[data-testid="stVerticalBlockBorderWrapper"] {

            background: #F9FCFB;

            border:
                1px solid #C7DEDB;

            border-radius: 18px;

            box-shadow:
                0 7px 24px rgba(
                    31,
                    95,
                    122,
                    0.065
                );
        }


        /* ========================================
           ALERTS
        ======================================== */

        div[data-testid="stAlert"] {

            border-radius: 16px;

            border-width: 1px;
        }


        /* ========================================
           PLOTLY CHARTS
        ======================================== */

        div[data-testid="stPlotlyChart"] {

            background: #F9FCFB;

            border-radius: 18px;

            padding: 0.35rem;

            border:
                1px solid #C7DEDB;

            box-shadow:
                0 7px 24px rgba(
                    31,
                    95,
                    122,
                    0.06
                );
        }


        /* ========================================
           DATAFRAME
        ======================================== */

        div[data-testid="stDataFrame"] {

            border-radius: 16px;

            overflow: hidden;
        }


        /* ========================================
           DIVIDERS
        ======================================== */

        hr {

            border-color:
                #BFD8D5 !important;

            margin-top: 1.5rem;

            margin-bottom: 1.5rem;
        }


        /* ========================================
           BUTTONS
        ======================================== */

        .stButton > button {

            border-radius: 12px;

            border:
                1px solid #BFD8D5;

            background: #F9FCFB;

            color: #1F5F7A;

            font-weight: 600;

            transition:
                all 0.2s ease;
        }


        .stButton > button:hover {

            border-color:
                #287C9E;

            color:
                #287C9E;

            transform:
                translateY(-1px);
        }


        /* ========================================
           EXPANDER
        ======================================== */

        div[data-testid="stExpander"] {

            border:
                1px solid #C7DEDB;

            border-radius: 16px;

            background: #F9FCFB;
        }


        /* ========================================
           STREAMLIT CHROME
        ======================================== */

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        </style>
        """,
        unsafe_allow_html=True
    )