import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
from retrievalAugmentedTransformer import RAT, Data_Processing
import os

# === PAGE CONFIG ===
st.set_page_config(page_title="Barclays RAT Predictor", page_icon="📈", layout="centered")

# === COLOR THEME ===
bg_color = "#0a409d"
text_color = "#ffffff"
label_color = "#ffffff"
card_color = "#143d8c"
shadow = "0px 2px 15px rgba(0,0,0,0.3)"

# === CUSTOM CSS ===
st.markdown(f"""
    <style>
        body {{ background-color: {bg_color}; }}
        .main {{
            background-color: {card_color};
            padding: 2rem 3rem;
            border-radius: 12px;
            box-shadow: {shadow};
            margin-top: 1.5rem;
            text-align: center;
        }}
        h1, h2, h3, h4, h5, h6, p, .stSubheader, .stMarkdown {{
            color: {text_color} !important;
            font-family: 'Segoe UI', sans-serif;
        }}
        .stTextInput label, .stDateInput label {{
            color: {label_color} !important;
        }}
        .stButton button {{
            background-color: #000000 !important;
            color: #ffffff !important;
            font-weight: 700;
            font-size: 1rem;
            border-radius: 8px;
            padding: 0.6em 1.5em;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        .stButton button:hover {{ background-color: #222222 !important; }}
        .block-container {{ padding-top: 1rem; }}
    </style>
""", unsafe_allow_html=True)

# === BARCLAYS + BITS LOGOS ===
try:
    col1, col2, col3 = st.columns([0.7, 2.3, .1])
    with col2:
        c1, c2 = st.columns([1, 1])
        with c1:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Barclays-Bank-logo.png")
            st.image(file_path, width=120)
        with c2:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_TExmZl1OGq1kivBg1ttY3w.png")
            st.image(file_path, width=120)
except Exception as e:
    st.warning("⚠️ One or both logos not found. Check filenames and paths.")
    st.exception(e)

# === HEADER CARD ===
st.markdown(f"""
<div class='main'>
    <h1>Barclays Stock Predictor</h1>
    <p style='margin-top: 0; font-size: 1rem;'>Powered by a Retrieval-Augmented Transformer (RAT)</p>
</div>
""", unsafe_allow_html=True)

# === INPUT FORM ===
st.markdown("### Enter Prediction Info", unsafe_allow_html=True)
ticker = st.text_input("Enter a stock ticker:", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

# === PREDICTION LOGIC ===
if st.button("Predict"): 
    st.write("Fetching stock data...")

    # Initialize RAT model
    data_process = Data_Processing()

    # Get stock + macroeconomic data as combined DataFrame
    data = data_process.get_data(ticker, start_date, end_date)
    data = data_process.load_features(ticker, data)
    data_process.save_features(ticker, data)

    model = RAT(stock=ticker, input_dim=len(data.columns), embed_dim=64, output_dim=2, output_length=15)

    input_length = 30
    output_length = 15

    X, y = data_process.format_data_combined(data, model, input_length, output_length)

    model.query_articles(ticker)

    with st.spinner("Training model..."):
        if not model.load_model():
            model.train_model(X, y)

        model.save_model()

    with st.spinner("Generating prediction..."):
        prediction, _ = model.predict(X)
        pred_array = prediction[:, 0, 0]

    st.subheader("Predicted Stock Prices")
    st.line_chart(pred_array)

    recent_input = X[:, 0, 0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(recent_input, label="Closing Price (Input)", color="#ffffff")
    #ax.plot(recent_input, label="Volume (Input)", color="#a0c4ff")
    ax.set_title("Recent Trends Influencing the Prediction", color=text_color)
    ax.set_xlabel("Days")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

# === FOOTER ===
st.markdown("---")
st.markdown(f"<p style='color:{text_color}; font-size:13px; text-align:center;'>© 2025 Barclays – Datathon Project</p>", unsafe_allow_html=True)