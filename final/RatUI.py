import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from retrievalAugmentedTransformer import RAT
from sentence_transformers import SentenceTransformer

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
        body {{
            background-color: {bg_color};
        }}
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
        .stButton button:hover {{
            background-color: #222222 !important;
        }}
        .block-container {{
            padding-top: 1rem;
        }}
    </style>
""", unsafe_allow_html=True)

# === LOAD EMBEDDING MODEL ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === BARCLAYS + BITS LOGOS SIDE-BY-SIDE (shifted right) ===
try:
    col1, col2, col3 = st.columns([0.7, 2.3, .1])
    with col2:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image("Barclays-Bank-logo.png", width=120)
        with c2:
            st.image("1_TExmZl1OGq1kivBg1ttY3w.png", width=120)
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
    st.markdown(f"<h4 style='color:{text_color}'>Running prediction for {ticker}...</h4>", unsafe_allow_html=True)

    try:
        model = RAT(input_dim=10, embed_dim=64, num_heads=4, num_layers=2, output_dim=1)
        raw_data = model.get_data(ticker)
        combined_data = raw_data.values.astype('float32')

        input_len = 30
        output_len = 7
        X, y = model.format_data_combined(combined_data, input_len, output_len)

        with st.spinner("Training model..."):
            model.train_model(epochs=30, batch_size=16, X=X, y=y)

        with st.spinner("Generating prediction..."):
            prediction = model.predict(X[-1].unsqueeze(0))
            pred_array = prediction.squeeze().numpy()

        st.subheader("Predicted Stock Prices")
        st.line_chart(pred_array)

        # === EXPLAINABILITY CHART ===
        st.subheader("Why Did the Model Predict This?")
        recent_input = X[-1].squeeze().numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(recent_input[:, 0], label="Closing Price (Input)", color="#ffffff")
        if recent_input.shape[1] > 1:
            ax.plot(recent_input[:, 1], label="Volume (Input)", color="#a0c4ff")
        ax.set_title("Recent Trends Influencing the Prediction", color=text_color)
        ax.set_xlabel("Days")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# === FOOTER ===
st.markdown("---")
st.markdown(f"<p style='color:{text_color}; font-size:13px; text-align:center;'>© 2025 Barclays – Datathon Project</p>", unsafe_allow_html=True)