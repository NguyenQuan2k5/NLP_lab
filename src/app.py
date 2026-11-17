"""Ứng dụng Streamlit demo phân loại ý định khách hàng 4 lớp."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "intent_model.joblib"

st.set_page_config(page_title="Phân loại Ý định Khách hàng", page_icon="💬", layout="centered")
st.title("Hỗ trợ Khách hàng — Phân loại Ý định")
st.write("Nhập tin nhắn khách hàng để dự đoán ý định (Intent). Bản demo này chỉ thực hiện phân loại văn bản, không liên kết thực hiện hành động thật.")

if not MODEL_PATH.exists():
    st.warning("Chưa tìm thấy mô hình đã huấn luyện. Hãy chạy `python train.py` trước.")
    st.stop()

model = joblib.load(MODEL_PATH)
message = st.text_area(
    "Tin nhắn của khách hàng (Customer message)",
    placeholder="Ví dụ: Where is my order? hoặc I want a refund",
    height=120
)

if st.button("Dự đoán ý định (Predict intent)", type="primary"):
    if not message.strip():
        st.error("Vui lòng nhập nội dung tin nhắn.")
    else:
        predicted = model.predict([message])[0]
        st.success(f"Ý định được dự đoán: **{predicted}**")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([message])[0]
            classes = model.named_steps["classifier"].classes_
            proba_df = pd.DataFrame({
                "Ý định (Intent)": classes,
                "Xác suất": probabilities
            }).sort_values("Xác suất", ascending=False)
            st.subheader("Xác suất dự đoán theo từng lớp (Class probabilities)")
            st.bar_chart(proba_df.set_index("Ý định (Intent)"))
            st.dataframe(proba_df.assign(Xác_suất=proba_df["Xác suất"].map(lambda x: f"{x:.1%}")).rename(columns={"Xác_suất": "Xác suất (Probability)"})[["Ý định (Intent)", "Xác suất (Probability)"]], hide_index=True)

st.divider()
st.caption("Mô hình cơ sở (Baseline): TF-IDF (unigram + bigram) + Hồi quy Logistic. Kết quả dự đoán phụ thuộc vào tập dữ liệu mẫu huấn luyện.")
