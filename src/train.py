"""Huấn luyện và đánh giá mô hình baseline phân loại ý định (Option C)."""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from preprocessing import load_data, dataset_report

ROOT = Path(__file__).resolve().parent

# 1. Tải và kiểm tra dữ liệu
df = load_data()
print("Báo cáo tập dữ liệu (Dataset report):")
print(dataset_report(df))
print("\nPhân phối các lớp ý định (Class distribution):")
print((df["intent"].value_counts().sort_index()).to_string())

# 2. Phân chia tập huấn luyện và kiểm thử (bảo toàn tỷ lệ các lớp)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["intent"],
    test_size=0.25,
    random_state=42,
    stratify=df["intent"]
)

# 3. Xây dựng pipeline: TF-IDF (unigram + bigram) + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

# 4. Huấn luyện mô hình
start = time.perf_counter()
pipeline.fit(X_train, y_train)
training_time = time.perf_counter() - start

# 5. Dự đoán và tính toán các chỉ số đánh giá
pred = pipeline.predict(X_test)
vocabulary_size = len(pipeline.named_steps["tfidf"].vocabulary_)

labels = ["cancel_order", "payment_problem", "refund", "track_order"]
accuracy = accuracy_score(y_test, pred)
precision_macro = precision_score(y_test, pred, labels=labels, average="macro", zero_division=0)
recall_macro = recall_score(y_test, pred, labels=labels, average="macro", zero_division=0)
f1_macro = f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)

# 6. Lưu kết quả số liệu vào results.csv
results = pd.DataFrame([{
    "Model": "TF-IDF + Logistic Regression",
    "Training Time (s)": training_time,
    "Vocabulary Size": vocabulary_size,
    "Accuracy": accuracy,
    "Precision (macro)": precision_macro,
    "Recall (macro)": recall_macro,
    "F1 (macro)": f1_macro,
    "Train Samples": len(X_train),
    "Test Samples": len(X_test),
    "Random State": 42
}])
results.to_csv(ROOT / "results.csv", index=False)
print("\nKết quả đánh giá (Results):")
print(results.to_string(index=False))
print("\nBáo cáo phân loại chi tiết (Classification report):")
print(classification_report(y_test, pred, labels=labels, digits=4, zero_division=0))

# 7. Vẽ và lưu biểu đồ Confusion Matrix
cm = confusion_matrix(y_test, pred, labels=labels)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(9, 7))
disp.plot(ax=ax, values_format="d", xticks_rotation=30, colorbar=False)
ax.set_title("Ma trận nhầm lẫn — Phân loại ý định (Confusion Matrix)")
fig.tight_layout()
fig.savefig(ROOT / "confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 8. Lưu mô hình đã huấn luyện
joblib.dump(pipeline, ROOT / "intent_model.joblib")

# 9. Phân tích lỗi và ghi ra file error_analysis.md bằng tiếng Việt
errors = []
for text, true_label, predicted_label in zip(X_test, y_test, pred):
    if true_label != predicted_label:
        errors.append((text, true_label, predicted_label))

lines = [
    "# Phân tích Lỗi (Error Analysis)",
    "",
    f"Số lượng mẫu kiểm thử phân loại sai: {len(errors)} / {len(y_test)}.",
    "",
    "Tập dữ liệu này có quy mô nhỏ phục vụ minh họa. Hãy xem các lỗi dưới đây như ví dụ phân tích trực quan về sự trùng lặp từ vựng, không coi đây là ước lượng chuẩn mực tuyệt đối trong môi trường thực tế.",
    ""
]
if errors:
    for i, (text, true_label, predicted_label) in enumerate(errors, start=1):
        lines += [
            f"## Ví dụ {i}",
            f"- **Ý định thực tế (True intent)**: `{true_label}`",
            f"- **Ý định dự đoán (Predicted intent)**: `{predicted_label}`",
            f"- **Nội dung tin nhắn (Text)**: \"{text}\"",
            ""
        ]
else:
    lines.append("Không có mẫu nào bị phân loại sai trên tập kiểm thử này. Điều này không đồng nghĩa mô hình đạt độ tổng quát hóa tuyệt đối.")

(ROOT / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")

print(f"\nĐã lưu: {ROOT / 'results.csv'}")
print(f"Đã lưu: {ROOT / 'confusion_matrix.png'}")
print(f"Đã lưu: {ROOT / 'intent_model.joblib'}")
print(f"Đã lưu: {ROOT / 'error_analysis.md'}")
