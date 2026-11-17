# NLP Lab — Customer Intent Classification

Dự án phân loại ý định tin nhắn hỗ trợ khách hàng (Customer Support Intent Classification) thuộc Lab NLP.

## Cấu trúc thư mục
Toàn bộ mã nguồn và tài liệu của bài lab nằm trong thư mục [`src/`](file:///d:/hus/NLP/NLP_lab/src):
- [`src/data/intents.csv`](file:///d:/hus/NLP/NLP_lab/src/data/intents.csv): Tập dữ liệu mẫu 4 lớp intent (`refund`, `cancel_order`, `track_order`, `payment_problem`).
- [`src/preprocessing.py`](file:///d:/hus/NLP/NLP_lab/src/preprocessing.py): Xử lý và đọc dữ liệu.
- [`src/train.py`](file:///d:/hus/NLP/NLP_lab/src/train.py): Huấn luyện và đánh giá mô hình (TF-IDF + Logistic Regression).
- [`src/app.py`](file:///d:/hus/NLP/NLP_lab/src/app.py): Giao diện web demo Streamlit tương tác trực quan.
- [`src/classification.ipynb`](file:///d:/hus/NLP/NLP_lab/src/classification.ipynb): Notebook thực thi trên Google Colab / Jupyter.
- **Báo cáo và tài liệu:**
  - [`BT/BT.pdf`](file:///d:/hus/NLP/NLP_lab/BT/BT.pdf): Bản scan bài làm viết tay phần bài tập tính toán lý thuyết.
  - [`src/calculations.md`](file:///d:/hus/NLP/NLP_lab/src/calculations.md): Trình bày chi tiết lời giải các bài tập tính toán lý thuyết và công thức các chỉ số đánh giá (Accuracy, Macro Precision, Recall, F1).
  - [`src/error_analysis.md`](file:///d:/hus/NLP/NLP_lab/src/error_analysis.md): Phân tích các trường hợp mô hình phân loại sai trên tập test.
  - [`src/prediction.md`](file:///d:/hus/NLP/NLP_lab/src/prediction.md): Ví dụ mã nguồn thực thi dự đoán và kết quả mẫu.
  - [`src/reflection.md`](file:///d:/hus/NLP/NLP_lab/src/reflection.md): Báo cáo số liệu thực nghiệm, nhận xét, hạn chế và đề xuất cải tiến.
  - [`src/README.md`](file:///d:/hus/NLP/NLP_lab/src/README.md): Hướng dẫn chi tiết cài đặt và sử dụng.
