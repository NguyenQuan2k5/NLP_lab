# LAB 05 — Sequence Models: RNN, LSTM, and GRU

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (Natural Language Processing)  
**Học kỳ:** Học kỳ I - 2026 | **Đại học Khoa học Tự nhiên - ĐHQGHN**  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** *From Static Representations to Sequential Modeling*  

---

## 1. Cấu trúc Thư mục Nộp bài (Deliverables Structure)

Toàn bộ dự án bài lab được tổ chức chuẩn xác theo đúng khung yêu cầu tại **Section 29 (Deliverables)** của tài liệu hướng dẫn `W5.pdf`:

```text
lab05/
├── README.md                      # Hướng dẫn tổng quan, mô tả cấu trúc và tóm tắt kết quả
├── calculations.md                # Lời giải chi tiết các bài tập tính toán giải tích và công thức
├── prediction.md                  # 5 giả thuyết dự đoán trước thực nghiệm và mã nguồn suy luận
├── rnn_cell.py                    # Cài đặt SimpleRNNCell từ đầu và kiểm chứng với PyTorch
├── sequence_classification.ipynb  # Notebook thực nghiệm toàn diện đã chạy sẵn (Full Outputs)
├── results.csv                    # Bảng số liệu định lượng chi tiết tất cả các mô hình
├── learning_curves.png            # Đồ thị Train Loss vs Validation Loss qua các Epochs
├── confusion_matrix.png           # Biểu đồ Heatmap Ma trận nhầm lẫn của các mô hình Recurrent
├── error_analysis.md              # Phân tích định tính chuyên sâu các trường hợp đúng và sai
└── reflection.md                  # Suy ngẫm, so sánh LAB 04-05, Learning Check và AI Statement
```

---

## 2. Mô tả Chi tiết Các Thành phần

| Tệp tin (File) | Định dạng | Mô tả nội dung chi tiết |
| :--- | :---: | :--- |
| [`calculations.md`](calculations.md) | Markdown | Trình bày từng bước giải tích cho **Bài 1** (Một bước Forward của RNN), **Bài 2** (Dòng chảy thông tin phủ định qua câu), **Bài 3** (Thứ tự từ trong câu), **Bài 4** (Hiện tượng Vanishing Gradient và tích Jacobian), **Bài 5** (Tính toán cổng LSTM), và công thức các chỉ số đánh giá. |
| [`prediction.md`](prediction.md) | Markdown | Ghi lại 5 dự đoán giả thuyết trước khi huấn luyện (kèm theo Cơ chế giải thích và Độ tin cậy), module Python suy luận (`predict_sentiment`), và 4 kịch bản kiểm thử ngôn ngữ đặc thù. |
| [`rnn_cell.py`](rnn_cell.py) | Python | Lớp `SimpleRNNCell` tự cài đặt theo công thức $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$. Tích hợp kiểm thử tự động so khớp độ chính xác tuyệt đối với `torch.nn.RNNCell` (sai số cực đại $< 10^{-7}$). |
| [`sequence_classification.ipynb`](sequence_classification.ipynb) | Jupyter Notebook | Toàn bộ pipeline thực nghiệm chạy hoàn chỉnh từ đầu đến cuối trên GPU, bao gồm: khảo sát dữ liệu IMDb, vẽ biểu đồ phân phối độ dài, thử nghiệm Padding/Truncation, Baseline TF-IDF + Logistic Regression, huấn luyện RNN/LSTM/GRU qua các sequence length $L \in \{50, 100, 200\}$, và thử nghiệm phụ thuộc xa (Long-range probe). |
| [`results.csv`](results.csv) | CSV | Bảng số liệu định lượng tổng hợp: Model, Representation, Max Length, Số tham số, Thời gian huấn luyện, Accuracy, Precision, Recall, Macro F1. |
| [`learning_curves.png`](learning_curves.png) | Hình ảnh PNG | Trực quan hóa diễn biến hàm mất mát (Train Loss vs Validation Loss) qua từng epoch của Simple RNN, LSTM, và GRU ($L=100$), phục vụ phân tích hiện tượng Overfitting. |
| [`confusion_matrix.png`](confusion_matrix.png) | Hình ảnh PNG | Ma trận nhầm lẫn định lượng trên 2,500 mẫu kiểm thử của cả 3 kiến trúc hồi quy. |
| [`error_analysis.md`](error_analysis.md) | Markdown | Phân tích định tính chuyên sâu trên ít nhất 3 mẫu đoán đúng và 4 mẫu đoán sai điển hình. Trả lời trực tiếp câu hỏi: *Lỗi này có phải do sequence modeling hay không?* |
| [`reflection.md`](reflection.md) | Markdown | Đánh giá so sánh có hệ thống giữa LAB 04 và LAB 05, đối chiếu kết quả với dự đoán ban đầu, sơ đồ mạch nối kiến thức chuyển tiếp sang Attention & Transformer, giải đáp bộ câu hỏi Individual Learning Check (Section 28), và Tuyên bố AI Assistance Statement (Section 27). |

---

## 3. Tóm tắt Kết quả Thực nghiệm Định lượng

Bảng tổng hợp trích xuất trực tiếp từ [`results.csv`](results.csv):

| Model | Biểu diễn (Representation) | Max Length ($L$) | Số tham số | Thời gian train (s) | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | TF-IDF (Full BoW) | Full Doc | 10,001 | 1.14 s | **0.8628** | **0.8627** |
| **Simple RNN** | Sequential Embedding (128d) | 50 | 1,313,282 | 2.53 s | 0.5488 | 0.5488 |
| **LSTM** | Sequential Embedding (128d) | 50 | 1,412,354 | 2.19 s | 0.6820 | 0.6814 |
| **GRU** | Sequential Embedding (128d) | 50 | 1,379,330 | 2.11 s | **0.6936** | **0.6936** |
| **Simple RNN** | Sequential Embedding (128d) | 100 | 1,313,282 | 2.44 s | 0.5536 | 0.5535 |
| **LSTM** | Sequential Embedding (128d) | 100 | 1,412,354 | 2.70 s | 0.6744 | 0.6744 |
| **GRU** | Sequential Embedding (128d) | 100 | 1,379,330 | 2.24 s | **0.6956** | **0.6902** |
| **Simple RNN** | Sequential Embedding (128d) | 200 | 1,313,282 | 2.44 s | 0.4924 | 0.4564 |
| **LSTM** | Sequential Embedding (128d) | 200 | 1,412,354 | 4.07 s | 0.5096 | 0.4597 |
| **GRU** | Sequential Embedding (128d) | 200 | 1,379,330 | 3.12 s | 0.5128 | 0.4657 |

### Nhận xét và Phát hiện Cốt lõi:
1. **Kiến trúc Gate vượt trội:** LSTM và GRU đạt F1 vượt trội so với Simple RNN (~0.69 vs ~0.55 tại $L=100$). Cơ chế cổng giúp duy trì luồng thông tin và hạn chế tối đa suy hao gradient.
2. **GRU hiệu quả cao:** GRU đạt kết quả tương đương hoặc nhỉnh hơn LSTM trong khi ít hơn 33,000 tham số và thời gian huấn luyện nhanh hơn.
3. **Hiệu ứng Triệt tiêu Gradient ở Simple RNN:** Khi tăng $L$ lên 200, Simple RNN bị sụt giảm F1 nghiêm trọng (về 0.4564), chứng minh trên thực tế lý thuyết Vanishing Gradient qua tích Jacobian nhiều bước.
4. **Vị thế của Baseline:** TF-IDF + Logistic Regression đạt 0.8628 nhờ tận dụng được toàn bộ văn bản (không bị cắt ngắn ở 100 token) và đặc trưng n-gram tần suất toàn cục, cho thấy mô hình sâu huấn luyện từ đầu cần nhiều dữ liệu hơn để phát huy ưu thế chuỗi.

---

## 4. Hướng dẫn Cài đặt & Thực thi

### 4.1. Cài đặt Môi trường
Yêu cầu Python 3.10+ cùng các thư viện:
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn datasets nbformat nbconvert
```

### 4.2. Chạy Kiểm chứng RNN Cell và Bài tập Tính toán
Chạy script kiểm chứng độc lập:
```bash
python lab05/rnn_cell.py
```
*Kết quả:* Sẽ thực hiện so sánh với `torch.nn.RNNCell` và in ra kết quả giải Bài tập 1.

### 4.3. Khởi chạy Jupyter Notebook
Mở notebook để xem trực tiếp các biểu đồ và kết quả thực thi:
```bash
jupyter notebook lab05/sequence_classification.ipynb
```
Hoặc mở trực tiếp trên VS Code / Google Colab.
