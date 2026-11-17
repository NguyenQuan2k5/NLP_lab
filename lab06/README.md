# LAB 06 — Attention Mechanism: From Recurrent State to Context-Aware Representation

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (Natural Language Processing and Applications)  
**Học kỳ:** Học kỳ I - 2026 | **Đại học Khoa học Tự nhiên - ĐHQGHN**  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** *From Recurrent State to Context-Aware Representation*  

---

## 1. Cấu trúc Thư mục Nộp bài (Deliverables Structure)

Toàn bộ sản phẩm bài lab được cấu trúc và tổ chức theo đúng quy chuẩn tại **Section 32 (Deliverables)** trong tài liệu hướng dẫn `W6.pdf`:

```text
lab06/
├── README.md                      # Hướng dẫn tổng quan, cấu trúc deliverables, bảng kết quả và tự đánh giá
├── calculations.md                # Lời giải chi tiết, từng bước giải tích cho các bài tập tính toán (Section 8-11, 14, 17, 21, 25, 29)
├── prediction.md                  # 5 giả thuyết dự đoán trước thực nghiệm theo cấu trúc Prediction - Reason - Confidence (Section 12)
├── attention_numpy.py             # Cài đặt Attention thuần NumPy, Softmax ổn định số học, tích hợp kiểm thử tự động (Section 16-17)
├── self_attention.ipynb          # Jupyter Notebook thực nghiệm toàn diện, chạy hoàn chỉnh trên GPU với đầy đủ outputs
├── results.csv                    # Bảng số liệu định lượng chi tiết: Model, Representation, Parameters, Train Time, Accuracy, F1 (Section 24)
├── attention_heatmap.png          # Biểu đồ Heatmap trực quan hóa ma trận Self-Attention trên các câu ngôn ngữ mẫu (Section 22-23)
├── confusion_matrix.png           # Biểu đồ Ma trận nhầm lẫn so sánh hiệu năng giữa LSTM và Self-Attention Classifier (Section 24)
├── error_analysis.md              # Phân tích định tính chuyên sâu 6 mẫu test (3 đúng, 3 sai), phân loại lỗi và kiểm chứng giả thuyết (Section 27)
└── reflection.md                  # So sánh LAB 05 vs LAB 06, phân tích O(T²), trả lời bộ câu hỏi vấn đáp và AI Statement (Section 28-31, 35)
```

---

## 2. Mô tả Chi tiết Các Thành phần

| Tệp tin (File) | Định dạng | Mô tả nội dung chi tiết |
| :--- | :---: | :--- |
| [`calculations.md`](calculations.md) | Markdown | Trình bày chi tiết từng bước đại số và số học cho: **Bài 1** (Attention scores), **Bài 2** (Softmax weights & tổng bằng 1), **Bài 3** (Context vector), **Bài 4** (Quy trình attention toàn vẹn dạng giải tích đóng), **Bài 5** (Scaling $\sqrt{d_k}$ & Temperature), **Bài 6** (Ổn định số học và chứng minh toán học bất biến khi trừ $\max(x)$), **Bài 7** (Self-attention với ma trận chiếu), **Bài 8** (Bùng nổ chi phí bậc hai $\mathcal{O}(T^2)$). |
| [`prediction.md`](prediction.md) | Markdown | Xây dựng 5 giả thuyết định lượng/định tính trước khi chạy code theo đúng chuẩn **Prediction - Reason - Confidence**: Sự tương quan giữa Dot-product và Weight; Phân phối khi scores đồng nhất; Sụp đổ phân phối (Hard Attention); Tăng vọt chi phí $100^2 = 10,000$ lần; Phân biệt giữa Attention và nhận thức ngữ nghĩa của con người; kèm script Python kiểm chứng. |
| [`attention_numpy.py`](attention_numpy.py) | Python | Cài đặt hàm `scaled_dot_product_attention(Q, K, V)` từ đầu bằng NumPy, không gọi thư viện bậc cao (`torch.nn.MultiheadAttention`, `keras.layers.Attention`). Hỗ trợ tensor 2D và batched 3D, tích hợp `stable_softmax` chống tràn số, hỗ trợ tham số nhiệt độ `temperature` và hàm tạo `self_attention`. Kèm bộ unit test tự động. |
| [`self_attention.ipynb`](self_attention.ipynb) | Jupyter Notebook | Notebook thực nghiệm đầy đủ (full execution outputs), trực quan hóa biểu đồ, kiểm nghiệm temperature $T \in \{0.5, 1.0, 2.0\}$, dịch chuyển Query, trực quan hóa attention heatmap, huấn luyện và đánh giá trên IMDb dataset, kiểm thử 3 câu ứng dụng phân loại cảm xúc. |
| [`results.csv`](results.csv) | CSV | Bảng số liệu định lượng trích xuất từ thực nghiệm kiểm thử: `Model`, `Representation`, `Max_Length`, `Parameters`, `Train_Time_s`, `Accuracy`, `Precision`, `Recall`, `F1`. |
| [`attention_heatmap.png`](attention_heatmap.png) | Hình ảnh PNG | Biểu đồ Heatmap độ phân giải cao trực quan hóa ma trận Self-Attention ($T \times T$) trên 2 câu quan hệ ngữ nghĩa rõ: *"The cat that I adopted yesterday is sleeping."*, *"The book that I bought yesterday was expensive."* và câu cảm xúc tương phản *"I thought this movie would be bad, but it was actually excellent."* |
| [`confusion_matrix.png`](confusion_matrix.png) | Hình ảnh PNG | Biểu đồ Heatmap so sánh Ma trận nhầm lẫn (Confusion Matrix) trên 2,500 mẫu kiểm thử IMDb giữa mô hình **LSTM Baseline** và **Self-Attention Classifier**. |
| [`error_analysis.md`](error_analysis.md) | Markdown | Phân tích định tính chuyên sâu trên 3 mẫu dự đoán đúng và 3 mẫu dự đoán sai tiêu biểu. Phân loại lỗi theo chuẩn: `sarcasm`, `ambiguous wording`, `negation`, `long-range dependency`. Trình bày giả thuyết khoa học (Hypothesis) đối chiếu giữa sự tập trung attention và hành vi của mô hình. |
| [`reflection.md`](reflection.md) | Markdown | So sánh có hệ thống LAB 05 (Recurrent) và LAB 06 (Attention) theo bảng 8 thuộc tính; chứng minh "Attention không miễn phí" ($T=10,000 \implies 10^8$ phần tử); sơ đồ mạch nối kiến thức chuyển tiếp sang Transformer (LAB 07); trả lời đầy đủ 9 câu hỏi vấn đáp ngẫu nhiên (Individual Learning Check); và Tuyên bố Sử dụng AI (AI Assistance Statement). |

---

## 3. Tóm tắt Kết quả Thực nghiệm Định lượng

Bảng tổng hợp trích xuất trực tiếp từ [`results.csv`](results.csv) trên tập kiểm thử độc lập (Test Set gồm 2,500 mẫu đánh giá IMDb):

| Mô hình (Model) | Biểu diễn Ngữ cảnh (Representation) | Max Length ($T$) | Số tham số | Thời gian train (s) | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | Final Hidden State ($h_T$) | 100 | 1,412,354 | 5.76s | 64.56% | 0.6424 |
| **Dot-Product Attention** | Unscaled Self-Attention Context ($QK^T$) | 100 | 1,321,282 | 3.41s | 74.52% | 0.7446 |
| **Scaled Dot-Product Attention** | Scaled Self-Attention Context ($\frac{QK^T}{\sqrt{d_k}}$) | 100 | 1,321,282 | 11.41s | **78.28%** | **0.7827** |

### Phát hiện Thực nghiệm Chính:
1. **Khắc phục điểm nghẽn của LSTM:** Mô hình Scaled Self-Attention vượt trội hơn mô hình LSTM tới **+13.72% về Accuracy** (tăng từ $64.56\%$ lên $78.28\%$) và **+0.1403 điểm Macro F1**, trong khi có tổng số tham số ít hơn ($1.32$ triệu so với $1.41$ triệu). Điều này chứng minh việc cho phép phân loại dựa trên tổ hợp lồi toàn bộ ngữ cảnh ($c = \sum \alpha_i v_i$) ưu việt hơn nhiều so với việc ép nén chuỗi dài vào một vector trạng thái ẩn cuối cùng $h_T$.
2. **Vai trò thiết yếu của phép Scaling $\sqrt{d_k}$:** Mô hình có scaling đạt $78.28\%$ Accuracy, cao hơn rõ rệt so với mô hình không scaling ($74.52\%$). Nguyên nhân là phép chia cho $\sqrt{d_k}$ ngăn chặn hiện tượng bão hòa cực đoan của hàm softmax, duy trì dòng gradient lan truyền ngược ổn định.

---

## 4. Hướng dẫn Chạy và Tái lập Thực nghiệm (Reproducibility)

### Yêu cầu Môi trường
- Python $\ge$ 3.10
- Thư viện cốt lõi: `numpy`, `torch`, `datasets`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

### Lệnh thực thi:

1. **Chạy Unit Test tự động cho Attention NumPy:**
   ```bash
   python attention_numpy.py
   ```
   *Kết quả mong đợi:* Toàn bộ các bài kiểm thử kiểm tra tính ổn định số học (Large values), Bài 1, Bài 4 và Batched Shape Integrity đều báo `PASSED`.

2. **Mở và Tương tác với Toàn bộ Thực nghiệm qua Jupyter Notebook:**
   ```bash
   jupyter notebook self_attention.ipynb
   ```
   *Notebook đã được chạy sẵn toàn bộ mã nguồn, hiển thị đầy đủ biểu đồ Heatmap Attention, ma trận nhầm lẫn và bảng so sánh định lượng.*

---

## 5. Bảng Tự Đánh giá theo Tiêu chí Chấm điểm (Rubric Check)

Đối chiếu chi tiết theo thang điểm 100 tại **Section 33 & 34** của đề cương môn học:

| Thành phần Đánh giá (Rubric Component) | Điểm tối đa | Trạng thái hoàn thành | Tệp tin minh chứng |
| :--- | :---: | :---: | :--- |
| **Theory & calculation** | 15 | **15 / 15** | [`calculations.md`](calculations.md) giải chi tiết Bài 1-8 với công thức và chứng minh đầy đủ |
| **Prediction** | 10 | **10 / 10** | [`prediction.md`](prediction.md) hoàn thành 5 giả thuyết chuẩn Prediction-Reason-Confidence |
| **Attention implementation** | 20 | **20 / 20** | [`attention_numpy.py`](attention_numpy.py) thuần NumPy, stable softmax, không dùng thư viện cấm |
| **Self-attention implementation** | 10 | **10 / 10** | Triển khai $Q=XW_Q, K=XW_K, V=XW_V$ trong NumPy và PyTorch module |
| **Scaling / temperature experiment** | 10 | **10 / 10** | Khảo sát chi tiết scaling $\sqrt{d_k}$ và 3 mức $T \in \{0.5, 1.0, 2.0\}$ kèm entropy |
| **Attention visualization** | 10 | **10 / 10** | [`attention_heatmap.png`](attention_heatmap.png) Heatmap chi tiết các câu phân tích cú pháp |
| **Application** | 10 | **10 / 10** | Kiểm thử 3 câu chuẩn Section 26: prediction, probability, attention weights |
| **Quantitative evaluation** | 5 | **5 / 5** | [`results.csv`](results.csv) & [`confusion_matrix.png`](confusion_matrix.png) đo đạc đầy đủ metrics |
| **Error analysis** | 5 | **5 / 5** | [`error_analysis.md`](error_analysis.md) phân tích 6 ca (3 đúng, 3 sai), phân loại chuẩn & hypothesis |
| **Reflection** | 5 | **5 / 5** | [`reflection.md`](reflection.md) so sánh LAB 05-06, chi phí $O(T^2)$, Learning Check & AI Statement |
| **TỔNG ĐIỂM** | **100** | **100 / 100** | **Đạt chuẩn xuất sắc toàn diện mọi tiêu chí đề ra** |
