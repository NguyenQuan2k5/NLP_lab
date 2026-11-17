# Calculations & Evaluation Metrics

Tài liệu này bao gồm phần giải chi tiết bài tập tính toán lý thuyết và định nghĩa các chỉ số đánh giá thực nghiệm trong bài lab.

---

## Phần 1: Bài tập tính toán lý thuyết

### Bài 1: Tính xác suất với Logistic Regression
**Đề bài:**
Cho vector đặc trưng $x = [1, 2, 0]$, vector trọng số $w = [0.5, -0.2, 0.4]$, và hệ số chệch $b = 0.1$.

**Lời giải:**
1. Tính tổng tuyến tính $z = w^T x + b$:
$$z = (0.5 \times 1) + (-0.2 \times 2) + (0.4 \times 0) + 0.1$$
$$z = 0.5 - 0.4 + 0 + 0.1 = 0.2$$

2. Tính xác suất dự đoán $P(y=1|x)$ qua hàm kích hoạt Sigmoid $\sigma(z)$:
$$P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-0.2}} \approx 0.5498$$

---

### Bài 2: Phân loại theo ngưỡng quyết định (Threshold)
Dựa vào xác suất tính được ở Bài 1 ($P(y=1|x) \approx 0.5498$):

1. **Dự đoán nhãn theo ngưỡng $T$:**
   - Với ngưỡng $T = 0.5$: Vì $P(y=1|x) = 0.5498 \ge 0.5 \implies$ Mô hình dự đoán nhãn $\hat{y} = 1$.
   - Với ngưỡng $T = 0.8$: Vì $P(y=1|x) = 0.5498 < 0.8 \implies$ Mô hình dự đoán nhãn $\hat{y} = 0$.

2. **Quy luật biến thiên Precision - Recall khi điều chỉnh ngưỡng:**
   - **Khi tăng ngưỡng threshold:** Mô hình thận trọng hơn khi dự đoán nhãn positive $\implies$ **Precision tăng, Recall giảm**.
   - **Khi giảm ngưỡng threshold:** Mô hình dự đoán nhiều mẫu là positive hơn $\implies$ **Precision giảm, Recall tăng**.

---

### Bài 3: Tính Precision, Recall, F1 và ý nghĩa thực tế
**Đề bài:** Cho ma trận nhầm lẫn nhị phân với $TP = 80$, $FP = 10$, $FN = 20$.

**Lời giải:**
1. **Precision (Độ chuẩn xác):**
$$Precision = \frac{TP}{TP + FP} = \frac{80}{80 + 10} = \frac{8}{9} \approx 0.8889 \quad (88.89\%)$$

2. **Recall (Độ thu hồi):**
$$Recall = \frac{TP}{TP + FN} = \frac{80}{80 + 20} = \frac{80}{100} = 0.8 \quad (80.00\%)$$

3. **F1-score:**
$$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} = 2 \times \frac{0.8889 \times 0.8}{0.8889 + 0.8} \approx 0.8421$$

4. **Ứng dụng trong hệ thống phát hiện email Spam:**
   - **Ưu tiên Recall:** Khi muốn hạn chế tối đa việc bỏ sót thư spam lọt vào hộp thư đến (chấp nhận có thể đưa nhầm một vài thư bình thường vào spam).
   - **Ưu tiên Precision:** Khi muốn tránh việc đưa nhầm email công việc/học tập quan trọng vào mục spam (chỉ đánh dấu spam khi mô hình cực kỳ chắc chắn).

---

### Bài 4: Đánh giá mô hình trên dữ liệu mất cân bằng
**Tình huống:** Dữ liệu mất cân bằng nghiêm trọng (95% Negative, 5% Positive).

**Nhận xét & giải pháp:**
- **Không thể kết luận** mô hình tốt hay xấu chỉ dựa trên chỉ số **Accuracy**. Một mô hình đơn giản luôn dự đoán toàn bộ là Negative cũng sẽ đạt Accuracy $95\%$, nhưng hoàn toàn thất bại trong việc phát hiện lớp Positive (lớp cần quan tâm nhất).
- **Giải pháp:** Cần sử dụng thêm các thước đo đánh giá:
  - **Precision, Recall, F1-score** cho riêng lớp thiểu số (Positive).
  - **Balanced Accuracy** (trung bình Recall của từng lớp).
  - **PR-AUC (Precision-Recall AUC)** hoặc **ROC-AUC** để có cái nhìn toàn diện về chất lượng mô hình.

---

## Phần 2: Đánh giá mô hình Intent Classification (Option C)

Bài toán phân loại ý định khách hàng có 4 lớp (`cancel_order`, `payment_problem`, `refund`, `track_order`), do đó báo cáo sử dụng **Macro Average** để mỗi lớp có vai trò công bằng.

- **Accuracy**: Tỷ lệ dự đoán đúng trên toàn bộ 20 mẫu tập test.
- **Precision (macro)**: Trung bình cộng Precision của 4 lớp.
- **Recall (macro)**: Trung bình cộng Recall của 4 lớp.
- **F1 (macro)**: Trung bình cộng F1 của 4 lớp.
- **Training time**: Thời gian khớp mô hình `pipeline.fit()`.
- **Vocabulary size**: Kích thước bộ từ vựng TF-IDF sinh ra từ tập huấn luyện.

Công thức Macro-F1 với $K = 4$ lớp:
$$Macro\text{-}F1 = \frac{1}{K}\sum_{i=1}^{K}F1_i$$

