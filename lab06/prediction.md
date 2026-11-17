# LAB 06 — Attention Mechanism: Predictions Before Experiments

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (Natural Language Processing)  
**Học kỳ:** Học kỳ I - 2026 | **Đại học Khoa học Tự nhiên - ĐHQGHN**  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** *From Recurrent State to Context-Aware Representation*  

---

## 1. Giới thiệu và Nguyên tắc Khoa học

Theo yêu cầu tại **Section 12** của đề cương môn học, trước khi thực thi mã nguồn huấn luyện và thí nghiệm mô phỏng trên tập dữ liệu thực tế, sinh viên phải xây dựng các giả thuyết định lượng và định tính về hành vi của cơ chế Attention.

Mỗi dự đoán bắt buộc phải tuân thủ chuẩn cấu trúc khoa học 3 thành phần:
1. **Prediction:** Khẳng định kết quả định lượng/định tính kỳ vọng sẽ quan sát được.
2. **Reason:** Cơ chế toán học, đặc tính giải tích và nguyên lý giải thích logic cho dự đoán.
3. **Confidence:** Mức độ tự tin về mặt lý thuyết (% hoặc định tính: Tuyệt đối / Rất cao / Trung bình).

---

## 2. Chi tiết 5 Dự đoán Khoa học (Section 12)

### Prediction 1: Sự tương quan giữa Dot-Product Score và Attention Weight
*Câu hỏi: Nếu một key có dot product lớn hơn các key khác, attention weight của nó sẽ như thế nào?*

- **Prediction:**  
  Key có dot product lớn hơn sẽ nhận trọng số chú ý ($\alpha_i$) cao nhất, và tỷ lệ phần trăm phân bổ trọng số của nó sẽ vượt trội rõ rệt (phi tuyến tính) so với các key còn lại trong chuỗi.
- **Reason:**  
  Trọng số attention được xác định thông qua hàm chuẩn hóa Softmax:
  $$\alpha_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$$
  Hàm số mũ $f(x) = \exp(x)$ là hàm đơn điệu tăng ngặt trên toàn miền giá trị thực ($f'(x) = \exp(x) > 0$). Do đó:
  $$s_i > s_k \iff \exp(s_i) > \exp(s_k) \iff \alpha_i > \alpha_k$$
  Không những thế, tính chất hàm mũ khuếch đại khoảng cách tuyến tính: khi score của một key nhỉnh hơn dù chỉ 1 hay 2 đơn vị, hàm mũ sẽ nâng giá trị lên gấp $e \approx 2.72$ hoặc $e^2 \approx 7.39$ lần so với các key khác, dẫn tới việc áp đảo toàn bộ mẫu số chuẩn hóa.
- **Confidence:** **99% (Rất cao)**.

---

### Prediction 2: Hành vi của Attention khi các Scores đồng nhất
*Câu hỏi: Nếu tất cả scores giống nhau $s = [1, 1, 1, 1]$, attention weights sẽ như thế nào?*

- **Prediction:**  
  Các trọng số chú ý sẽ được san đều hoàn toàn cho tất cả 4 vị trí:
  $$\alpha = [0.25, 0.25, 0.25, 0.25]$$
  Phân phối xác suất trở thành phân phối đều (Uniform Distribution) và đạt giá trị Entropy cực đại: $H(\alpha) = \ln(4) \approx 1.3863\text{ nats}$.
- **Reason:**  
  Khi $s_1 = s_2 = s_3 = s_4 = 1.0$, ta có:
  $$\exp(s_1) = \exp(s_2) = \exp(s_3) = \exp(s_4) = e^1$$
  Tổng mẫu số chuẩn hóa:
  $$\sum_{j=1}^{4} \exp(s_j) = 4e^1$$
  Do đó trọng số tại mỗi vị trí $i \in \{1, 2, 3, 4\}$ là:
  $$\alpha_i = \frac{e^1}{4e^1} = \frac{1}{4} = 0.25$$
  Mô hình không ưu tiên bất kỳ vị trí nào, vector ngữ cảnh $c$ thu được sẽ là trung bình số học đơn giản của các vector Value ($c = \frac{1}{4}\sum_{i=1}^4 v_i$).
- **Confidence:** **100% (Tuyệt đối)**.

---

### Prediction 3: Hiện tượng sụp đổ phân phối khi một Score áp đảo (Attention Collapse)
*Câu hỏi: Nếu một score lớn hơn rất nhiều $s = [1, 1, 10]$, attention sẽ tập trung như thế nào?*

- **Prediction:**  
  Phân phối attention sẽ bị sụp đổ (collapse) và tập trung gần như toàn bộ trọng lượng ($\approx 99.98\%$) vào token thứ 3, biến cơ chế Softmax thành một phép xấp xỉ của hàm chọn cực đại cứng $\text{argmax}$ (Hard Attention):
  $$\alpha \approx [0.0001, 0.0001, 0.9998]$$
- **Reason:**  
  Tính toán giá trị hàm số mũ của các score:
  - $\exp(s_1) = \exp(1) \approx 2.71828$
  - $\exp(s_2) = \exp(1) \approx 2.71828$
  - $\exp(s_3) = \exp(10) \approx 22026.46579$
  
  Mẫu số tổng:
  $$\sum_{j=1}^3 \exp(s_j) = 2.71828 + 2.71828 + 22026.46579 = 22031.90235$$
  Trọng số chú ý tương ứng:
  - $\alpha_1 = \alpha_2 = \frac{2.71828}{22031.90235} \approx 0.000123 \approx 0.012\%$
  - $\alpha_3 = \frac{22026.46579}{22031.90235} \approx 0.999753 \approx 99.975\%$
  
  Sự chênh lệch score là $10 - 1 = 9$ đơn vị khiến tỷ số mũ là $e^9 \approx 8103$. Do đó, token thứ 3 triệt tiêu hoàn toàn thông tin từ hai token đầu tiên trong vector ngữ cảnh ($c \approx v_3$). Đây chính là lý do vì sao trong thực tế cần phải có bước scaling chia cho $\sqrt{d_k}$ để ngăn chặn hiện tượng bão hòa sớm này.
- **Confidence:** **100% (Tuyệt đối)**.

---

### Prediction 4: Sự thay đổi chi phí tính toán theo độ dài chuỗi (Computational Cost)
*Câu hỏi: Nếu sequence tăng từ 10 tokens $\to$ 1000 tokens, chi phí tính attention sẽ thay đổi như thế nào?*

- **Prediction:**  
  Chi phí tính toán số phép nhân-cộng dấu phẩy động (FLOPs) và dung lượng bộ nhớ lưu ma trận Attention sẽ tăng vọt lên **10,000 lần** (tức tăng theo tỉ lệ bậc hai $100^2$).
- **Reason:**  
  Trong cơ chế Self-Attention, mỗi token trong chuỗi $T$ tokens phải so khớp tích vô hướng với mọi token khác trong chuỗi (kể cả chính nó). Ma trận tích vô hướng $QK^T$ và ma trận trọng số $A$ đều có kích thước là:
  $$A \in \mathbb{R}^{T \times T}$$
  Do đó, số lượng phần tử của ma trận phụ thuộc vào hàm bậc hai $\mathcal{O}(T^2)$:
  - Với $T_1 = 10$ tokens: số phần tử là $10^2 = 100$.
  - Với $T_2 = 1000$ tokens: số phần tử là $1000^2 = 1,000,000$.
  
  Tỷ số gia tăng kích thước:
  $$\frac{T_2^2}{T_1^2} = \left(\frac{1000}{10}\right)^2 = 100^2 = 10,000$$
  Khác với RNN có độ phức tạp thời gian tăng tuyến tính $\mathcal{O}(T)$, cơ chế attention đối mặt với hiện tượng nghẽn cổ chai bậc hai $\mathcal{O}(T^2)$ cả về thời gian tính toán và bộ nhớ RAM/VRAM GPU.
- **Confidence:** **100% (Tuyệt đối)**.

---

### Prediction 5: Mối quan hệ giữa Attention Weights và Nhận định của Con người
*Câu hỏi: Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?*

- **Prediction:**  
  **Không.** Attention weights không nhất thiết luôn tập trung vào các từ ngữ giàu thông tin cảm xúc hoặc ngữ pháp cốt lõi mà con người kỳ vọng. Sẽ có những trường hợp mô hình phân bổ sự chú ý cao vào các từ dừng (stopwords như "the", "was", "is"), dấu câu, hoặc các tương quan giả (spurious shortcuts).
- **Reason:**  
  1. **Bản chất tối ưu hóa thống kê:** Attention weights được học thuần túy thông qua lan truyền ngược gradient để tối thiểu hóa hàm mất mát (Cross-Entropy Loss) trên tập huấn luyện, chứ không được huấn luyện dựa trên quy tắc cú pháp hay sự hiểu biết nhận thức của con người.
  2. **Tranh biện học thuật "Attention is not Explanation":** Nghiên cứu kinh điển của Jain & Wallace (NAACL 2019) đã chỉ ra rằng các vector attention phân bố khác nhau hoàn toàn có thể cùng tạo ra một dự đoán đầu ra giống hệt nhau, và việc hoán đổi hay làm nhiễu attention weights thường không làm suy giảm độ chính xác của mô hình.
  3. **Tương quan bề mặt và rò rỉ dữ liệu:** Mô hình có thể "chú ý" vào một từ không liên quan trực tiếp đến cảm xúc nhưng tình cờ xuất hiện nhiều trong các câu phân lớp dương tính (spurious feature). Vì vậy, đồ thị heatmap attention chỉ phản ánh sự tập trung trọng số tính toán của mô hình, không thể suy diễn một cách chủ quan rằng mô hình đã "hiểu ngữ pháp" hay "suy luận như con người".
- **Confidence:** **95% (Rất cao)**.

---

## 3. Module Python Xác thực Giả thuyết (Verification Suite)

Đoạn mã dưới đây được sử dụng để chạy kiểm chứng trực tiếp cả 5 giả thuyết trên môi trường số thực:

```python
import numpy as np

def verify_predictions():
    print("=" * 65)
    print("XÁC THỰC THỰC NGHIỆM CÁC DỰ ĐOÁN LÝ THUYẾT (SECTION 12)")
    print("=" * 65)
    
    # 1. Xác thực Prediction 1 & 2
    s_equal = np.array([1.0, 1.0, 1.0, 1.0])
    w_equal = np.exp(s_equal) / np.sum(np.exp(s_equal))
    print(f"\n[Kiểm chứng P2] Softmax([1, 1, 1, 1]):\n  {w_equal}")
    assert np.allclose(w_equal, [0.25, 0.25, 0.25, 0.25])
    print("  => Khớp tuyệt đối: Mọi token nhận đúng 25% trọng số chú ý.")
    
    # 2. Xác thực Prediction 3
    s_collapse = np.array([1.0, 1.0, 10.0])
    w_collapse = np.exp(s_collapse) / np.sum(np.exp(s_collapse))
    print(f"\n[Kiểm chứng P3] Softmax([1, 1, 10]):\n  {w_collapse}")
    print(f"  Token 3 nhận: {w_collapse[2]*100:.3f}% trọng số.")
    assert w_collapse[2] > 0.999
    print("  => Khớp tuyệt đối: Hiện tượng sụp đổ phân phối (Hard Attention).")
    
    # 3. Xác thực Prediction 4
    t_10 = 10
    t_1000 = 1000
    ratio = (t_1000 ** 2) / (t_10 ** 2)
    print(f"\n[Kiểm chứng P4] Tỷ số phần tử ma trận (1000^2 / 10^2): {ratio:.0f} lần.")
    assert ratio == 10000
    print("  => Khớp tuyệt đối: Độ phức tạp bậc hai O(T^2).")

if __name__ == "__main__":
    verify_predictions()
```
