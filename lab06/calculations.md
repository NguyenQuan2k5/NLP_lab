# LAB 06 — Attention Mechanism: Calculations & Theoretical Analysis

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (Natural Language Processing)  
**Học kỳ:** Học kỳ I - 2026 | **Đại học Khoa học Tự nhiên - ĐHQGHN**  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** *From Recurrent State to Context-Aware Representation*  

---

## Mục lục
1. [Khung lý thuyết: Từ Recurrent State sang Context-Aware Representation](#1-khung-lý-thuyết-từ-recurrent-state-sang-context-aware-representation)
2. [Bài 1 — Attention Scores (Section 8)](#2-bài-1--attention-scores-section-8)
3. [Bài 2 — Softmax Attention Weights (Section 9)](#3-bài-2--softmax-attention-weights-section-9)
4. [Bài 3 — Context Vector (Section 10)](#4-bài-3--context-vector-section-10)
5. [Bài 4 — Toàn bộ quy trình Attention Computation (Section 11)](#5-bài-4--toàn-bộ-quy-trình-attention-computation-section-11)
6. [Bài 5 — Scaled Dot-Product Attention & Temperature (Section 14, 15)](#6-bài-5--scaled-dot-product-attention--temperature-section-14-15)
7. [Bài 6 — Ổn định số học trong Softmax (Section 17)](#7-bài-6--ổn-định-số-học-trong-softmax-section-17)
8. [Bài 7 — Tự chú ý Self-Attention với ma trận phép chiếu (Section 21)](#8-bài-7--tự-chú-ý-self-attention-với-ma-trận-phép-chiếu-section-21)
9. [Bài 8 — Độ phức tạp tính toán O(T²) và "Attention không miễn phí" (Section 25, 29)](#9-bài-8--độ-phức-tạp-tính-toán-ot-và-attention-không-miễn-phí-section-25-29)

---

## 1. Khung lý thuyết: Từ Recurrent State sang Context-Aware Representation

### 1.1. Giới hạn cố hữu của Recurrent Models (RNN/LSTM/GRU)
Trong mô hình tuần tự truyền thống (LAB 05), một chuỗi văn bản $x_1, x_2, \dots, x_T$ được xử lý theo từng bước thời gian:
$$h_t = f(h_{t-1}, x_t)$$
Để phân loại câu, vector trạng thái ẩn cuối cùng $h_T$ được đưa qua bộ phân loại:
$$x_1 \to x_2 \to \dots \to x_T \implies h_T \implies \text{Classifier}$$

**Vấn đề điểm nghẽn thông tin (Information Bottleneck):**
- Toàn bộ ngữ nghĩa của chuỗi văn bản (dù dài $50, 100$ hay $1000$ tokens) bị ép nén vào một vector cố định $h_T \in \mathbb{R}^d$.
- Khi khoảng cách giữa các từ tăng lên, hiện tượng tiêu biến đạo hàm (vanishing gradient) và suy giảm thông tin ngữ cảnh khiến $h_T$ bị chi phối nặng nề bởi các token xuất hiện ở cuối câu (recency bias), làm mất các thông tin định đoạt ở đầu câu.

### 1.2. Đột phá của Cơ chế Attention
Thay vì chỉ dùng một vector $h_T$, Attention cho phép mô hình truy cập trực tiếp vào toàn bộ tập hợp trạng thái ẩn:
$$\mathcal{H} = \{h_1, h_2, \dots, h_T\}$$
Và tính toán động vector ngữ cảnh $c$ (Context Vector) thông qua cơ chế so khớp giữa **Query**, **Key** và **Value**:

```text
Query (q)
  │
  ├── dot-product(K) ──> Scores (s) ──> softmax ──> Attention Weights (α)
  │                                                           │
Value (V) ────────────────────────────────────────────────── weighted sum ──> Context Vector (c)
```

**Ý nghĩa vai trò của bộ ba Query - Key - Value:**
- **Query ($q$):** Vector đại diện cho câu hỏi hoặc mục tiêu tìm kiếm ("Tôi đang quan tâm đến thông tin gì?").
- **Key ($k_i$):** Vector đại diện cho nhãn/địa chỉ định danh của token thứ $i$ ("Tôi chứa thông tin loại gì?").
- **Value ($v_i$):** Vector mang tải trọng thông tin thực sự của token thứ $i$ ("Nội dung thông tin cụ thể của tôi là gì?").

---

## 2. Bài 1 — Attention Scores (Section 8)

### Đề bài
Cho query $q$ và ma trận key $K$:
$$q = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$
Trong đó các vector key tương ứng với từng hàng của $K$:
$$k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad k_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
Tính attention score theo tích vô hướng:
$$s_i = q^T k_i$$
Viết toàn bộ score vector $s$.

---

### Lời giải chi tiết
Score được định nghĩa là tích vô hướng giữa query $q$ và từng key $k_i$, phản ánh mức độ tương đồng định hướng trong không gian vector biểu diễn.

1. **Tính score $s_1$ giữa $q$ và $k_1$:**
   $$s_1 = q^T k_1 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1 \cdot 1 + 2 \cdot 0 = 1$$

2. **Tính score $s_2$ giữa $q$ và $k_2$:**
   $$s_2 = q^T k_2 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 1 \cdot 0 + 2 \cdot 1 = 2$$

3. **Tính score $s_3$ giữa $q$ và $k_3$:**
   $$s_3 = q^T k_3 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \cdot 1 + 2 \cdot 1 = 1 + 2 = 3$$

Dưới dạng ma trận:
$$s = K q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

**Kết luận:**  
Toàn bộ score vector là:
$$s = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \quad \text{hoặc} \quad s = [1, 2, 3]^T$$

---

## 3. Bài 2 — Softmax Attention Weights (Section 9)

### Đề bài
Cho vector điểm số:
$$s = [1, 2, 3]$$
Tính vector trọng số chú ý:
$$\alpha = \text{softmax}(s)$$
Chỉ ra rằng:
$$\sum_i \alpha_i = 1$$
Sau đó trả lời câu hỏi: *Vì sao phần tử có score lớn nhất nhận weight lớn nhất?*

---

### Lời giải chi tiết

#### 3.1. Tính toán giá trị Softmax
Hàm softmax ánh xạ các score thực sang phân phối xác suất hợp lệ:
$$\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^{3} \exp(s_j)}$$

Bước 1: Tính lũy thừa tự nhiên $\exp(s_i)$ của từng score:
- $\exp(s_1) = e^1 \approx 2.7182818$
- $\exp(s_2) = e^2 \approx 7.3890561$
- $\exp(s_3) = e^3 \approx 20.0855369$

Bước 2: Tính tổng các giá trị mũ (mẫu số chuẩn hóa):
$$\sum_{j=1}^{3} \exp(s_j) = 2.7182818 + 7.3890561 + 20.0855369 = 30.1928748$$

Bước 3: Tính trọng số chú ý $\alpha_i$ cho từng thành phần:
- $\alpha_1 = \frac{e^1}{\sum \exp} = \frac{2.7182818}{30.1928748} \approx 0.090031 \approx 9.00\%$
- $\alpha_2 = \frac{e^2}{\sum \exp} = \frac{7.3890561}{30.1928748} \approx 0.244728 \approx 24.47\%$
- $\alpha_3 = \frac{e^3}{\sum \exp} = \frac{20.0855369}{30.1928748} \approx 0.665241 \approx 66.52\%$

Vậy vector trọng số attention là:
$$\alpha = [0.0900, 0.2447, 0.6652]$$

#### 3.2. Chứng minh tổng trọng số bằng 1
$$\sum_{i=1}^{3} \alpha_i = 0.090031 + 0.244728 + 0.665241 = 1.000000$$
Theo định nghĩa giải tích:
$$\sum_{i=1}^{3} \alpha_i = \sum_{i=1}^{3} \frac{\exp(s_i)}{\sum_{j=1}^{3} \exp(s_j)} = \frac{\sum_{i=1}^{3} \exp(s_i)}{\sum_{j=1}^{3} \exp(s_j)} = 1$$
Điều này đảm bảo $\alpha$ là một tổ hợp lồi (convex combination), đóng vai trò phân phối xác suất trên các vị trí trong chuỗi.

#### 3.3. Giải thích: Vì sao phần tử có score lớn nhất nhận weight lớn nhất?
1. **Tính đơn điệu tăng ngặt của hàm mũ:** Đạo hàm của hàm $f(x) = e^x$ luôn dương:
   $$\frac{d}{dx} e^x = e^x > 0 \quad \forall x \in \mathbb{R}$$
   Do đó, nếu $s_a > s_b$, chắc chắn $e^{s_a} > e^{s_b}$. Vì mẫu số chuẩn hóa chung cho tất cả các phần tử, ta có:
   $$s_a > s_b \iff \alpha_a > \alpha_b$$
2. **Hiệu ứng khuếch đại phi tuyến (Exponential Amplification):** Hàm mũ làm tăng sự chênh lệch tuyến tính thành tỉ lệ chênh lệch cấp số nhân. Trong bài tập trên, dù score chỉ chênh lệch 1 đơn vị ($1 \to 2 \to 3$), trọng số nhận được đã tăng vọt từ $9.00\%$ lên $24.47\%$ và đạt đỉnh ở $66.52\%$. Cơ chế này giúp mô hình "tập trung sắc bén" (sharpening) vào vị trí phù hợp nhất thay vì phân bổ mờ nhạt.

---

## 4. Bài 3 — Context Vector (Section 10)

### Đề bài
Cho vector trọng số:
$$\alpha = [0.1, 0.2, 0.7]$$
Các vector giá trị (values):
$$v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$
Tính vector ngữ cảnh:
$$c = 0.1 v_1 + 0.2 v_2 + 0.7 v_3$$
Giải thích: *Value nào đóng góp nhiều nhất vào context vector?*

---

### Lời giải chi tiết

#### 4.1. Tính toán chi tiết
Context vector $c$ là trung bình có trọng số (weighted average) của các value vector:
$$c = \sum_{i=1}^{3} \alpha_i v_i$$
Thay số:
$$c = 0.1 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.2 \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.7 \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$
$$c = \begin{bmatrix} 0.1 \cdot 1 \\ 0.1 \cdot 0 \end{bmatrix} + \begin{bmatrix} 0.2 \cdot 0 \\ 0.2 \cdot 1 \end{bmatrix} + \begin{bmatrix} 0.7 \cdot 2 \\ 0.7 \cdot 2 \end{bmatrix}$$
$$c = \begin{bmatrix} 0.1 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0.2 \end{bmatrix} + \begin{bmatrix} 1.4 \\ 1.4 \end{bmatrix} = \begin{bmatrix} 0.1 + 0 + 1.4 \\ 0 + 0.2 + 1.4 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 1.6 \end{bmatrix}$$

**Kết luận:**
$$c = \begin{bmatrix} 1.5 \\ 1.6 \end{bmatrix}$$

#### 4.2. Giải thích: Value nào đóng góp nhiều nhất?
- Vector **$v_3 = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$** đóng góp nhiều nhất vào context vector $c$.
- **Lý do định lượng:** Trọng số tương ứng $\alpha_3 = 0.7$, chiếm tới **70%** toàn bộ tổng trọng lượng thông tin được tổng hợp. Đóng góp thành phần của $v_3$ là $\begin{bmatrix} 1.4 \\ 1.4 \end{bmatrix}$, chiếm $\frac{1.4}{1.5} \approx 93.3\%$ giá trị tọa độ thứ nhất và $\frac{1.4}{1.6} = 87.5\%$ giá trị tọa độ thứ hai của $c$.
- **Ý nghĩa cơ chế:** Context vector $c$ sẽ hướng mạnh về phía vector đặc trưng của token mang trọng số chú ý cao nhất, phản ánh trung thực nguyên lý "chọn lọc thông tin quan trọng nhất" của attention.

---

## 5. Bài 4 — Toàn bộ quy trình Attention Computation (Section 11)

### Đề bài
Cho query:
$$q = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$
Các keys:
$$k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad k_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
Các values:
$$v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 2 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

Thực hiện toàn bộ quy trình không dùng thư viện:
$$q \implies \text{scores} \implies \text{softmax} \implies \text{attention weights} \implies \text{weighted sum} \implies \text{context vector}$$

---

### Lời giải chi tiết

#### Bước 1: Tính toán Attention Scores ($s_i = q^T k_i$)
- $s_1 = q^T k_1 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1 \cdot 1 + 0 \cdot 0 = 1$
- $s_2 = q^T k_2 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 1 \cdot 0 + 0 \cdot 1 = 0$
- $s_3 = q^T k_3 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \cdot 1 + 0 \cdot 1 = 1$

Vector điểm số:
$$s = [1, 0, 1]$$

#### Bước 2: Tính Softmax Attention Weights ($\alpha = \text{softmax}(s)$)
- $\exp(s_1) = e^1 = e \approx 2.7182818$
- $\exp(s_2) = e^0 = 1.0$
- $\exp(s_3) = e^1 = e \approx 2.7182818$

Tổng lũy thừa:
$$\sum_{j=1}^{3} \exp(s_j) = e + 1 + e = 2e + 1 \approx 2(2.7182818) + 1 = 6.4365636$$

Các trọng số chú ý phân tích theo dạng đóng:
$$\alpha_1 = \frac{e}{2e + 1} \approx \frac{2.7182818}{6.4365636} \approx 0.422319$$
$$\alpha_2 = \frac{1}{2e + 1} \approx \frac{1.0}{6.4365636} \approx 0.155362$$
$$\alpha_3 = \frac{e}{2e + 1} \approx \frac{2.7182818}{6.4365636} \approx 0.422319$$

Kiểm tra tổng: $\alpha_1 + \alpha_2 + \alpha_3 = 0.422319 + 0.155362 + 0.422319 = 1.000000$.

Vector trọng số attention:
$$\alpha = [0.4223, 0.1554, 0.4223]$$

#### Bước 3: Tính Weighted Sum tạo Context Vector ($c = \sum_i \alpha_i v_i$)
Dạng giải tích:
$$c = \alpha_1 v_1 + \alpha_2 v_2 + \alpha_3 v_3$$
$$c = \frac{e}{2e+1} \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \frac{1}{2e+1} \begin{bmatrix} 0 \\ 2 \end{bmatrix} + \frac{e}{2e+1} \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$
$$c = \begin{bmatrix} \frac{e \cdot 1 + 1 \cdot 0 + e \cdot 2}{2e+1} \\ \frac{e \cdot 0 + 1 \cdot 2 + e \cdot 1}{2e+1} \end{bmatrix} = \begin{bmatrix} \frac{3e}{2e+1} \\ \frac{e+2}{2e+1} \end{bmatrix}$$

Thay số xấp xỉ ($e \approx 2.7182818$, $2e+1 \approx 6.4365636$):
- $c_1 = \frac{3 \cdot 2.7182818}{6.4365636} = \frac{8.1548455}{6.4365636} \approx 1.266956$
- $c_2 = \frac{2.7182818 + 2}{6.4365636} = \frac{4.7182818}{6.4365636} \approx 0.733044$

Vector ngữ cảnh cuối cùng:
$$c \approx \begin{bmatrix} 1.2670 \\ 0.7330 \end{bmatrix}$$

---

## 6. Bài 5 — Scaled Dot-Product Attention & Temperature (Section 14, 15)

### 6.1. Bài toán Scaling (Section 14)
**Đề bài:**  
Cho $d_k = 4$ và vector tích vô hướng $QK^T = [4, 2, 0]$.
1. Tính scaled scores: $\frac{QK^T}{\sqrt{d_k}} = \frac{QK^T}{\sqrt{4}}$.
2. Tính softmax của scaled scores và so sánh với trường hợp không scaling.
3. Trả lời: *Scaling làm thay đổi độ tập trung của softmax như thế nào?*

**Tính toán chi tiết:**
- Scaled scores:
  $$s_{\text{scaled}} = \frac{[4, 2, 0]}{\sqrt{4}} = \frac{[4, 2, 0]}{2} = [2, 1, 0]$$

- Softmax khi **KHÔNG scaling** ($s = [4, 2, 0]$):
  - $\exp(4) \approx 54.59815$
  - $\exp(2) \approx 7.38906$
  - $\exp(0) = 1.0$
  - Tổng $= 54.59815 + 7.38906 + 1.0 = 62.98721$
  - $\alpha_{\text{unscaled}} = \left[\frac{54.59815}{62.98721}, \frac{7.38906}{62.98721}, \frac{1.0}{62.98721}\right] = [0.8668, 0.1173, 0.0159]$

- Softmax khi **CÓ scaling** ($s_{\text{scaled}} = [2, 1, 0]$):
  - $\exp(2) \approx 7.38906$
  - $\exp(1) \approx 2.71828$
  - $\exp(0) = 1.0$
  - Tổng $= 7.38906 + 2.71828 + 1.0 = 11.10734$
  - $\alpha_{\text{scaled}} = \left[\frac{7.38906}{11.10734}, \frac{2.71828}{11.10734}, \frac{1.0}{11.10734}\right] = [0.6652, 0.2447, 0.0900]$

**Bảng so sánh đối chiếu:**

| Đại lượng | Không Scaling ($QK^T = [4, 2, 0]$) | Có Scaling ($s_{\text{scaled}} = [2, 1, 0]$) |
| :--- | :---: | :---: |
| **Max Weight ($\alpha_1$)** | **0.8668 (86.68%)** | **0.6652 (66.52%)** |
| **Weight thứ hai ($\alpha_2$)** | 0.1173 (11.73%) | 0.2447 (24.47%) |
| **Min Weight ($\alpha_3$)** | 0.0159 (1.59%) | 0.0900 (9.00%) |
| **Entropy $H(\alpha)$** | **0.4632 nats** | **0.8252 nats** |

**Trả lời câu hỏi:**
> *Scaling làm thay đổi độ tập trung của softmax như thế nào?*
1. **Làm giảm độ nhọn cực đoan (Smoother Distribution):** Khi chia cho $\sqrt{d_k} = 2$, độ chênh lệch tuyệt đối giữa các score giảm đi một nửa (từ khoảng cách $4-0=4$ xuống còn $2-0=2$). Do đó, tỉ lệ dồn trọng số vào giá trị cực đại giảm từ $86.68\%$ xuống còn $66.52\%$, phân phối trở nên cân bằng hơn.
2. **Khắc phục hiện tượng triệt tiêu đạo hàm (Vanishing Gradient):** Khi $d_k$ lớn (ví dụ trong Transformer tiêu chuẩn $d_k = 64$ hoặc $128$), phương sai của tích vô hướng $\sum_{i=1}^{d_k} q_i k_i$ bằng $d_k$. Nếu không chia cho $\sqrt{d_k}$, score có thể đạt độ lớn hàng chục hoặc hàng trăm, đẩy hàm softmax vào vùng bão hòa cực đoan (gần như one-hot vector). Tại vùng bão hòa này, đạo hàm của softmax $\frac{\partial \alpha_i}{\partial s_j} = \alpha_i (\delta_{ij} - \alpha_j)$ tiệm cận về $0$, khiến việc huấn luyện mô hình bị tắc nghẽn hoàn toàn. Phép chia cho $\sqrt{d_k}$ đưa phương sai của score trở về xấp xỉ $1$, duy trì gradient ổn định.

---

### 6.2. Thí nghiệm Temperature (Section 15)
Công thức tổng quát có tham số nhiệt độ $T > 0$:
$$\alpha_i(T) = \frac{\exp(s_i / T)}{\sum_j \exp(s_j / T)}$$
Cho vector điểm số $s = [2, 1, 0]$. Khảo sát với 3 mức nhiệt độ $T \in \{0.5, 1.0, 2.0\}$:

1. **Với $T = 0.5$ (Low Temperature):**
   - $s / 0.5 = [4, 2, 0]$
   - $\alpha(0.5) = [0.8668, 0.1173, 0.0159]$
   - Max weight: $0.8668$
   - Entropy: $H = -(0.8668 \ln 0.8668 + 0.1173 \ln 0.1173 + 0.0159 \ln 0.0159) = 0.4632$

2. **Với $T = 1.0$ (Standard Temperature):**
   - $s / 1.0 = [2, 1, 0]$
   - $\alpha(1.0) = [0.6652, 0.2447, 0.0900]$
   - Max weight: $0.6652$
   - Entropy: $H = -(0.6652 \ln 0.6652 + 0.2447 \ln 0.2447 + 0.0900 \ln 0.0900) = 0.8252$

3. **Với $T = 2.0$ (High Temperature):**
   - $s / 2.0 = [1.0, 0.5, 0.0]$
   - $\exp(s/2) = [2.7183, 1.6487, 1.0]$, $\sum = 5.3670$
   - $\alpha(2.0) = [0.5065, 0.3072, 0.1863]$
   - Max weight: $0.5065$
   - Entropy: $H = -(0.5065 \ln 0.5065 + 0.3072 \ln 0.3072 + 0.1863 \ln 0.1863) = 1.0253$

**Bảng kết quả chuẩn theo yêu cầu Section 15:**

| Temperature ($T$) | Max weight | Entropy ($H$) | Nhận xét |
| :---: | :---: | :---: | :--- |
| **0.5** | **0.8668** | **0.4632** | Phân phối cực kỳ tập trung, sắc bén (sharp). Softmax tiệm cận phép toán cứng $\text{argmax}$ (Hard Attention). |
| **1.0** | **0.6652** | **0.8252** | Phân phối tiêu chuẩn, cân bằng giữa việc tập trung vào token quan trọng và duy trì ngữ cảnh toàn cục. |
| **2.0** | **0.5065** | **1.0253** | Phân phối bị làm phẳng đáng kể (diffuse/flat), tiệm cận phân phối đều (Uniform), tăng độ đa dạng chú ý. |

**Quy luật tổng kết:**
- Khi $T \to 0^+$, $\alpha_i \to 1$ với $i = \arg\max_k s_k$, và $\alpha_j \to 0$ với mọi $j \neq i$ (Hard Attention, Deterministic).
- Khi $T \to \infty$, $s_i / T \to 0 \implies \exp(s_i / T) \to 1 \implies \alpha_i \to \frac{1}{N}$ (Uniform Attention, Maximum Entropy).

---

## 7. Bài 6 — Ổn định số học trong Softmax (Section 17)

### Đề bài
1. Trình bày công thức softmax ổn định số học (numerically stable softmax).
2. Test với vector đầu vào có giá trị cực lớn: $x = [1000, 1001, 1002]$.
3. Trả lời: *Tại sao phép trừ $\max(x)$ không làm thay đổi kết quả softmax?*

---

### Lời giải chi tiết

#### 7.1. Phân tích hiện tượng Overflow khi tính trực tiếp
Nếu tính trực tiếp $\exp(x)$ trong máy tính với chuẩn dấu phẩy động 32-bit (float32) hoặc 64-bit (float64):
- Ngưỡng giới hạn của float32 là khoảng $e^{88} \approx 3.4 \times 10^{38}$.
- Ngưỡng giới hạn của float64 là khoảng $e^{709} \approx 1.8 \times 10^{308}$.
Với $x = [1000, 1001, 1002]$, giá trị $\exp(1000) \approx 1.97 \times 10^{434}$, vượt quá giới hạn biểu diễn và dẫn tới lỗi tràn số dương (**Floating Point Overflow / `inf`**). Khi chia `inf / inf`, kết quả trả về `NaN` (Not a Number), phá hỏng toàn bộ pipeline huấn luyện.

#### 7.2. Công thức Softmax ổn định số học
Đặt $m = \max_j (x_j)$. Ta trừ $m$ khỏi mọi phần tử trước khi tính lũy thừa:
$$\text{softmax}(x_i) = \frac{\exp(x_i - m)}{\sum_{j} \exp(x_j - m)}$$

#### 7.3. Chứng minh toán học: Phép trừ $\max(x)$ bất biến với kết quả
Xét biểu thức phân tích đại số:
$$\frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)} = \frac{\exp(x_i) \cdot \exp(-m)}{\sum_j \left[\exp(x_j) \cdot \exp(-m)\right]}$$
Do hằng số $\exp(-m) > 0$ độc lập với chỉ số lấy tổng $j$, ta đưa ra ngoài dấu tổng ở mẫu số:
$$= \frac{\exp(x_i) \cdot \exp(-m)}{\exp(-m) \cdot \sum_j \exp(x_j)}$$
Triệt tiêu đại số thừa số $\exp(-m)$ ở cả tử số và mẫu số:
$$= \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
**Kết luận:** Về mặt lý thuyết toán học giải tích, hai biểu thức hoàn toàn đồng nhất!

#### 7.4. Thực thi kiểm nghiệm số học với $x = [1000, 1001, 1002]$
- Tìm giá trị lớn nhất: $m = \max(1000, 1001, 1002) = 1002$.
- Dịch chuyển vector đầu vào:
  $$x - m = [1000 - 1002, 1001 - 1002, 1002 - 1002] = [-2, -1, 0]$$
- Nhận xét: Mọi phần tử đều $\le 0$, do đó $\exp(x_i - m) \in (0, 1]$, triệt tiêu hoàn toàn nguy cơ Overflow!
- Tính lũy thừa an toàn:
  - $\exp(-2) \approx 0.135335$
  - $\exp(-1) \approx 0.367879$
  - $\exp(0) = 1.000000$
  - Tổng $= 0.135335 + 0.367879 + 1.000000 = 1.503214$
- Tính xác suất softmax:
  - $\alpha_1 = \frac{0.135335}{1.503214} \approx 0.090031$
  - $\alpha_2 = \frac{0.367879}{1.503214} \approx 0.244728$
  - $\alpha_3 = \frac{1.000000}{1.503214} \approx 0.665241$

Kết quả trả về chính xác tuyệt đối $\alpha = [0.0900, 0.2447, 0.6652]$ (khớp hoàn toàn với vector $[1, 2, 3]$ ở Bài 2 vì $[1000, 1001, 1002] = [1, 2, 3] + 999$).

---

## 8. Bài 7 — Tự chú ý Self-Attention với ma trận phép chiếu (Section 21)

### Đề bài
Cho sequence gồm 3 token: `["I", "like", "cats"]`.  
Giả sử ma trận embedding của chuỗi là $X \in \mathbb{R}^{3 \times 2}$:
$$X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$
(Hàng 1: "I", Hàng 2: "like", Hàng 3: "cats").  
Cho các ma trận trọng số chiếu tuyến tính $W_Q, W_K, W_V \in \mathbb{R}^{2 \times 2}$ ($d_k = 2$):
$$W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W_K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad W_V = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$$

Thực hiện tính toán từng bước:
1. Chiếu sang các ma trận $Q = X W_Q$, $K = X W_K$, $V = X W_V$.
2. Tính ma trận tích vô hướng $QK^T$.
3. Tính ma trận trọng số tự chú ý $A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$.
4. Tính ma trận biểu diễn ngữ cảnh $C = AV \in \mathbb{R}^{3 \times 2}$.

---

### Lời giải chi tiết

#### Bước 1: Tính ma trận $Q, K, V$
1. **Query matrix $Q = X W_Q$:**
   $$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$
2. **Key matrix $K = X W_K$:**
   $$K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}$$
3. **Value matrix $V = X W_V$:**
   $$V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & 2 \\ 1 & 3 \end{bmatrix}$$

#### Bước 2: Tính ma trận điểm số thô $QK^T$
Chuyển vị $K^T$:
$$K^T = \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix}$$
Tích ma trận $QK^T \in \mathbb{R}^{3 \times 3}$:
$$QK^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} (1\cdot 0 + 0\cdot 1) & (1\cdot 1 + 0\cdot 0) & (1\cdot 1 + 0\cdot 1) \\ (0\cdot 0 + 1\cdot 1) & (0\cdot 1 + 1\cdot 0) & (0\cdot 1 + 1\cdot 1) \\ (1\cdot 0 + 1\cdot 1) & (1\cdot 1 + 1\cdot 0) & (1\cdot 1 + 1\cdot 1) \end{bmatrix} = \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$

#### Bước 3: Tính ma trận trọng số Attention $A = \text{softmax}\left(\frac{QK^T}{\sqrt{2}}\right)$
Với $d_k = 2 \implies \sqrt{d_k} = \sqrt{2} \approx 1.4142$.  
Ma trận scaled scores:
$$S = \frac{QK^T}{\sqrt{2}} \approx \begin{bmatrix} 0 & 0.7071 & 0.7071 \\ 0.7071 & 0 & 0.7071 \\ 0.7071 & 0.7071 & 1.4142 \end{bmatrix}$$

Áp dụng Softmax trên từng hàng (row-wise softmax):
- **Hàng 1 (Token "I"):**
  - Scores: $[0, 0.7071, 0.7071]$
  - Exps: $[1.0000, 2.0281, 2.0281]$, Tổng $= 5.0562$
  - Row 1: $\left[\frac{1.0}{5.0562}, \frac{2.0281}{5.0562}, \frac{2.0281}{5.0562}\right] = [0.1978, 0.4011, 0.4011]$
- **Hàng 2 (Token "like"):**
  - Scores: $[0.7071, 0, 0.7071]$
  - Exps: $[2.0281, 1.0000, 2.0281]$, Tổng $= 5.0562$
  - Row 2: $\left[\frac{2.0281}{5.0562}, \frac{1.0}{5.0562}, \frac{2.0281}{5.0562}\right] = [0.4011, 0.1978, 0.4011]$
- **Hàng 3 (Token "cats"):**
  - Scores: $[0.7071, 0.7071, 1.4142]$
  - Exps: $[2.0281, 2.0281, 4.1132]$, Tổng $= 8.1694$
  - Row 3: $\left[\frac{2.0281}{8.1694}, \frac{2.0281}{8.1694}, \frac{4.1132}{8.1694}\right] = [0.2483, 0.2483, 0.5034]$

Ma trận Attention hoàn chỉnh:
$$A \approx \begin{bmatrix} 0.1978 & 0.4011 & 0.4011 \\ 0.4011 & 0.1978 & 0.4011 \\ 0.2483 & 0.2483 & 0.5034 \end{bmatrix}$$

#### Bước 4: Tính ma trận ngữ cảnh $C = AV$
$$C = \begin{bmatrix} 0.1978 & 0.4011 & 0.4011 \\ 0.4011 & 0.1978 & 0.4011 \\ 0.2483 & 0.2483 & 0.5034 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 2 \\ 1 & 3 \end{bmatrix}$$
- **Hàng 1 ($c_1$ cho "I"):**
  - $c_{11} = 0.1978(1) + 0.4011(0) + 0.4011(1) = 0.5989$
  - $c_{12} = 0.1978(1) + 0.4011(2) + 0.4011(3) = 0.1978 + 0.8022 + 1.2033 = 2.2033$
  - $c_1 = [0.5989, 2.2033]$
- **Hàng 2 ($c_2$ cho "like"):**
  - $c_{21} = 0.4011(1) + 0.1978(0) + 0.4011(1) = 0.8022$
  - $c_{22} = 0.4011(1) + 0.1978(2) + 0.4011(3) = 0.4011 + 0.3956 + 1.2033 = 2.0000$
  - $c_2 = [0.8022, 2.0000]$
- **Hàng 3 ($c_3$ cho "cats"):**
  - $c_{31} = 0.2483(1) + 0.2483(0) + 0.5034(1) = 0.7517$
  - $c_{32} = 0.2483(1) + 0.2483(2) + 0.5034(3) = 0.2483 + 0.4966 + 1.5102 = 2.2551$
  - $c_3 = [0.7517, 2.2551]$

Ma trận Context Representation hoàn chỉnh:
$$C \approx \begin{bmatrix} 0.5989 & 2.2033 \\ 0.8022 & 2.0000 \\ 0.7517 & 2.2551 \end{bmatrix}$$

**Ý nghĩa học thuật:**
Mỗi token ban đầu có vector tĩnh $X_i$ hoàn toàn độc lập với các từ xung quanh. Sau Self-Attention, vector mới $C_i$ đã được "nhúng ngữ cảnh" (context-aware), tích hợp thông tin liên kết trực tiếp từ mọi token khác trong toàn bộ câu.

---

## 9. Bài 8 — Độ phức tạp tính toán O(T²) và "Attention không miễn phí" (Section 25, 29)

### 9.1. Phân tích định lượng kích thước ma trận Attention
Ma trận chú ý của cơ chế Self-Attention trên chuỗi độ dài $T$ có kích thước:
$$A \in \mathbb{R}^{T \times T}$$
Số lượng phần tử cần tính toán và lưu trong bộ nhớ tỷ lệ thuận với bình phương độ dài chuỗi:
$$\mathcal{O}(T^2)$$

**Bảng thống kê số phần tử và chi phí bộ nhớ theo độ dài chuỗi:**

| Sequence Length ($T$) | Kích thước ma trận Attention ($T \times T$) | Tổng số phần tử | Bộ nhớ Float32 (1 Head, 1 Layer) | Bộ nhớ mô hình tiêu chuẩn (12 Heads, 12 Layers) |
| :---: | :---: | :---: | :---: | :---: |
| **10** | $10 \times 10$ | 100 | $400$ Bytes | $\approx 57.6$ KB |
| **100** | $100 \times 100$ | 10,000 | $40$ KB | $\approx 5.76$ MB |
| **500** | $500 \times 500$ | 250,000 | $1$ MB | $\approx 144$ MB |
| **1,000** | $1,000 \times 1,000$ | 1,000,000 | $4$ MB | $\approx 576$ MB |
| **10,000** | $10,000 \times 10,000$ | 100,000,000 | **400 MB** | **$\approx 57.6$ GB (Out of VRAM)** |

### 9.2. Trả lời câu hỏi trọng tâm: Tại sao Self-Attention có vấn đề Computational Cost khi sequence rất dài?
1. **Sự bùng nổ bậc hai (Quadratic Explosion):** Khi độ dài chuỗi tăng $k$ lần, số phép tính tích vô hướng và bộ nhớ lưu trữ ma trận attention $A$ tăng lên $k^2$ lần. Ví dụ, tăng chuỗi từ $10$ lên $1,000$ tokens ($100$ lần), chi phí tăng $100^2 = 10,000$ lần!
2. **Nghẽn băng thông bộ nhớ GPU (Memory Bandwidth Bottleneck):** Ma trận attention $(B, H, T, T)$ bắt buộc phải được ghi tạm thời vào bộ nhớ HBM của GPU để thực hiện phép toán Softmax và đạo hàm ngược (Backpropagation). Khi $T \ge 8k$ hoặc $32k$, bộ nhớ GPU 16GB/24GB lập tức bị tràn (Out of Memory - OOM).
3. **So sánh với RNN:** RNN có độ phức tạp thời gian tuyến tính $\mathcal{O}(T)$ và bộ nhớ $\mathcal{O}(1)$ khi suy luận (Inference), cho phép xử lý luồng dữ liệu vô hạn. Attention giải quyết được bài toán phụ thuộc xa của RNN nhưng phải trả một cái giá tính toán cực kỳ đắt đỏ.
4. **Hệ quả trong thiết kế kiến trúc hiện đại:** Đây là lý do ra đời các nghiên cứu khắc phục độ phức tạp bậc hai trong các bài học nâng cao: FlashAttention (tối ưu IO SRAM), Sparse Attention, Linear Attention, Longformer, và các kiến trúc State Space Models như Mamba.
