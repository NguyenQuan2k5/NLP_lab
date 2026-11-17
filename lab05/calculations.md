# Calculations & Theoretical Analysis — LAB 05

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** From Static Representations to Sequential Modeling (RNN, LSTM, GRU)  
**Tác giả:** Sinh viên thực hiện LAB 05  

Tài liệu này trình bày chi tiết lời giải các bài tập tính toán giải tích, phân tích dòng chảy thông tin (information flow), hiện tượng triệt tiêu gradient (vanishing gradient), cơ chế cổng LSTM, và các công thức đánh giá định lượng trong bài toán Sequence Classification.

---

## 1. Bài tập 1: Tính toán một bước Forward của Simple RNN (Section 6)

### 1.1. Đề bài
Cho mô hình Simple RNN tại bước thời gian $t$ với:
- Vector đầu vào: 
  $$x_t = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \in \mathbb{R}^2$$
- Hidden state từ timestep trước đó ($t-1$): 
  $$h_{t-1} = \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix} \in \mathbb{R}^2$$
- Ma trận trọng số input-to-hidden ($W_{xh}$):
  $$W_{xh} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$
- Ma trận trọng số recurrent hidden-to-hidden ($W_{hh}$):
  $$W_{hh} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$
- Vector hệ số chệch (bias):
  $$b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \in \mathbb{R}^2$$

**Yêu cầu:**
1. Tính giá trị affine activation: $a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$.
2. Tính hidden state tại thời điểm hiện tại: $h_t = \tanh(a_t)$.

---

### 1.2. Lời giải chi tiết từng bước

#### Bước 1: Tính thành phần biến đổi tuyến tính từ đầu vào $W_{xh} x_t$
$$W_{xh} x_t = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 \times 1 + 0.2 \times 0 \\ 0.1 \times 1 + 0.4 \times 0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}$$

#### Bước 2: Tính thành phần biến đổi tuyến tính từ hidden state quá khứ $W_{hh} h_{t-1}$
$$W_{hh} h_{t-1} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix} \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 0.3 \times 0.5 + 0.1 \times 0.2 \\ 0.2 \times 0.5 + 0.5 \times 0.2 \end{bmatrix} = \begin{bmatrix} 0.15 + 0.02 \\ 0.10 + 0.10 \end{bmatrix} = \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}$$

#### Bước 3: Tính tổng kết hợp (affine activation $a_t$)
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix} + \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 + 0.17 \\ 0.1 + 0.20 \end{bmatrix} = \begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}$$

#### Bước 4: Áp dụng hàm phi tuyến $\tanh$ để tìm $h_t$
Hàm $\tanh(z)$ được định nghĩa:
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

Áp dụng element-wise cho từng phần tử của vector $a_t$:
- Phần tử 1: 
  $$h_{t,1} = \tanh(0.67) = \frac{e^{0.67} - e^{-0.67}}{e^{0.67} + e^{-0.67}} \approx \frac{1.954237 - 0.511709}{1.954237 + 0.511709} = \frac{1.442528}{2.465946} \approx 0.584980$$
- Phần tử 2:
  $$h_{t,2} = \tanh(0.30) = \frac{e^{0.30} - e^{-0.30}}{e^{0.30} + e^{-0.30}} \approx \frac{1.349859 - 0.740818}{1.349859 + 0.740818} = \frac{0.609041}{2.090677} \approx 0.291313$$

**Kết luận:**
$$a_t = \begin{bmatrix} 0.6700 \\ 0.3000 \end{bmatrix}, \quad h_t = \begin{bmatrix} 0.584980 \\ 0.291313 \end{bmatrix}$$

*(Kết quả này đã được xác thực khớp 100% bằng script trong `rnn_cell.py`)*.

---

## 2. Bài tập 2: Dòng chảy thông tin (Information Flow) qua Sequence (Section 7)

### 2.1. Đề bài
Cho câu văn bản:
$$\text{"I did not like this movie"}$$
Giả sử mô hình RNN quét tuần tự từ trái sang phải qua các timestep $t = 1, 2, \dots, 6$:
$$\text{"I"} \to \text{"did"} \to \text{"not"} \to \text{"like"} \to \text{"this"} \to \text{"movie"}$$

**Câu hỏi:**
1. Mô tả trạng thái ẩn ($h_t$) sau từng bước.
2. Tại thời điểm mô hình đọc từ `"like"` ($t=4$), thông tin của từ `"not"` ($t=3$) ảnh hưởng tới $h_4$ như thế nào?
3. So sánh cơ chế này với biểu diễn Average Word2Vec từ LAB 04.

---

### 2.2. Trả lời chi tiết

#### 1. Quá trình truyền trạng thái ẩn qua từng timestep:
- **Bước 1 ($t=1$, token = `"I"`):**
  $$h_1 = \tanh(W_{xh} x_{\text{"I"}} + W_{hh} h_0 + b_h)$$
  Khởi tạo ngữ cảnh với chủ ngữ của câu.
- **Bước 2 ($t=2$, token = `"did"`):**
  $$h_2 = \tanh(W_{xh} x_{\text{"did"}} + W_{hh} h_1 + b_h)$$
  Mô hình ghi nhận trợ động từ quá khứ, kết hợp với chủ ngữ trong $h_1$.
- **Bước 3 ($t=3$, token = `"not"`):**
  $$h_3 = \tanh(W_{xh} x_{\text{"not"}} + W_{hh} h_2 + b_h)$$
  Trạng thái ẩn $h_3$ được nạp tín hiệu mang sắc thái **phủ định** mạnh mẽ.
- **Bước 4 ($t=4$, token = `"like"`):**
  $$h_4 = \tanh(W_{xh} x_{\text{"like"}} + W_{hh} h_3 + b_h)$$
  Tại đây, từ `"like"` (mang tính tích cực khi đứng độc lập) được nạp qua $W_{xh} x_{\text{"like"}}$, nhưng được cộng trực tiếp với $W_{hh} h_3$. Vì $h_3$ đã mang ngữ cảnh của `"not"`, phép biến đổi phi tuyến $\tanh$ cho phép mô hình triệt tiêu hoặc đảo chiều cực tính tích cực của `"like"`, biến ngữ cảnh thành "phủ định tình cảm" (dislike/negative).
- **Bước 5 ($t=5$, token = `"this"`):**
  $$h_5 = \tanh(W_{xh} x_{\text{"this"}} + W_{hh} h_4 + b_h)$$
  Bổ sung chỉ định từ xác định đối tượng.
- **Bước 6 ($t=6$, token = `"movie"`):**
  $$h_6 = \tanh(W_{xh} x_{\text{"movie"}} + W_{hh} h_5 + b_h)$$
  Vector cuối cùng $h_6$ đại diện cho toàn bộ câu (sentence representation) để đưa vào classifier: $z = W_{hy} h_6 + b_y$.

#### 2. Ảnh hưởng của `"not"` tới `"like"`:
Tại $t=4$:
$$h_4 = \tanh\Big( W_{xh} x_{\text{"like"}} + W_{hh} \underbrace{\tanh(W_{xh} x_{\text{"not"}} + W_{hh} h_2 + b_h)}_{h_3} + b_h \Big)$$
Thông tin của từ `"not"` được nén trong $h_3$ và tương tác phi tuyến trực tiếp với vector của từ `"like"` thông qua ma trận tương tác $W_{hh}$. Trọng số $W_{hh}$ được học qua backpropagation để phát hiện các mẫu phủ định phổ biến (negation pattern), làm suy giảm hoặc đảo ngược giá trị activation trong $h_4$.

#### 3. So sánh với Average Word2Vec (LAB 04):
Trong Average Word2Vec, biểu diễn cả câu là:
$$v_{\text{AvgW2V}} = \frac{1}{6} \left( v_{\text{"I"}} + v_{\text{"did"}} + v_{\text{"not"}} + v_{\text{"like"}} + v_{\text{"this"}} + v_{\text{"movie"}} \right)$$
- Phép cộng vector có **tính chất giao hoán** ($A + B = B + A$) và kết hợp.
- Vector $v_{\text{"not"}}$ và vector $v_{\text{"like"}}$ chỉ đơn thuần cộng dồn như hai điểm độc lập trong không gian hình học. Mô hình tuyến tính không thể xác định được `"not"` đứng cạnh từ nào (phủ định `"like"`, phủ định `"movie"` hay phủ định một từ khác).
- Ngược lại, **Simple RNN là mô hình có trạng thái phụ thuộc thứ tự** ($h_t = f(x_t, h_{t-1})$), giúp lưu giữ cấu trúc ngữ pháp và phạm vi phủ định (negation scope) một cách tự nhiên.

---

## 3. Bài tập 3: Thứ tự từ trong chuỗi (Sequence Order) (Section 8)

### 3.1. Đề bài
Cho hai chuỗi văn bản:
- $S_1 = \text{"dog bites man"}$ (chó cắn người — chuyện bình thường)
- $S_2 = \text{"man bites dog"}$ (người cắn chó — tin chấn động, giật gân)

Giả sử cả hai câu đều sử dụng chung một tập từ vựng (vocabulary).

**Câu hỏi:**
1. Average Word2Vec có thể phân biệt được hai câu này dựa trên thứ tự hay không? Giải thích toán học.
2. RNN phân biệt chúng dựa trên cơ chế mô hình như thế nào?

---

### 3.2. Trả lời chi tiết

#### 1. Average Word2Vec:
- Biểu diễn của $S_1$:
  $$v(S_1) = \frac{1}{3} \left( v_{\text{dog}} + v_{\text{bites}} + v_{\text{man}} \right)$$
- Biểu diễn của $S_2$:
  $$v(S_2) = \frac{1}{3} \left( v_{\text{man}} + v_{\text{bites}} + v_{\text{dog}} \right)$$
- Do phép cộng vector trên không gian $\mathbb{R}^d$ có tính chất giao hoán:
  $$v_{\text{dog}} + v_{\text{bites}} + v_{\text{man}} \equiv v_{\text{man}} + v_{\text{bites}} + v_{\text{dog}} \implies v(S_1) = v(S_2)$$
- **Kết luận:** Average Word2Vec cho ra đúng **cùng một vector biểu diễn duy nhất** cho cả hai câu. Mọi bộ phân loại (Logistic Regression, SVM, MLP) nhận đầu vào là $v(S)$ đều sẽ đưa ra dự đoán giống hệt nhau $P(y|S_1) = P(y|S_2)$. Nó **hoàn toàn bất lực** trong việc phân biệt chủ ngữ (agent) và tân ngữ (patient).

#### 2. Simple RNN:
RNN duyệt tuần tự qua từng vị trí:
- Đối với $S_1 = (\text{dog}, \text{bites}, \text{man})$:
  $$h_1^{(1)} = \tanh(W_{xh} x_{\text{dog}})$$
  $$h_2^{(1)} = \tanh(W_{xh} x_{\text{bites}} + W_{hh} h_1^{(1)})$$
  $$h_3^{(1)} = \tanh(W_{xh} x_{\text{man}} + W_{hh} h_2^{(1)})$$
- Đối với $S_2 = (\text{man}, \text{bites}, \text{dog})$:
  $$h_1^{(2)} = \tanh(W_{xh} x_{\text{man}})$$
  $$h_2^{(2)} = \tanh(W_{xh} x_{\text{bites}} + W_{hh} h_1^{(2)})$$
  $$h_3^{(2)} = \tanh(W_{xh} x_{\text{dog}} + W_{hh} h_2^{(2)})$$

- Tại timestep 3 (trạng thái cuối cùng để phân loại):
  - Ở $S_1$, đầu vào hiện tại là $x_{\text{man}}$ kết hợp với quá khứ $(W_{hh} h_2^{(1)})$ mang dấu ấn của `"dog"`.
  - Ở $S_2$, đầu vào hiện tại là $x_{\text{dog}}$ kết hợp với quá khứ $(W_{hh} h_2^{(2)})$ mang dấu ấn của `"man"`.
- Vì các phép nhân ma trận $W_{hh} W_{xh}$ và hàm phi tuyến $\tanh$ không có tính giao hoán:
  $$h_3^{(1)} \neq h_3^{(2)}$$
- Do đó, vector đại diện câu của $S_1$ và $S_2$ hoàn toàn khác nhau trong không gian đặc trưng. Classifier ở tầng trên có thể dễ dàng tách biệt hai mẫu này dựa trên vị trí cú pháp của từng thực thể.

---

## 4. Bài tập 4: Hiện tượng Triệt tiêu Gradient (Vanishing Gradient) trong RNN (Section 9)

### 4.1. Cơ chế Backpropagation Through Time (BPTT)
Xét một chuỗi có độ dài $T$, hàm mất mát $L$ được tính tại bước cuối $T$ dựa trên $h_T$. Để cập nhật ma trận trọng số (ví dụ $W_{xh}$ hoặc $W_{hh}$), ta cần tính gradient của $L$ theo $h_t$ tại bước thời gian $t \ll T$.

Theo quy tắc dây chuyền nhiều biến (Multivariate Chain Rule):
$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \frac{\partial h_T}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Trong đó, ma trận Jacobian chuyển trạng thái giữa hai timestep liên tiếp là:
$$\frac{\partial h_k}{\partial h_{k-1}} = \operatorname{diag}\left( 1 - h_k^2 \right) W_{hh}^T$$
*(với $\operatorname{diag}(1 - h_k^2)$ là đạo hàm của hàm $\tanh$ tại điểm $a_k$)*.

### 4.2. Giải thích vì sao chuỗi càng dài thì RNN càng khó học phụ thuộc xa:
1. **Tích lũy suy giảm theo hàm mũ:** Đạo hàm của hàm $\tanh$:
   $$0 < 1 - h_k^2 \le 1$$
   Tại hầu hết các điểm hoạt động thực tế, $1 - h_k^2 < 1$. Nếu giá trị riêng lớn nhất (spectral radius) của ma trận trọng số $W_{hh}$ thỏa mãn $\rho(W_{hh}) \le 1$, thì định mức (norm) của ma trận Jacobian luôn nhỏ hơn 1:
   $$\left\| \frac{\partial h_k}{\partial h_{k-1}} \right\| \le \gamma < 1$$
2. Khi khoảng cách thời gian giữa timestep hiện tại $T$ và timestep quá khứ $t$ tăng lên:
   $$\left\| \frac{\partial h_T}{\partial h_t} \right\| \le \prod_{k=t+1}^T \gamma = \gamma^{T-t} \xrightarrow{T - t \to \infty} 0$$
3. **Hệ quả trong huấn luyện:**
   - Khi gradient $\frac{\partial L}{\partial h_t} \approx 0$, biến thiên cập nhật trọng số $\Delta W = -\eta \frac{\partial L}{\partial W}$ cho các từ khóa ở đầu câu bị triệt tiêu gần như hoàn toàn.
   - Mô hình chỉ học được các mối liên hệ cục bộ ngắn hạn (short-term dependencies) trong khoảng 5 - 10 bước gần nhất, mất khả năng kết nối ngữ nghĩa giữa từ mở đầu và từ kết thúc trong một đoạn văn dài.

---

## 5. Bài tập 5: Tính toán Cổng trong LSTM (LSTM Gates) (Section 17)

### 5.1. Đề bài
Cho một timestep của mạng LSTM với các giá trị đã tính toán:
- Giá trị cổng quên (Forget gate): $f_t = 0.9$
- Giá trị cổng vào (Input gate): $i_t = 0.2$
- Trạng thái ứng viên mới (Candidate cell state): $\tilde{c}_t = 0.5$
- Trạng thái ô nhớ tại bước trước (Previous cell state): $c_{t-1} = 0.8$

**Yêu cầu:**
1. Tính trạng thái ô nhớ hiện tại: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.
2. Phân tích ý nghĩa khi $f_t \to 1$ và khi $f_t \to 0$.
3. Giải thích bản chất của cơ chế cổng trong việc điều khiển dòng thông tin.

---

### 5.2. Lời giải chi tiết

#### 1. Tính $c_t$:
$$c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t = (0.9 \times 0.8) + (0.2 \times 0.5) = 0.72 + 0.10 = 0.82$$

#### 2. Phân tích hành vi của Forget Gate ($f_t$):
- **Khi $f_t \to 1$ (ở đây $f_t = 0.9$ rất gần 1):**
  - Thành phần nhớ $f_t \cdot c_{t-1} = 0.72$ chiếm tới $\frac{0.72}{0.82} \approx 87.8\%$ tổng trạng thái mới.
  - Mô hình có xu hướng **bảo tồn gần như nguyên vẹn thông tin quá khứ**. Điều này cho phép thông tin quan trọng được lưu truyền liên tục qua hàng chục hoặc hàng trăm timestep mà không bị suy hao.
- **Khi $f_t \to 0$:**
  - Thành phần $f_t \cdot c_{t-1} \approx 0$.
  - Mô hình có xu hướng **xóa bỏ hoàn toàn (reset/forget) thông tin cũ**, loại bỏ các ngữ cảnh không còn liên quan (ví dụ khi bắt đầu một câu mới hoặc một mệnh đề mới) và chỉ tiếp nhận thông tin mới từ $\tilde{c}_t$.

#### 3. Vai trò điều khiển dòng thông tin (Information Flow):
- Cơ chế cổng sử dụng hàm kích hoạt Sigmoid $\sigma(z) \in [0, 1]$ đóng vai trò như các "van thủy lực" liên tục và khả vi:
  - Cổng quên ($f_t$): Điều tiết lượng thông tin cũ được giữ lại trong cell state.
  - Cổng vào ($i_t$): Điều tiết lượng thông tin ứng viên mới được nạp vào cell state.
  - Cổng ra ($o_t$): Điều tiết lượng thông tin từ cell state được giải phóng ra hidden state $h_t$ phục vụ dự đoán và truyền sang bước kế tiếp.
- **Giải quyết Vanishing Gradient:** Đường truyền của cell state là phép cộng tuyến tính ($c_t = f_t c_{t-1} + \dots$). Đạo hàm riêng:
  $$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$
  Khi mạng học cách giữ $f_t \approx 1$, gradient truyền ngược qua cell state không bị nhân với ma trận trọng số và không qua hàm phi tuyến co hẹp, tạo thành một **"xa lộ gradient" (error carousel)** truyền thẳng xuyên suốt các bước thời gian.

---

## 6. Công thức các Chỉ số Đánh giá Định lượng (Section 22)

Trong bài toán phân loại nhị phân cảm xúc phim (IMDb: Positive = 1, Negative = 0):
- **TP (True Positive):** Số mẫu thực tế Positive được dự đoán đúng là Positive.
- **TN (True Negative):** Số mẫu thực tế Negative được dự đoán đúng là Negative.
- **FP (False Positive):** Số mẫu thực tế Negative bị dự đoán nhầm thành Positive (Lỗi Type I).
- **FN (False Negative):** Số mẫu thực tế Positive bị dự đoán nhầm thành Negative (Lỗi Type II).

### Các chỉ số:
1. **Accuracy (Độ chính xác tổng thể):**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision (Độ chuẩn xác):**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Độ bao phủ / Độ nhạy):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score (Trung bình điều hòa giữa Precision và Recall):**
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$

---

## 7. Công thức Đếm số Tham số Mô hình (Model Parameters)

Với kích thước Embedding $E$, chiều ẩn Hidden Size $H$, kích thước từ vựng $V$, và số lớp nhãn $C = 2$:

1. **Simple RNN:**
   - Recurrent Cell: $W_{xh} \in \mathbb{R}^{H \times E}$, $W_{hh} \in \mathbb{R}^{H \times H}$, $b_h \in \mathbb{R}^H$.
   - Số tham số RNN: $H \times E + H \times H + H = H(E + H + 1)$.
2. **LSTM:**
   - 4 cổng ($f, i, \tilde{c}, o$), mỗi cổng có trọng số cho $x$ và $h$ kèm bias riêng.
   - Số tham số LSTM: $4 \times [H(E + H + 1)]$.
3. **GRU:**
   - 3 cổng (Reset gate $r$, Update gate $z$, Candidate state $\tilde{h}$).
   - Số tham số GRU: $3 \times [H(E + H + 1)]$.
4. **Phân loại (Linear Layer):**
   - $H \times C + C = 2H + 2$.
5. **Embedding Layer:**
   - $V \times E$.

*(Các công thức trên giải thích vì sao cùng kích thước ẩn $H=128$, LSTM có tham số gấp khoảng 4 lần Simple RNN và GRU có tham số gấp 3 lần Simple RNN)*.
