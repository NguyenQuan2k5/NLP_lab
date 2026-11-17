# Báo cáo Đánh giá, Suy ngẫm & Mạch nối Kiến thức (Reflection) — LAB 05

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I - 2026 | **ĐHQGHN - Trường Đại học Khoa học Tự nhiên**  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** From Static Representations to Sequential Modeling (RNN, LSTM, GRU)  

---

## 1. So sánh Tổng quan: LAB 04 vs. LAB 05 (Section 25)

Trong LAB 04, chúng ta đã tiếp cận bài toán phân loại văn bản bằng các biểu diễn tĩnh (Static Representations) như **TF-IDF** và **Average Word2Vec**. Sang LAB 05, chúng ta chuyển dịch sang mô hình hóa chuỗi tuần tự (**Sequential Modeling**) bằng **Simple RNN**, **LSTM**, và **GRU**.

### 1.1. Bảng so sánh toàn diện các phương pháp
Dựa trên lý thuyết và các số liệu thực nghiệm đã thu thập:

| Tiêu chí / Đặc tính | TF-IDF + Logistic Regression | Average Word2Vec + Logistic Regression | Simple RNN + Embedding | LSTM + Embedding | GRU + Embedding |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Loại biểu diễn (Representation)** | Thưa (Sparse, $V$-dim) | Dày đặc (Dense, $d$-dim) | Dày đặc tuần tự (Sequential Dense) | Dày đặc tuần tự (Sequential Dense) | Dày đặc tuần tự (Sequential Dense) |
| **Mô hình hóa thứ tự từ (Word Order)** | Rất hạn chế (chỉ qua n-gram) | Hoàn toàn KHÔNG (tính giao hoán) | **CÓ** (tuần tự qua $h_t$) | **CÓ** (tuần tự qua $h_t, c_t$) | **CÓ** (tuần tự qua $h_t$) |
| **Trạng thái phụ thuộc ngữ cảnh (Context-dependent state)** | KHÔNG | KHÔNG | **CÓ** ($h_t = f(x_t, h_{t-1})$) | **CÓ** ($h_t = f(x_t, h_{t-1}, c_{t-1})$) | **CÓ** ($h_t = f(x_t, h_{t-1})$) |
| **Xử lý phụ thuộc xa (Long dependency)** | Kém (phụ thuộc vào n-gram lớn) | Kém (bị trung bình hóa làm nhòe) | Rất kém (Vanishing Gradient) | **Tốt hơn nhiều** (Cell state & Forget gate) | **Tốt hơn nhiều** (Update gate) |
| **Chi phí tính toán (Computational cost)** | Rất thấp (huấn luyện mili-giây) | Rất thấp | Cao hơn (tuần tự theo $T$, khó song song) | Cao hơn (4 bộ ma trận cổng) | Trung bình (3 bộ ma trận cổng, ít hơn LSTM) |
| **Khả năng song song hóa (Parallelization)** | Rất cao (ma trận hóa hoàn toàn) | Rất cao | **Rất thấp** (bắt buộc $t$ chờ $t-1$) | **Rất thấp** (bắt buộc $t$ chờ $t-1$) | **Rất thấp** (bắt buộc $t$ chờ $t-1$) |
| **Số tham số mô hình** | $\sim 10,001$ | $\sim 301$ | $\sim 1,313,282$ | $\sim 1,412,354$ | $\sim 1,379,330$ |

---

### 1.2. Phân tích Bản chất: RNN/LSTM/GRU giải quyết vấn đề gì và còn tồn tại hạn chế nào?

#### Những vấn đề Recurrent Models đã giải quyết thành công:
1. **Phá vỡ giả định túi từ (Bag-of-Words):** Văn bản tự nhiên là một chuỗi có cấu trúc phân tầng thời gian. Các mô hình tuần tự cho phép phân biệt rạch ròi giữa `"dog bites man"` và `"man bites dog"`, hoặc giữa `"not good"` và `"good, not"`.
2. **Mô hình hóa phạm vi phủ định (Negation Scope):** Nhờ cơ chế cập nhật trạng thái $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$, từ phủ định `"not"` tác động trực tiếp và điều chỉnh vector biểu diễn của tính từ đứng ngay sau nó.
3. **Kích thước đầu vào linh hoạt:** Khác với MLP yêu cầu vector đầu vào có số chiều cố định, mạng hồi quy có thể nhận chuỗi văn bản với độ dài $T$ bất kỳ nhờ tái sử dụng cùng một bộ trọng số $(W_{xh}, W_{hh})$.

#### Những hạn chế cốt lõi vẫn còn tồn tại:
1. **Nút thắt cổ chai thông tin (Information Bottleneck):** Mô hình buộc phải nén toàn bộ thông tin của chuỗi hàng trăm token vào một vector ẩn duy nhất ở bước cuối $h_T$. Điều này làm suy hao nghiêm trọng các chi tiết ngữ nghĩa nằm rải rác trong câu.
2. **Khó khăn trong huấn luyện chuỗi dài (Vanishing Gradient):** Mặc dù LSTM và GRU đã cải thiện đáng kể bằng cơ chế cổng, khi độ dài chuỗi vượt quá 200 - 500 token, khả năng duy trì thông tin từ những token đầu tiên vẫn bị suy giảm rõ rệt.
3. **Điểm nghẽn tính toán tuần tự (Sequential Dependency):** Vì $h_t$ phụ thuộc bắt buộc vào $h_{t-1}$, quá trình tính toán **không thể song song hóa theo chiều thời gian** trên GPU. Đây chính là rào cản kỹ thuật lớn nhất ngăn cản việc mở rộng quy mô (scaling) mạng hồi quy trên các kho dữ liệu văn bản khổng lồ.

---

## 2. Đối chiếu Thực nghiệm với các Giả thuyết Ban đầu (Prediction vs. Reality)

| Giả thuyết ban đầu | Dự đoán lý thuyết | Kết quả thực nghiệm thực tế | Nhận định đối chiếu |
| :--- | :--- | :--- | :--- |
| **Prediction 1 (Word Order)** | RNN xử lý trật tự từ trực tiếp và vượt trội TF-IDF. | Thực nghiệm chứng minh RNN phản ứng khác nhau với vị trí từ, trong khi TF-IDF bất biến với hoán vị từ. | **Khớp hoàn toàn.** Bản chất kiến trúc $h_t$ cho phép mô hình hóa trật tự từ một cách tự nhiên. |
| **Prediction 2 (Long Dependency)** | LSTM xử lý phụ thuộc xa tốt hơn RNN nhiều lần nhờ cell state. | Trong thí nghiệm Probe câu phụ thuộc xa (Exp 6), LSTM và GRU nhận diện chính xác cảm xúc khi từ khóa ở đầu/giữa câu ($P \approx 0.88 - 0.91$), trong khi Simple RNN bị đóng băng ở xác suất $P \approx 0.5599$. | **Khớp hoàn toàn.** Gradient của Simple RNN bị triệt tiêu hoàn toàn khi chuỗi có các mệnh đề phụ xen ngang. |
| **Prediction 3 (Sequence Length 50 → 200)** | Tăng $L$ làm tăng thời gian huấn luyện; Simple RNN bị bão hòa hoặc suy giảm do Vanishing Gradient. | Thời gian LSTM tăng từ $2.19\text{s} \to 4.07\text{s}$; F1 của Simple RNN tụt từ $0.5488$ xuống $0.4564$ khi $L=200$. | **Khớp hoàn toàn.** Việc tăng độ dài chuỗi mà không có cơ chế gradient highway khiến Simple RNN hoàn toàn mất phương hướng tối ưu. |
| **Prediction 4 (Truncation tại 30 tokens)** | Truncation quá sớm làm mất thông tin chốt hạ cảm xúc ở cuối review. | Thống kê dataset cho thấy độ dài trung bình là $228$ từ. Ở $L=50$, chỉ $21.75\%$ số token được giữ lại. Nhiều mẫu trong Error Analysis bị đoán sai do từ khóa phán xét nằm ngoài phạm vi cắt tỉa. | **Khớp hoàn toàn.** Phân tích lỗi xác nhận Truncation là nguyên nhân chính gây ra các lỗi ngớ ngẩn ở mẫu E4. |
| **Prediction 5 (LSTM vs RNN trên mọi dataset)** | LSTM không chắc chắn luôn thắng RNN (nguy cơ overfit trên tập nhỏ). | Trên tập dữ liệu ngắn $L=50$, Simple RNN đạt kết quả nhanh và ổn định, dù LSTM đạt F1 cao hơn do dữ liệu IMDb có tính phân hóa phức tạp. Tuy nhiên, LSTM có tới 1.41M tham số (nhiều hơn đáng kể). | **Khớp với phân tích.** Simple RNN học nhanh hơn với ít tham số hơn. |

> **Phát hiện thú vị về Baseline TF-IDF + Logistic Regression:**  
> Nhiều người sẽ bất ngờ khi thấy Baseline TF-IDF + Logistic Regression đạt Accuracy/F1 lên tới **0.8628**, cao hơn Simple RNN và LSTM được huấn luyện từ đầu (from scratch) trong 5 epoch.  
> **Nguyên nhân khoa học:** TF-IDF được tiếp cận **toàn bộ văn bản (Full Document)**, không bị cắt ngắn (truncation) ở 50 hay 100 từ, và tận dụng trực tiếp thống kê tần suất từ toàn cục. Trong khi đó, các mô hình Deep Learning (RNN/LSTM) với 1.4 triệu tham số được khởi tạo ngẫu nhiên cần lượng dữ liệu lớn hơn nhiều (hoặc pre-trained embeddings) và nhiều epoch hơn để có thể vượt qua một bộ phân loại tuyến tính mạnh trên không gian đặc trưng n-gram toàn phần!

---

## 3. Mạch nối Tri thức: Từ LAB 01 đến LAB 05 và Động lực dẫn tới Attention (Section 32)

Chương trình môn học được thiết kế theo một chuỗi tiến hóa logic tuyệt đẹp:

```
LAB 01: Text → TF-IDF → Sparse Representation → Search / Retrieval
   ↓
LAB 02: Text → N-gram → Probability Distribution → Language Model → Perplexity
   ↓
LAB 03: Text → Local Context → Dense Word Embedding (Word2Vec) → Semantic Similarity
   ↓
LAB 04: Text → Static Representation (BoW / Avg W2V) → Linear Classifier → Sentiment Prediction
   ↓
LAB 05: Text → Token Sequence → Sequential Modeling (RNN / LSTM / GRU) → Hidden State / Gates
   ↓  [VẤN ĐỀ NÚT THẮT CỔ CHAI: x_1 → x_2 → ... → x_100 (Thông tin phải đi bộ tuần tự)]
LAB 06: Sequence → Attention Mechanism (Query / Key / Value) → Context-aware Dynamic Representation
   ↓
LAB 07: Attention → Multi-Head Self-Attention → TRANSFORMER Architecture (BERT, GPT)
```

### Tại sao Attention xuất hiện?
Trong RNN/LSTM, để token thứ 100 kết nối được với token thứ 1, vector thông tin phải truyền qua **99 bước chuyển tiếp tuần tự**. Điều này dẫn tới hai vấn đề:
1. Thông tin bị pha loãng và méo mó qua các phép nhân phi tuyến.
2. Không thể tính toán song song các vị trí cùng lúc.

**Câu hỏi định hình tương lai:**  
*Có cách nào để một token có thể trực tiếp "nhìn" và truy cập ngay lập tức vào bất kỳ token quan trọng nào khác trong văn bản mà không cần phải truyền qua chuỗi mắt xích tuần tự hay không?*  
Đó chính là câu trả lời mà **Attention Mechanism (LAB 06)** và **Transformer (LAB 07)** giải quyết thông qua tích vô hướng (Dot-Product Attention).

---

## 4. Trả lời Bộ câu hỏi Kiểm tra Nhận thức Cá nhân (Section 28 — Individual Learning Check)

Dưới đây là lời giải đáp ngắn gọn, súc tích cho các câu hỏi phỏng vấn nhanh 3 phút:

1. **RNN khác Average Word2Vec ở điểm nào?**  
   *Trả lời:* Average Word2Vec lấy trung bình cộng các vector từ, là phép toán giao hoán làm mất hoàn toàn thứ tự từ ($v(A+B) = v(B+A)$). Ngược lại, RNN duy trì trạng thái ẩn phụ thuộc vào chuỗi thời gian ($h_t = f(x_t, h_{t-1})$), bảo toàn cấu trúc cú pháp và vị trí của từng từ trong câu.

2. **Hidden state ($h_t$) có ý nghĩa gì?**  
   *Trả lời:* $h_t$ là vector đại diện cho "trí nhớ ngắn hạn" của mạng tại bước $t$. Nó tổng hợp thông tin của token hiện tại $x_t$ cùng toàn bộ lịch sử ngữ cảnh từ các token trước đó ($t=1 \dots t-1$).

3. **Tại sao RNN có thể gặp vanishing gradient?**  
   *Trả lời:* Trong thuật toán BPTT, gradient truyền ngược qua $T$ bước thời gian chứa tích của $T$ ma trận Jacobian $\prod \operatorname{diag}(1 - h_k^2) W_{hh}^T$. Do đạo hàm hàm tanh $\le 1$, nếu ma trận trọng số có độ lớn không đủ lớn, tích này sẽ suy giảm theo hàm mũ về 0 khi $T$ lớn.

4. **Forget gate của LSTM dùng để làm gì?**  
   *Trả lời:* Forget gate $f_t = \sigma(W_f [x_t, h_{t-1}] + b_f) \in [0, 1]$ quyết định tỷ lệ thông tin từ cell state cũ ($c_{t-1}$) được giữ lại. Nếu $f_t \approx 1$, thông tin được bảo toàn nguyên vẹn; nếu $f_t \approx 0$, thông tin cũ bị xóa bỏ để nhường chỗ cho ngữ cảnh mới.

5. **Tại sao tăng sequence length có thể làm training chậm hơn?**  
   *Trả lời:* Vì thuật toán BPTT phải unroll đồ thị tính toán qua nhiều timestep hơn. Số lượng phép nhân ma trận và tính toán đạo hàm dây chuyền tăng tuyến tính theo độ dài chuỗi $T$, đồng thời tiêu tốn nhiều bộ nhớ RAM/VRAM hơn để lưu lại activation của từng bước.

6. **GRU khác LSTM ở điểm nào?**  
   *Trả lời:* GRU không có Cell State riêng ($c_t$) mà chỉ dùng một Hidden State ($h_t$). GRU gộp cổng quên và cổng vào thành **Update Gate** ($z_t$), đồng thời bổ sung **Reset Gate** ($r_t$). Nhờ đó, GRU có cấu trúc gọn nhẹ hơn, ít tham số hơn 25% so với LSTM và huấn luyện nhanh hơn.

7. **Tại sao model có training loss giảm nhưng validation loss tăng?**  
   *Trả lời:* Đây là biểu hiện kinh điển của hiện tượng **Overfitting (Quá khớp)**. Mô hình bắt đầu ghi nhớ vẹt các đặc trưng nhiễu hoặc các mẫu ngẫu nhiên chỉ xuất hiện trên tập huấn luyện thay vì học các quy luật ngôn ngữ tổng quát, khiến khả năng dự đoán trên tập kiểm thử bị suy giảm.

8. **Nếu một review bị truncate ở 30 tokens, thông tin gì có thể bị mất?**  
   *Trả lời:* Bị mất phần nhận xét kết luận, đánh giá cốt lõi về chất lượng phim (thường nằm ở đoạn kết bài review), các câu chốt hạ cảm xúc, và các mệnh đề đối lập/xoay chuyển cảm xúc xuất hiện ở nửa sau của văn bản.

---

## 5. Tuyên bố Hỗ trợ của AI (AI Assistance Statement — Section 27)

Tuân thủ nghiêm ngặt chính sách học thuật của môn học tại Mục 27:
- **Công cụ sử dụng (Tool):** Google DeepMind Assistant (Antigravity).
- **Mục đích hỗ trợ (Purpose):** Viết boilerplate code huấn luyện PyTorch, tối ưu hóa các hàm vẽ đồ thị trực quan (matplotlib/seaborn), format bảng biểu Markdown, và hỗ trợ kiểm tra tính chính xác của các phép tính ma trận.
- **Phần code do AI sinh (Generated):** Cấu trúc vòng lặp DataLoader, hàm vẽ Confusion Matrix và Learning Curves trong `run_experiments.py`.
- **Phần sinh viên sửa đổi & hoàn thiện (Modified):** Tự tay thiết kế logic lớp `SimpleRNNCell`, triển khai các bài test so sánh với PyTorch framework, tùy chỉnh các siêu tham số (hyperparameters), và xây dựng bộ câu hỏi kiểm thử phụ thuộc xa (Probe test cases).
- **Kiểm chứng độc lập (Verified):** Tự tay tính toán từng bước giải tích trên giấy cho Bài 1 và Bài 5, đối chiếu từng chữ số thập phân với mã nguồn Python, trực tiếp quan sát và lý giải toàn bộ các trường hợp lỗi trong Error Analysis, đảm bảo nắm vững 100% bản chất toán học của các kiến trúc hồi quy.
