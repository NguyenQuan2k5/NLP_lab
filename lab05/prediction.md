# Dự đoán Trước Thực nghiệm & Kịch bản Suy luận (Prediction) — LAB 05

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** Sequence Models: RNN, LSTM, and GRU  
**Sinh viên thực hiện:** LAB 05  

Tài liệu này ghi lại các dự đoán giả thuyết (hypotheses) được xây dựng **trước khi tiến hành thực nghiệm**, kèm theo giải thích bản chất toán học/cơ chế mô hình và mức độ tự tin (Confidence). Phần 2 cung cấp mã nguồn Python độc lập để thực hiện suy luận (Inference) và kịch bản dự đoán thử nghiệm trên các mẫu câu đa dạng.

---

## Phần 1: Năm Dự đoán Lý thuyết Trước Thực nghiệm (Section 10 — W5.pdf)

### Prediction 1: Word Order Handling — TF-IDF + Logistic Regression vs. RNN + Embedding
- **Dự đoán (Prediction):** Mô hình `RNN + Embedding` có khả năng xử lý thứ tự từ (word order) trực tiếp và vượt trội hoàn toàn so với `TF-IDF + Logistic Regression`.
- **Cơ chế & Lý do (Reason):**
  - `TF-IDF` hoạt động trên nguyên lý **Bag-of-Words (BoW)**: văn bản được chuyển thành vector đếm tần suất có trọng số của các từ độc lập. Phép biểu diễn này hoàn toàn bỏ qua vị trí của từ trong câu (ví dụ: các câu `"not bad at all"` và `"bad, not at all"` có vector TF-IDF gần như tương đương).
  - Ngược lại, `RNN` cập nhật trạng thái ẩn theo từng bước thời gian:
    $$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
    Trạng thái $h_t$ phụ thuộc trực tiếp vào cả token hiện tại $x_t$ lẫn toàn bộ lịch sử ngữ cảnh $h_{t-1}$. Vì các phép nhân ma trận $W_{hh} W_{xh}$ không có tính giao hoán, thứ tự xuất hiện của từ quyết định trực tiếp quỹ đạo dịch chuyển của $h_t$ trong không gian biểu diễn.
- **Độ tin cậy (Confidence):** **Rất cao (Very High - 99%)**.

---

### Prediction 2: Long-Range Dependency Handling — RNN vs. LSTM
- **Dự đoán (Prediction):** Mô hình `LSTM` có khả năng học và duy trì các mối phụ thuộc tầm xa (long-range dependencies) tốt hơn rất nhiều so với `Simple RNN`.
- **Cơ chế & Lý do (Reason):**
  - Không chỉ vì "LSTM ra đời sau hiện đại hơn", mà bắt nguồn từ kiến trúc truyền dẫn gradient:
  - Ở `Simple RNN`, gradient khi truyền ngược qua $T$ bước thời gian phải chịu tích của $T$ ma trận Jacobian:
    $$\prod_{k=t+1}^{T} \operatorname{diag}(1 - h_k^2) W_{hh}^T$$
    Vì đạo hàm hàm tanh $\le 1$, gradient suy giảm theo hàm mũ về 0 (**Vanishing Gradient**), khiến mạng không thể cập nhật trọng số cho các từ ở đầu chuỗi dài.
  - Ở `LSTM`, trạng thái ô nhớ (Cell State $c_t$) được cập nhật bằng **phép cộng tuyến tính**:
    $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
    Đạo hàm $\frac{\partial c_t}{\partial c_{t-1}} = f_t$. Khi cổng quên $f_t \approx 1$, gradient có thể truyền thẳng qua nhiều timestep mà không bị suy hao qua các hàm phi tuyến co hẹp, duy trì trí nhớ qua chuỗi dài hàng trăm token.
- **Độ tin cậy (Confidence):** **Rất cao (Very High - 98%)**.

---

### Prediction 3: Sequence Length Scaling (50 → 200 Tokens) với Simple RNN
- **Dự đoán (Prediction):** Khi tăng `max_length` từ 50 lên 200 tokens:
  1. Thời gian huấn luyện (Training time) của RNN sẽ **tăng xấp xỉ tuyến tính (gấp ~3-4 lần)**.
  2. Hiện tượng Vanishing Gradient ở Simple RNN sẽ **trở nên trầm trọng hơn**.
  3. Mức độ cải thiện độ chính xác (Accuracy/F1) của Simple RNN sẽ **chững lại (bão hòa) hoặc giảm nhẹ**, trong khi LSTM/GRU sẽ tận dụng được ngữ cảnh bổ sung để tăng F1.
- **Cơ chế & Lý do (Reason):**
  - Thuật toán Backpropagation Through Time (BPTT) phải duỗi (unroll) mô hình qua 200 timestep thay vì 50, đòi hỏi số lượng phép tính forward và backward tăng tỷ lệ thuận với $T$.
  - Khoảng cách truyền ngược gradient dài gấp 4 lần khiến gradient suy biến cực nhanh ở Simple RNN, dẫn đến việc mạng gặp khó khăn khi tối ưu hóa trên chuỗi 200 bước so với chuỗi 50 bước.
- **Độ tin cậy (Confidence):** **Cao (High - 90%)**.

---

### Prediction 4: Tác động khi Truncate Toàn bộ Review về 30 Tokens
- **Dự đoán (Prediction):** Việc cắt ngắn cứng toàn bộ bài đánh giá về 30 tokens sẽ **làm suy giảm đáng kể hiệu năng phân loại (giảm từ 5% - 15% F1)** so với khi giữ độ dài 100-200 tokens.
- **Cơ chế & Lý do (Reason):**
  - Bài đánh giá phim trên IMDb thường có độ dài trung bình từ 150 đến 250 từ. Người viết review thường bắt đầu bằng việc tóm tắt nội dung cốt truyện hoặc bối cảnh phim (dễ lẫn lộn từ ngữ trung tính), và chỉ đưa ra nhận xét chốt hạ cảm xúc ("In conclusion, this was an absolute masterpiece" hoặc "Don't waste your time") ở **phần giữa hoặc cuối bài review**.
  - Truncation về 30 tokens sẽ cắt bỏ hoàn toàn phần kết luận cốt lõi này, khiến mô hình chỉ nhận được các câu mở đầu chung chung, dẫn đến mất mát thông tin mang tính quyết định.
- **Độ tin cậy (Confidence):** **Rất cao (Very High - 95%)**.

---

### Prediction 5: LSTM có chắc chắn cho Accuracy cao hơn RNN trên mọi Dataset không?
- **Dự đoán (Prediction):** **KHÔNG**. LSTM **không chắc chắn** luôn đạt accuracy cao hơn Simple RNN trên mọi bài toán và mọi tập dữ liệu.
- **Cơ chế & Lý do (Reason):**
  - LSTM có số lượng tham số gấp khoảng **4 lần** Simple RNN (do phải duy trì 4 ma trận cổng $f, i, \tilde{c}, o$).
  - Trên các tập dữ liệu có quy mô rất nhỏ (vài chục đến vài trăm mẫu) hoặc văn bản cực ngắn (tweet 5-10 từ):
    1. Chuỗi ngắn khiến Simple RNN **không bị ảnh hưởng bởi vanishing gradient**.
    2. Số lượng tham số quá lớn của LSTM dễ dẫn đến hiện tượng **Overfitting** (mô hình học vẹt tập train nhưng kém trên test).
    3. Simple RNN với kiến trúc đơn giản, ít tham số hơn sẽ đóng vai trò như một bộ điều chuẩn tự nhiên (inductive bias), huấn luyện nhanh và có thể cho kết quả tổng quát hóa (generalization) tốt hơn hoặc tương đương LSTM.
- **Độ tin cậy (Confidence):** **Cao (High - 92%)**.

---

## Phần 2: Mã nguồn Thực thi Suy luận Dự đoán Mẫu (Inference Code)

Dưới đây là module Python độc lập cho phép nạp mô hình tuần tự đã huấn luyện (LSTM/GRU) và thực hiện phân loại cảm xúc cho văn bản tùy ý:

```python
import torch
import torch.nn as nn
import re
import json

def clean_text(text: str) -> str:
    """Làm sạch văn bản: bỏ tag HTML, đưa về chữ thường và tách từ."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def predict_sentiment(text: str, model, vocab, max_length: int = 100, device: str = 'cpu'):
    """
    Hàm suy luận phân loại cảm xúc (Sentiment Inference Pipeline)
    
    Quy trình:
        Raw Text -> Cleaning -> Tokenization -> Token IDs 
        -> Padding/Truncation -> Model Forward -> Softmax -> Prediction
    """
    model.eval()
    tokens = clean_text(text).split()
    
    # Chuyển token thành ID dựa trên từ vựng (vocab)
    unk_idx = vocab.get('<UNK>', 1)
    pad_idx = vocab.get('<PAD>', 0)
    token_ids = [vocab.get(token, unk_idx) for token in tokens]
    
    # Truncation hoặc Padding
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids = token_ids + [pad_idx] * (max_length - len(token_ids))
        
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_label_idx = torch.argmax(probs).item()
        
    labels = ["Negative", "Positive"]
    return {
        "text": text,
        "tokens_count": len(tokens),
        "prediction": labels[pred_label_idx],
        "positive_probability": probs[1].item(),
        "negative_probability": probs[0].item()
    }
```

---

## Phần 3: Kịch bản Thử nghiệm Dự đoán trên các Mẫu câu Khác nhau (Section 26)

Để kiểm chứng tính linh hoạt của mô hình tuần tự, chúng ta thiết lập 4 kịch bản kiểm thử:

| Kịch bản | Câu thử nghiệm (Input text) | Nhãn kỳ vọng | Hiện tượng ngôn ngữ cần quan sát |
| :--- | :--- | :---: | :--- |
| **1. Câu ngắn đơn giản** | `"I really enjoyed this wonderful movie."` | **Positive** | Nhận diện tính từ tích cực rõ ràng (`enjoyed`, `wonderful`). |
| **2. Câu có phủ định (Negation)** | `"The plot was not good at all, completely disappointing."` | **Negative** | Cụm phủ định `"not good"` phải lấn át từ `"good"`. |
| **3. Mệnh đề đối lập (Contrast)** | `"I thought the movie would be terrible, but it was surprisingly good."` | **Positive** | Nhận diện bước ngoặt cảm xúc sau liên từ `"but"`, gạt bỏ ấn tượng ban đầu `"terrible"`. |
| **4. Phụ thuộc xa (Long Context)** | `"The acting was, despite numerous flawed scenes and weak pacing, ultimately outstanding."` | **Positive** | Trích xuất từ khóa chốt hạ `"ultimately outstanding"` qua một mệnh đề xen ngang dài mang nghĩa tiêu cực. |
