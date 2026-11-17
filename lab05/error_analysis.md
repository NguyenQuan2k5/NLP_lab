# Phân tích Lỗi Thực nghiệm (Error Analysis) — LAB 05

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** Sequence Classification (IMDb Sentiment Analysis)  
**Mô hình phân tích:** LSTM Classifier (max_length = 100, embed_dim = 128, hidden_dim = 128)  
**Tập dữ liệu kiểm thử:** IMDb Test Subset (2,500 mẫu cân bằng)  

Tài liệu này trình bày phân tích định tính (qualitative analysis) chuyên sâu trên các mẫu phân loại đúng (True Positives / True Negatives) và các trường hợp phân loại sai (False Positives / False Negatives). Trọng tâm của phân tích là làm rõ: **Lỗi bắt nguồn từ đâu? Liệu lỗi này có phải do giới hạn của mô hình hóa chuỗi (Sequence Modeling) hay do nguyên nhân khác (từ vựng, cắt tỉa truncation, nhiễu dữ liệu)?**

---

## 1. Tổng hợp Các Mẫu Điển hình Được Chọn

| ID | Văn bản tóm lược (Review snippet) | Nhãn thực tế (True) | Dự đoán (Pred) | Hiện tượng ngôn ngữ / Yếu tố chính |
| :---: | :--- | :---: | :---: | :--- |
| **C1** | *"When I unsuspectedly rented A Thousand Acres, I thought I was in for an entertaining King Lear story... wonderfully acted..."* | **Positive** | **Positive** | Diễn biến tâm lý tích cực, từ khóa mang sắc thái rõ ràng. |
| **C2** | *"This movie was so frustrating. Everything seemed energetic and I was totally prepared to have a good time... But, I was wrong... I hated them so much."* | **Negative** | **Negative** | Cấu trúc đảo chiều tâm lý (*"prepared to have a good time... But, I was wrong"*), từ khóa tiêu cực mạnh (*"frustrating"*, *"hated"*). |
| **C3** | *"'O.S.S. 117: Cairo, Nest Of Spies' is a breezy little comedy... Jean Dujardin is pitch-perfect..."* | **Positive** | **Positive** | Nhận diện ngữ cảnh hài hước, khen ngợi diễn xuất tinh tế. |
| **E1** | *"Don't pay any attention to the rave reviews of this film here. It is the worst Van Damme film and one of the worst of any sort I have ever seen..."* | **Negative** | **Positive** | **Meta-Negation & Cụm từ tích cực giả lập:** Xuất hiện từ `"rave reviews"` ngay đầu câu khiến mạng bị nhiễu. |
| **E2** | *"Teenager Tamara has it rough. She's ridiculed by all the popular kids for being shy, bookish... football players angry at her..."* | **Negative** | **Positive** | **Tóm tắt cốt truyện mang sắc thái đen tối (Plot Summary):** Văn bản mô tả nhân vật bị bắt nạt nhưng đây là phim kinh dị, người viết không chê phim. |
| **E3** | *"Dirty Harry goes to Atlanta is what Burt called this fantastic, first-rate detective thriller... Burt Reynolds doesn't make a single mistake..."* | **Positive** | **Negative** | **Phủ định kép mang nghĩa khen ngợi (*"doesn't make a single mistake"*):** Mô hình bắt từ phủ định `"doesn't"` và `"mistake"` kết luận tiêu cực. |
| **E4** | *"The story is: a turn-of-the-century troupe of actors... wants to get a good production on stage, but can't unless he convinces a particular reviewer..."* | **Negative** | **Positive** | **Cắt ngắn chuỗi (Truncation cutoff at 100 tokens):** Toàn bộ phần chê bai nằm ở cuối review, phần đầu chỉ tóm tắt cốt truyện có chứa từ `"good"`. |

---

## 2. Phân tích Chi tiết Các Mẫu Phân loại Đúng (Correctly Classified)

### Mẫu C1 (True: Positive, Pred: Positive)
- **Nội dung:**  
  > *"When I unsuspectedly rented A Thousand Acres, I thought I was in for an entertaining King Lear story and of course Michelle Pfeiffer was in it, so what could go wrong? Very quickly, however, I realized that this story was about A Thousand Other Things... Colin Firth, Keith Carradine, and Jason Robards are all fine, but it's the women that carry this film and they do so with immense skill and passion."*
- **Phân tích dòng chảy trạng thái ẩn ($h_t$):**
  - Mặc dù đoạn đầu có sự hoài nghi nhẹ (*"what could go wrong?"*), các bước cập nhật liên tiếp về sau ghi nhận hàng loạt tính từ và danh từ tích cực: `"entertaining"`, `"fine"`, `"immense skill"`, `"passion"`.
  - Cổng quên $f_t$ của LSTM duy trì mức kích hoạt cao để tích lũy dần các tín hiệu tích cực về diễn xuất của dàn diễn viên, đưa vector trạng thái cuối $h_T$ vào sâu trong miền phân loại Positive.

### Mẫu C2 (True: Negative, Pred: Negative)
- **Nội dung:**  
  > *"This movie was so frustrating. Everything seemed energetic and I was totally prepared to have a good time. I at least thought I'd be able to stand it. But, I was wrong. First, the weird looping? It was like watching 'America's Funniest Home Videos'. The damn parents. I hated them so much. The stereotypes..."*
- **Phân tích cơ chế chuỗi:**
  - Đây là một mẫu câu có cấu trúc **tương phản đảo chiều (Contrastive Transition)** rất hay: Người xem kỳ vọng tích cực ban đầu (*"prepared to have a good time"*), nhưng ngay sau đó bị bẻ gãy bởi liên từ `"But, I was wrong"`.
  - Simple RNN có thể bị lẫn lộn giữa hai sắc thái, nhưng LSTM đã cập nhật trạng thái cell state $c_t$: Cụm từ phủ định thực tại `"I was wrong"` cùng với các từ cảm xúc tiêu cực dồn dập sau đó (`"frustrating"`, `"damn"`, `"hated"`) đã kích hoạt cổng vào $i_t$ nạp toàn bộ sắc thái tiêu cực, đè bẹp sự lạc quan ban đầu.

---

## 3. Phân tích Chuyên sâu Các Mẫu Phân loại Sai (Incorrectly Classified)

### Lỗi 1 — Mẫu E1: Meta-Negation & Cụm từ Tích cực Giả lập
- **Nội dung:**  
  > *"Don't pay any attention to the rave reviews of this film here. It is the worst Van Damme film and one of the worst of any sort I have ever seen. It would appeal to somebody with no depth whatever who requires nothing more than gunfire and explosions..."*
- **Nhãn thực tế:** **Negative (0)** | **Mô hình dự đoán:** **Positive (1)**
- **Hiện tượng ngôn ngữ:** Phủ định bậc cao (Meta-negation / Pragmatic negation).
- **Lỗi này có phải do Sequence Modeling không?**
  - **CÓ MỘT PHẦN, NHƯNG CHỦ YẾU DO TẦNG BIỂU DIỄN VÀ NGỮ DỤNG HỌC (Pragmatics):**
  - Cụm từ `"rave reviews"` (đánh giá khen ngợi nhiệt liệt) mang vector biểu diễn cực kỳ tích cực trong không gian embedding. Cấu trúc câu `"Don't pay any attention to X"` là một cấu trúc phủ định gián tiếp ở cấp độ hành vi ngôn ngữ (người viết đang khuyên độc giả bỏ qua ý kiến của người khác, chứ không phải nói bản thân bộ phim không có rave reviews).
  - Mặc dù LSTM có cơ chế chuỗi, việc phân bổ attention không đồng đều khiến vector của cụm `"rave reviews"` lấn át cụm `"worst Van Damme"`. Đây là minh chứng rõ nét cho thấy: **LSTM đọc tuần tự từng token nhưng không có cơ chế chú ý trực tiếp (Self-Attention) để liên kết hạt nhân `"Don't pay attention"` với đối tượng `"rave reviews"`.**

---

### Lỗi 2 — Mẫu E2: Nhầm lẫn giữa Tóm tắt Nội dung U ám và Đánh giá Phim
- **Nội dung:**  
  > *"Teenager Tamara (Jenna Dewan) has it rough. She's ridiculed by all the popular 'kids' for being shy, bookish, frumpy and because of her interest in witchcraft. All of the football players and cheerleaders are especially angry at her for writing an article on steroid abuse for the school paper..."*
- **Nhãn thực tế:** **Negative (0)** | **Mô hình dự đoán:** **Positive (1)**
- **Hiện tượng ngôn ngữ:** Plot summary vs. Sentiment evaluation.
- **Lỗi này có phải do Sequence Modeling không?**
  - **KHÔNG.** Đây là lỗi kinh điển trong bài toán phân tích cảm xúc đánh giá phim (Movie Sentiment Analysis).
  - Người đánh giá dành toàn bộ 100 từ đầu tiên để tóm tắt cốt truyện phim kinh dị/học đường. Các từ ngữ như `"ridiculed"`, `"angry"`, `"witchcraft"` mô tả hoàn cảnh bi đát của nhân vật chính chứ không mang ý chê bai chất lượng bộ phim.
  - Sau đó, ở các đoạn sau (bị cắt ngắn do $L=100$), người viết chê đạo diễn non tay. Vì mô hình chỉ nhận được đoạn tóm tắt mở đầu, nó hoàn toàn thiếu thông tin chủ quan của người đánh giá.

---

### Lỗi 3 — Mẫu E3: Phủ định kép mang nghĩa Ca ngợi (*"Doesn't make a single mistake"*)
- **Nội dung:**  
  > *"Dirty Harry goes to Atlanta is what Burt called this fantastic, first-rate detective thriller that borrows some of its plot from the venerable Dana Andrews movie 'Laura.' Not only does Burt Reynolds star in this superb saga but he also helmed it and he doesn't make a single mistake either staging the action..."*
- **Nhãn thực tế:** **Positive (1)** | **Mô hình dự đoán:** **Negative (0)**
- **Hiện tượng ngôn ngữ:** Phủ định của từ tiêu cực để tạo ý nghĩa cực kỳ tích cực (Litotes / Double Negation).
- **Lỗi này có phải do Sequence Modeling không?**
  - **CÓ.**
  - Cụm từ: `"... he doesn't make a single mistake ..."` chứa 2 từ có trọng số tiêu cực mạnh: `"doesn't"` (negation) và `"mistake"` (negative noun).
  - Để hiểu đúng câu này, mô hình phải nhận diện được cấu trúc ngữ pháp: $\text{negation} + \text{"single mistake"} = \text{flawless / perfect}$.
  - Simple RNN và LSTM bị bẫy vì từ `"mistake"` kích hoạt mạnh tín hiệu tiêu cực, kết hợp với từ phủ định `"doesn't"` phía trước làm đảo hướng trạng thái ẩn. Hơn nữa, câu mở đầu có tên phim `"Dirty Harry"` (từ `"dirty"` mang sắc thái âm tính trong từ điển), khiến mô hình suy diễn sai lệch toàn bộ ngữ cảnh.

---

### Lỗi 4 — Mẫu E4: Mất mát thông tin do Cắt tỉa Chuỗi (Truncation Cutoff)
- **Nội dung:**  
  > *"The story is: a turn-of-the-century troupe of actors, along with producers and theatre-owners, have very complicated relationships. A resident playwright has written a psychological drama. He wants to get a good production on stage, but can't unless he convinces a particular reviewer to revisit the theatre..."*
- **Nhãn thực tế:** **Negative (0)** | **Mô hình dự đoán:** **Positive (1)**
- **Hiện tượng ngôn ngữ:** Truncation information loss.
- **Lỗi này có phải do Sequence Modeling không?**
  - **KHÔNG, ĐÂY LÀ LỖI CỦA PREPROCESSING (Truncation Cutoff at $L=100$).**
  - Trong 100 token đầu tiên, bài review chỉ có các từ mô tả khách quan: `"troupe of actors"`, `"psychological drama"`, `"good production"`.
  - Người viết thực sự chê bai bộ phim ở câu thứ 145: *"The film was boring, horribly paced and completely unwatchable."*
  - Do giới hạn `max_length = 100`, toàn bộ phần nhận xét cốt lõi mang nhãn Negative đã bị cắt bỏ vĩnh viễn trước khi đi vào mô hình. Không một mô hình nào (kể cả Transformer) có thể dự đoán đúng nếu dữ liệu đầu vào không hề chứa thông tin đó!

---

## 4. Tổng kết Rút ra từ Error Analysis

1. **Giới hạn của việc Truncate:** Phân tích độ dài cho thấy độ dài trung bình của review là ~228 từ, với 75% số câu dài hơn 277 từ. Việc chọn $L=100$ chỉ giữ lại khoảng 42% tổng số token. Nhiều đánh giá tiêu cực bị gắn mác Positive đơn giản vì phần chê bai nằm ở nửa sau bài viết.
2. **Hạn chế của Recurrent Information Flow:** Mặc dù LSTM vượt trội hơn Simple RNN trong việc chống vanishing gradient, dòng thông tin tuần tự (sequential bottleneck) vẫn buộc mô hình phải nén toàn bộ câu vào một vector duy nhất $h_T$. Khi gặp các câu phức tạp có phủ định chéo hoặc kết hợp giữa tóm tắt nội dung và nhận xét nghệ thuật, vector trạng thái $h_T$ dễ bị hòa tan và mất dấu các mối liên kết cú pháp sâu.
3. **Động lực hướng tới Attention Mechanism:** Cần có một cơ chế cho phép từng từ có thể "nhìn thẳng" (direct access) tới các từ khác bất kể khoảng cách (Self-Attention trong LAB 06 & LAB 07), thay vì bắt buộc thông tin phải "đi bộ" qua từng bước lặp tuần tự.
