# LAB 06 — Attention Mechanism: Comprehensive Qualitative Error Analysis

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (Natural Language Processing)  
**Học kỳ:** Học kỳ I - 2026 | **Đại học Khoa học Tự nhiên - ĐHQGHN**  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** *From Recurrent State to Context-Aware Representation*  

---

## 1. Giới thiệu và Nguyên tắc Phân tích

Mục tiêu của tệp phân tích lỗi định tính này là đi sâu vào bản chất hành vi biểu diễn của mô hình **Self-Attention Context-Aware Classifier** đã được huấn luyện trên tập dữ liệu IMDb (đo lường tại Section 24, 26).

Theo yêu cầu nghiêm ngặt tại **Section 27** của đề cương môn học:
1. Lựa chọn tối thiểu **3 mẫu dự đoán đúng (Correct Cases)** và **3 mẫu dự đoán sai (Incorrect Cases)** tiêu biểu từ tập kiểm thử độc lập (Test Set).
2. Phân loại nguyên nhân lỗi theo bảng phân loại chuẩn mực:
   - `negation` (phủ định, cấu trúc đảo chiều ngữ nghĩa).
   - `long-range dependency` (phụ thuộc tầm xa giữa các từ cách xa nhau).
   - `ambiguous wording` (ngữ nghĩa mơ hồ, đa nghĩa, từ mang sắc thái kép).
   - `sarcasm` (châm biếm, mỉa mai ngược nghĩa).
   - `rare word` (từ hiếm gặp, out-of-vocabulary hoặc sub-word không phổ biến).
   - `domain shift` (chuyển dịch ngữ cảnh / phong cách văn phong).
   - `truncation` (cắt cụt chuỗi vượt quá `max_length = 100`).
   - `noise` (nhiễu ký tự, lỗi chính tả, câu lủng củng).
3. Với mô hình Attention, sinh viên phải kiểm tra: *Attention weights có thực sự tập trung vào các token mà con người dự đoán là quan trọng hay không?*
4. **Nguyên tắc phương pháp luận khoa học:** Tuyệt đối không được quy kết chủ quan: *"Attention nhìn sai từ nên mô hình dự đoán sai"*. Hiện tượng này phải được định nghĩa chuẩn xác như một **giả thuyết (hypothesis)**, sau đó đối chiếu đa chiều giữa cơ chế pooling, ma trận trọng số và không gian biểu diễn của mô hình.

---

## 2. Bảng Tổng hợp Các Mẫu Phân tích

| ID | Test Index | True Label | Pred Label | Độ tự tin ($P$) | Phân loại Lỗi / Hiện tượng | Kết quả |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **C1** | #4 | Negative | Negative | 98.08% | Rõ ràng, từ vựng tiêu cực dứt khoát ("waste of time", "no script") | **ĐÚNG** |
| **C2** | #11 | Negative | Negative | 99.81% | Đánh giá trực diện, tính từ phân cực mạnh ("worst", "crappy") | **ĐÚNG** |
| **C3** | #33 | Negative | Negative | 96.30% | Phụ thuộc xa & Đảo chiều cảm xúc ("disappointed", "unlike") | **ĐÚNG** |
| **E1** | #108 | Negative | Positive | 97.86% | `sarcasm` (Mỉa mai, khen đểu "most hilarious", "put Carrey out of job") | **SAI** |
| **E2** | #6 | Positive | Negative | 58.14% | `ambiguous wording` (Cảm xúc bi kịch tích cực "sad", "touching", "abuse") | **SAI** |
| **E3** | #32 | Positive | Negative | 86.79% | `negation` & Cụm từ thành ngữ ("too bad that nobody knows", "low-budget") | **SAI** |

---

## 3. Phân tích Chi tiết 3 Trường hợp Dự đoán ĐÚNG (Correct Cases)

### Mẫu C1 (Test Index #4)
- **Văn bản gốc:**  
  > *"This movie spends most of its time preaching that it is the script that makes the movie, but apparently there was no script when they shot this waste of time! The trailer makes this out to be a comedy..."*
- **Độ dài chuỗi:** 89 tokens | **Nhãn thực tế:** `Negative` | **Dự đoán:** `Negative` ($P = 98.08\%$).
- **Phân tích cơ chế Attention:**
  - Mô hình dồn trọng số pooling cao vào các token `waste`, `time`, `no`, `script`.
  - Cơ chế Self-Attention kết nối từ `no` với `script`, triệt tiêu khả năng gán nhãn tích cực thông thường của từ "script".
- **Lý do thành công:** Cấu trúc câu có điểm nhấn phủ định rõ ràng và từ ngữ phân cực âm tính mạnh mẽ xuất hiện ở vị trí trung tâm, giúp vector ngữ cảnh $c$ định hình vững chắc trong vùng không gian phân loại Negative.

---

### Mẫu C2 (Test Index #11)
- **Văn bản gốc:**  
  > *"OK, it was a risky move to rent this flick, but I thought I had nothing to lose. Well, I was wrong. This is, next to Bloodsurf, the worst horrormovie I have ever seen. Crappy actors, crappy technique..."*
- **Độ dài chuỗi:** 117 tokens (cắt ở 100) | **Nhãn thực tế:** `Negative` | **Dự đoán:** `Negative` ($P = 99.81\%$).
- **Phân tích cơ chế Attention:**
  - Attention weights phân bổ đậm đặc vào các tính từ chỉ trích nặng nề: `worst`, `crappy`, `wrong`.
  - Khác với RNN dễ bị phân tâm bởi câu mào đầu trung tính ("risky move", "nothing to lose"), Self-Attention cho phép các token cuối câu tương tác trực tiếp với phân lớp, đưa độ tự tin tiêu cực lên tới $99.81\%$.
- **Lý do thành công:** Tín hiệu cảm xúc dồi dào, từ vựng chỉ trích rõ ràng, vượt qua mọi sự nhiễu loạn của ngữ cảnh dẫn nhập.

---

### Mẫu C3 (Test Index #33)
- **Văn bản gốc:**  
  > *"Unlike Terms of Endearment and Steel Magnolia's, I left the movie theater feeling VERY disappointed. I started to get into the characters and their complex mother/daughter and father/daughter relationships..."*
- **Độ dài chuỗi:** 76 tokens | **Nhãn thực tế:** `Negative` | **Dự đoán:** `Negative` ($P = 96.30\%$).
- **Phân tích cơ chế Attention:**
  - Token `disappointed` nhận được trọng số cực lớn ($> 30\%$).
  - Self-Attention liên kết `disappointed` với mệnh đề đối chiếu `unlike` ở đầu câu.
- **Lý do thành công:** Mô hình vượt qua được bẫy các tên phim kinh điển được khen ngợi ở vế đầu để nắm bắt được trạng thái thất vọng của người xem ở vế sau.

---

## 4. Phân tích Chuyên sâu 3 Trường hợp Dự đoán SAI (Incorrect Cases)

### Mẫu E1 (Test Index #108) — Lỗi Mỉa mai, Châm biếm (`sarcasm`)
- **Văn bản gốc:**  
  > *"This is, without a doubt, the most hilarious movie I've ever seen. Seriously, if the makers of this movie are ever discovered, they'll put guys like Jim Carrey out of a job. Rent Jack-O tonight! Believe me..."*
- **Độ dài chuỗi:** 41 tokens | **Nhãn thực tế:** `Negative` (Đánh giá phim kinh dị hạng B dở tệ) | **Dự đoán của mô hình:** `Positive` ($P = 97.86\%$).
- **Phân phối Attention quan sát được:**
  - Attention phân bổ cao vào: `hilarious` (28.4%), `without` (14.2%), `job` (11.5%), `discovered` (9.8%).
- **Kiểm tra câu hỏi của Section 27:**
  - *Attention có tập trung vào token mà con người dự đoán là quan trọng không?*  
    **Có.** Đối với người đọc bề mặt, các từ như "most hilarious", "put Jim Carrey out of a job" hiển nhiên là các từ tích cực mạnh nhất. Attention đã tập trung chính xác vào cụm từ "hilarious".
  - *Điều này có thể giải thích dự đoán sai không? Trình bày dưới dạng Giả thuyết (Hypothesis):*  
    > **Giả thuyết H1 (Semantic Inversion Failure):** Mô hình Attention một tầng không có cơ chế suy luận logic phản thực tế (counterfactual reasoning) hay phát hiện ngữ điệu mỉa mai. Mô hình chỉ thực hiện phép chiếu tuyến tính và tích vô hướng từ vựng. Khi người đánh giá dùng từ ngữ khen ngợi tột bậc ("most hilarious", "put Jim Carrey out of a job") để châm biếm một bộ phim rác, vector ngữ cảnh $c$ bị kéo lệch hoàn toàn về phía cụm biểu diễn tích cực. Vì vậy, việc Attention tập trung đúng vào từ "hilarious" lại chính là nguyên nhân trực tiếp dẫn tới dự đoán sai, do từ này mang nghĩa đảo ngược so với nhãn thực tế.

---

### Mẫu E2 (Test Index #6) — Ngữ nghĩa Mơ hồ / Cảm xúc Bi kịch (`ambiguous wording`)
- **Văn bản gốc:**  
  > *"This is a really sad, and touching movie! It deals with the subject of child abuse. It's really sad, but mostly a true story, because it happens everyday. Elijah Wood and Joseph Mazzello play the two brothers..."*
- **Độ dài chuỗi:** 119 tokens | **Nhãn thực tế:** `Positive` (Khán giả xúc động trước một tác phẩm nhân văn sâu sắc) | **Dự đoán của mô hình:** `Negative` ($P = 58.14\%$).
- **Phân phối Attention quan sát được:**
  - Attention tập trung vào: `abuse` (24.1%), `sad` (21.7%), `child` (12.3%), `touching` (7.2%).
- **Kiểm tra câu hỏi của Section 27:**
  - *Attention có tập trung vào token mà con người dự đoán là quan trọng không?*  
    **Không hoàn toàn.** Con người nhận ra "touching" (cảm động) là từ định đoạt thái độ tích cực của người viết đối với bộ phim, trong khi "sad" và "child abuse" chỉ là mô tả đề tài kịch bản. Nhưng Attention lại bị hút vào các từ tiêu cực nặng như "abuse" và "sad".
  - *Điều này có thể giải thích dự đoán sai không? Trình bày dưới dạng Giả thuyết (Hypothesis):*  
    > **Giả thuyết H2 (Negative Dominance in Affective Space):** Trong kho ngữ liệu đánh giá phim, các từ vựng như "abuse", "sad", "horrible" có tần suất xuất hiện áp đảo trong các bài phê bình 1-2 sao. Khi chiếu vào không gian nhúng, vector biểu diễn của "child abuse" mang biên độ tiêu cực cực mạnh, lấn át từ "touching". Cơ chế Attention tính tổng có trọng số phẳng (linear weighted combination) đã cộng dồn độ tiêu cực của hai lần xuất hiện từ "sad" cộng với "abuse", làm chìm lấp sắc thái tán thưởng của người xem. Đây không đơn thuần là lỗi nhìn sai vị trí, mà là sự thiếu hụt khả năng phân tách giữa **"chủ đề của bộ phim" (movie subject)** và **"đánh giá chất lượng của người xem" (viewer appraisal)**.

---

### Mẫu E3 (Test Index #32) — Phủ định Thành ngữ & Cấu trúc Tiếc nuối (`negation`)
- **Văn bản gốc:**  
  > *"It's really too bad that nobody knows about this movie. I think if it were just spruced up a little and if it weren't so low-budget, I think one of the major film companies might have wanted to take it..."*
- **Độ dài chuỗi:** 95 tokens | **Nhãn thực tế:** `Positive` (Khen phim hay và tiếc vì phim ít người biết đến) | **Dự đoán của mô hình:** `Negative` ($P = 86.79\%$).
- **Phân phối Attention quan sát được:**
  - Attention phân bổ cao vào: `bad` (27.8%), `nobody` (18.5%), `lowbudget` (15.2%).
- **Kiểm tra câu hỏi của Section 27:**
  - *Attention có tập trung vào token mà con người dự đoán là quan trọng không?*  
    **Không.** Con người hiểu cụm thành ngữ *"It's really too bad that..."* là biểu thị sự nuối tiếc cho một tác phẩm hay bị lãng quên (underrated gem). Tuy nhiên, Attention đã tách rời các từ đơn lẻ và gán trọng số lớn cho token "bad" và "nobody".
  - *Điều này có thể giải thích dự đoán sai không? Trình bày dưới dạng Giả thuyết (Hypothesis):*  
    > **Giả thuyết H3 (Idiomatic Compositionality Breakdown):** Cơ chế Self-Attention đơn tầng với kích thước từ vựng unigram không có khả năng nắm bắt trọn vẹn ngữ nghĩa thành ngữ đa từ (Multi-word Expressions / Idioms). Token "bad" trong ngữ cảnh này mang vai trò cú pháp phụ thuộc trong mệnh đề thành ngữ, nhưng lại có vector nhúng mang giá trị định hướng tiêu cực cao. Do thiếu cơ chế Multi-Head Attention chuyên biệt để phát hiện quan hệ cú pháp và thiếu Positional Encoding tinh vi, mô hình không thể nhận diện được cấu trúc ngữ pháp đảo ngược ngữ nghĩa này.

---

## 5. Kết luận Phương pháp luận từ Error Analysis

1. **Attention không tương đương với Khả năng Suy luận (Reasoning):**  
   Kết quả phân tích 6 mẫu trên khẳng định rằng trọng số Attention cao chỉ phản ánh sự cộng hưởng toán học giữa Query và Key trong không gian nhúng của tập dữ liệu huấn luyện. Việc mô hình chú ý vào một từ không đồng nghĩa với việc mô hình "hiểu" ngữ pháp hay logic ngữ cảnh của câu.
2. **Hạn chế của phép cộng tuyến tính $c = \sum \alpha_i v_i$:**  
   Sau khi tính toán attention weights, bước tổng hợp Context Vector vẫn là một phép tổ hợp lồi tuyến tính. Phép toán này gặp khó khăn nghiêm trọng khi phải xử lý các câu có cảm xúc xung đột (mixed sentiment), mỉa mai (`sarcasm`), hoặc các quan hệ điều kiện ("if it were...").
3. **Định hướng sang LAB 07 (Transformer Block):**  
   Để khắc phục các lỗi trên, hệ thống cần nhiều Attention Heads độc lập (Multi-Head Attention) để một head tập trung vào cú pháp, một head bắt quan hệ phủ định, kết hợp với các tầng Feed-Forward phi tuyến tính sâu (MLP) và Layer Normalization.
