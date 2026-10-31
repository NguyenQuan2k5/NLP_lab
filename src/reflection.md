# Báo cáo Đánh giá & Phản ánh (Reflection)

Báo cáo kết quả và phân tích đánh giá mô hình sau khi thực hiện huấn luyện bằng lệnh `python train.py`.

## 1. Kết quả thực nghiệm
- **Số lượng mẫu (Number of samples)**: 80 (Tập huấn luyện: 60 mẫu, Tập kiểm thử: 20 mẫu — phân chia theo tỷ lệ 75/25 có phân tầng stratified)
- **Số lượng lớp nhãn (Number of classes)**: 4 (`cancel_order`, `payment_problem`, `refund`, `track_order` — 20 mẫu/lớp)
- **Thời gian huấn luyện (Training time)**: ~0.012 s
- **Kích thước từ vựng (Vocabulary size)**: 408 đặc trưng (từ vựng TF-IDF kết hợp unigram và bigram)
- **Độ chính xác (Accuracy)**: 0.9000 (90.00% — đúng 18/20 mẫu trên tập kiểm thử)
- **Độ chuẩn xác trung bình (Macro Precision)**: 0.9083 (90.83%)
- **Độ thu hồi trung bình (Macro Recall)**: 0.9000 (90.00%)
- **Chỉ số F1 trung bình (Macro F1)**: 0.8995 (89.95%)
- **Số mẫu dự đoán sai (Test errors)**: 2 / 20 mẫu (tỷ lệ lỗi 10%)

## 2. Nhận xét chi tiết
- **Lớp nào được dự đoán tốt hơn?**
  - Lớp `track_order` đạt kết quả tuyệt đối hoàn hảo (Precision: 1.0000, Recall: 1.0000, F1: 1.0000) do các từ khóa hỏi vị trí, trạng thái bưu kiện ("where", "status", "delivery", "track") rất đặc trưng, không bị trùng lặp với các nhóm intent khác.
  - Lớp `cancel_order` cũng đạt hiệu quả rất cao (Recall: 1.0000, F1: 0.9091) nhờ sự xuất hiện của các động từ hành động dứt khoát ("cancel", "stop", "void").
  - Hai lớp `payment_problem` (F1: 0.8000) và `refund` (F1: 0.8889) có độ chính xác thấp hơn do trường từ vựng liên quan đến tiền bạc, thanh toán ("money", "payment", "charge") dễ gây nhầm lẫn qua lại giữa hai nhóm này.

- **Các lỗi phân loại thường gặp là gì?**
  - **Trùng lặp từ khóa liên ngành (Lexical Overlap)**: Tin nhắn thuộc ý định `refund` ("*I need my payment returned.*") có chứa từ "payment" nên mô hình dự đoán nhầm sang `payment_problem`.
  - **Từ ngữ đa nghĩa tùy ngữ cảnh**: Tin nhắn báo lỗi thanh toán chứa từ "order" ("*I paid but the order still says unpaid.*") bị nhầm thành `cancel_order` do từ "order" xuất hiện với tần suất áp đảo ở các nhãn liên quan đến đơn hàng.
  - Mô hình TF-IDF túi từ (Bag-of-Words) không nắm bắt được thứ tự từ và quan hệ ngữ nghĩa sâu của toàn câu, dẫn đến việc đưa ra quyết định dựa chủ yếu vào sự xuất hiện của các từ khóa đơn lẻ.

- **Dataset có đủ đa dạng để đại diện tin nhắn khách hàng thật không?**
  - Chưa đủ. Đây là tập dữ liệu tổng hợp nhỏ (80 mẫu), các câu có độ dài ngắn, cấu trúc ngữ pháp chuẩn mực và phong cách tương đối đồng đều.
  - Trong thực tế, tin nhắn hỗ trợ khách hàng chứa rất nhiều lỗi gõ phím, viết tắt, từ lóng, câu dài phức tạp chứa đồng thời nhiều ý định (multi-intent), hoặc phụ thuộc chặt chẽ vào ngữ cảnh hội thoại trước đó.

## 3. Hạn chế
Dataset là dữ liệu mẫu nhỏ phục vụ trình diễn bài lab, vì vậy các điểm số có thể biến thiên lớn nếu thay đổi random seed hoặc tỷ lệ phân chia train/test. Kết quả này không phản ánh hiệu năng khi đưa vào môi trường sản xuất thực tế.

## 4. Hướng cải thiện
- Thu thập và gán nhãn thêm dữ liệu hội thoại thực tế đa dạng cách diễn đạt.
- Cân bằng số lượng mẫu giữa các intent và mở rộng tập kiểm thử độc lập.
- Áp dụng phương pháp kiểm chứng chéo (K-Fold Cross-Validation) để đánh giá độ ổn định.
- So sánh và nâng cấp lên các mô hình phân loại mạnh hơn như Linear SVM hoặc các mô hình ngôn ngữ tiền huấn luyện (BERT, RoBERTa, PhoBERT nếu dùng tiếng Việt).
- Khi triển khai thực tế trên Chatbot, cần thiết kế thêm bước xác nhận người dùng (confirmation step) trước khi tự động kích hoạt các thao tác nhạy cảm như hủy đơn hoặc hoàn tiền.
