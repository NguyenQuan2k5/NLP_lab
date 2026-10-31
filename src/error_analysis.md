# Phân tích Lỗi (Error Analysis)

Số lượng mẫu kiểm thử phân loại sai: 2 / 20.

Tập dữ liệu này có quy mô nhỏ phục vụ minh họa. Hãy xem các lỗi dưới đây như ví dụ phân tích trực quan về sự trùng lặp từ vựng, không coi đây là ước lượng chuẩn mực tuyệt đối trong môi trường thực tế.

## Ví dụ 1
- **Ý định thực tế (True intent)**: `payment_problem`
- **Ý định dự đoán (Predicted intent)**: `cancel_order`
- **Nội dung tin nhắn (Text)**: "I paid but the order still says unpaid."

## Ví dụ 2
- **Ý định thực tế (True intent)**: `refund`
- **Ý định dự đoán (Predicted intent)**: `payment_problem`
- **Nội dung tin nhắn (Text)**: "I need my payment returned."
