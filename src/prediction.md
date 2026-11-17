# Ví dụ Dự đoán Ý định (Prediction Examples)

Sau khi hoàn tất bước huấn luyện bằng lệnh `python train.py`, tệp mô hình `intent_model.joblib` sẽ được lưu lại. Dưới đây là đoạn mã Python mẫu để tải mô hình và thực hiện dự đoán ý định cho các tin nhắn mới:

```python
import joblib

# 1. Tải mô hình đã lưu
model = joblib.load("intent_model.joblib")

# 2. Danh sách các tin nhắn cần phân loại
messages = [
    "I want my money back, please.",          # Yêu cầu hoàn tiền
    "Please cancel my order.",                # Yêu cầu hủy đơn
    "Can you tell me where my package is?",   # Tra cứu vị trí đơn hàng
    "My payment was declined."                # Lỗi thanh toán bị từ chối
]

# 3. Thực hiện dự đoán
predictions = model.predict(messages)

# 4. In kết quả
for message, intent in zip(messages, predictions):
    print(f"Tin nhắn: \"{message}\" -> Ý định: [{intent}]")
```

**Kết quả thực thi mẫu:**
```text
Tin nhắn: "I want my money back, please." -> Ý định: [refund]
Tin nhắn: "Please cancel my order." -> Ý định: [cancel_order]
Tin nhắn: "Can you tell me where my package is?" -> Ý định: [track_order]
Tin nhắn: "My payment was declined." -> Ý định: [payment_problem]
```

> **Lưu ý:** Kết quả dự đoán phụ thuộc vào tập dữ liệu huấn luyện và phiên bản môi trường thực thi. Đối với các tin nhắn có câu từ khác biệt hoặc ngữ cảnh phức tạp, mô hình baseline có thể đưa ra kết quả chưa chính xác.
