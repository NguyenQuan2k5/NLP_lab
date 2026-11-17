# Ứng dụng — Lựa chọn C: Phân loại ý định (Intent Classification)

## Mục tiêu
Xây dựng một ứng dụng NLP nhỏ để phân loại các tin nhắn hỗ trợ khách hàng thành 4 ý định (intents):

- `refund`: Hoàn tiền / trả hàng hoàn tiền
- `cancel_order`: Hủy đơn hàng
- `track_order`: Tra cứu / theo dõi trạng thái đơn hàng
- `payment_problem`: Vấn đề về thanh toán

Quy trình xử lý cơ bản (Baseline pipeline):

`Tin nhắn khách hàng → TF-IDF → Hồi quy Logistic (Logistic Regression) → Dự đoán ý định`

Dự án bao gồm: tập dữ liệu mẫu, notebook huấn luyện/đánh giá, script huấn luyện và giao diện demo trực quan bằng Streamlit.

## Lưu ý quan trọng về dữ liệu
File `data/intents.csv` là **tập dữ liệu khởi đầu tổng hợp có quy mô nhỏ (80 mẫu)**, được tạo phục vụ mục đích minh họa quy trình từ đầu đến cuối (end-to-end), không phải là tập dữ liệu benchmark chuẩn thu thập từ môi trường thực tế. Kết quả trên tập dữ liệu này mang tính chất thử nghiệm quy trình. Để triển khai thực tế, cần thu thập thêm nhiều dữ liệu thực tế hoặc sử dụng các bộ dữ liệu công khai lớn hơn.

## Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10+.

```bash
pip install -r requirements.txt
```

## Huấn luyện và Đánh giá
Chạy script huấn luyện:
```bash
python train.py
```
Lệnh này sẽ tự động sinh ra các file:
- `results.csv`: Bảng tổng hợp các chỉ số đánh giá (Accuracy, Precision, Recall, F1 macro, v.v.).
- `confusion_matrix.png`: Biểu đồ ma trận nhầm lẫn (Confusion Matrix).
- `error_analysis.md`: Báo cáo phân tích chi tiết các mẫu dự đoán sai trên tập test.
- `intent_model.joblib`: Pipeline mô hình đã huấn luyện hoàn chỉnh.

Quá trình đánh giá sử dụng kỹ thuật chia tập train/test phân tầng (stratified split, tỉ lệ 75/25) để bảo toàn tỷ lệ giữa các lớp nhãn.

## Chạy ứng dụng Demo
```bash
streamlit run app.py
```
Nhập tin nhắn của khách hàng vào ô văn bản và nhấn **Dự đoán ý định (Predict intent)**. Ứng dụng sẽ hiển thị ý định được dự đoán cùng biểu đồ xác suất chi tiết của từng lớp.

## Chạy trên Google Colab
1. Tải lên và giải nén dự án trên môi trường Colab.
2. Cài đặt các thư viện cần thiết: `!pip install -r requirements.txt`.
3. Huấn luyện mô hình: `!python train.py`.
4. Để chạy giao diện Streamlit trên Colab, cần sử dụng công cụ tunnel (như `localtunnel` hoặc `ngrok`). Chạy cục bộ bằng lệnh `streamlit run app.py` thường là cách thuận tiện nhất.

## Định nghĩa các nhãn (Labels)
- `refund`: Khách hàng yêu cầu hoàn lại tiền hoặc lấy lại tiền thanh toán.
- `cancel_order`: Khách hàng muốn hủy đơn đặt hàng.
- `track_order`: Khách hàng hỏi về tiến độ giao hàng hoặc tình trạng đơn hàng.
- `payment_problem`: Thanh toán bị từ chối, thất bại, trừ tiền 2 lần hoặc sai số tiền.

## Lưu ý & Hạn chế
- Các câu ví dụ là dữ liệu mẫu nhân tạo nhằm trình diễn luồng hoạt động của bài toán.
- Bộ phân loại TF-IDF + Logistic Regression là mô hình đường cơ sở (baseline), chưa thể nắm bắt hết các ngữ cảnh phức tạp hoặc cách diễn đạt biến thể tinh vi.
- Bản demo chỉ phục vụ phân loại văn bản, không liên kết thực hiện hành động thật đối với hệ thống ngân hàng hoặc quản lý đơn hàng.
