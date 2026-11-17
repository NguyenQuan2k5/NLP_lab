# NLP Lab 05 — Sequence Models: RNN, LSTM and GRU

Dự án bài tập lớn **LAB 05 — Sequence Models: RNN, LSTM and GRU** thuộc học phần Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP), Trường Đại học Khoa học Tự nhiên - ĐHQGHN.

## 📁 Cấu trúc Thư mục

Toàn bộ tài liệu gốc và sản phẩm bàn giao (deliverables) được tổ chức chuẩn hóa theo yêu cầu:

```text
lab5/
├── BT/
│   └── BT.pdf                         # Bản scan bài tập tính toán viết tay (tạo từ 3 ảnh)
├── docs/
│   └── W5.pdf                         # Tài liệu đề bài và hướng dẫn thực hành LAB 05
└── lab05/                             # Thư mục sản phẩm bàn giao chính thức (Section 29)
    ├── README.md                      # Hướng dẫn chi tiết, cài đặt và tóm tắt kết quả
    ├── calculations.md                # Lời giải chi tiết các bài tập giải tích và lý thuyết
    ├── prediction.md                  # 5 giả thuyết trước thực nghiệm và mã suy luận mẫu
    ├── rnn_cell.py                    # Cài đặt SimpleRNNCell và kiểm chứng với PyTorch
    ├── sequence_classification.ipynb  # Notebook thực nghiệm hoàn chỉnh (đã chạy sẵn outputs)
    ├── results.csv                    # Bảng số liệu định lượng chi tiết tất cả mô hình
    ├── learning_curves.png            # Đồ thị Train Loss vs Val Loss qua các Epochs
    ├── confusion_matrix.png           # Ma trận nhầm lẫn của các mô hình tuần tự
    ├── error_analysis.md              # Phân tích định tính chuyên sâu các trường hợp lỗi
    └── reflection.md                  # Báo cáo đánh giá, đối chiếu, Learning Check & AI Statement
```

Chi tiết hướng dẫn cài đặt và nội dung từng thành phần vui lòng xem tại [`lab05/README.md`](lab05/README.md).
