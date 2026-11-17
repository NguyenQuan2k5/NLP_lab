# NLP Lab 06 — Attention Mechanism: From Recurrent State to Context-Aware Representation

Dự án bài tập lớn **LAB 06 — Attention Mechanism: From Recurrent State to Context-Aware Representation** thuộc học phần Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP), Trường Đại học Khoa học Tự nhiên - ĐHQGHN.

## 📁 Cấu trúc Thư mục

Toàn bộ tài liệu gốc và sản phẩm bàn giao (deliverables) được tổ chức chuẩn hóa theo yêu cầu tại **Section 32 (Deliverables)** trong tài liệu `docs/W6.pdf`:

```text
lab6/
├── BT/
│   └── BT.pdf                         # Bản scan bài tập tính toán viết tay (tạo từ 6 ảnh)
├── docs/
│   └── W6.pdf                         # Tài liệu đề bài và hướng dẫn thực hành LAB 06
└── lab06/                             # Thư mục sản phẩm bàn giao chính thức (Section 32)
    ├── README.md                      # Hướng dẫn chi tiết, cài đặt và tóm tắt kết quả
    ├── calculations.md                # Lời giải chi tiết các bài tập giải tích và lý thuyết
    ├── prediction.md                  # 5 giả thuyết trước thực nghiệm và mã suy luận mẫu
    ├── attention_numpy.py             # Cài đặt Attention thuần NumPy & Stable Softmax
    ├── self_attention.ipynb          # Notebook thực nghiệm hoàn chỉnh (đã chạy sẵn outputs)
    ├── results.csv                    # Bảng số liệu định lượng chi tiết tất cả mô hình
    ├── attention_heatmap.png          # Biểu đồ Heatmap Attention ma trận tương tác từ
    ├── confusion_matrix.png           # Ma trận nhầm lẫn của mô hình Attention vs LSTM
    ├── error_analysis.md              # Phân tích định tính chuyên sâu các trường hợp lỗi
    └── reflection.md                  # Báo cáo đánh giá, đối chiếu, Learning Check & AI Statement
```

Chi tiết hướng dẫn cài đặt và nội dung từng thành phần vui lòng xem trực tiếp tại [`lab06/README.md`](lab06/README.md).
