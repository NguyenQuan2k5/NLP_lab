"""
LAB 05 — Sequence Models: RNN, LSTM and GRU
Module: Simple RNN Cell Implementation from scratch
Đại học Khoa học Tự nhiên - ĐHQGHN
Học kỳ I - 2026

File: rnn_cell.py
Mục đích:
- Tự cài đặt một RNN Cell theo đúng công thức:
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
- Kiểm tra tính đúng đắn về mặt giải tích và so khớp với torch.nn.RNNCell của PyTorch.
- Giải bài toán tính toán Bài 1 trong Section 6 của tài liệu hướng dẫn.
"""

import torch
import torch.nn as nn
import numpy as np


class SimpleRNNCell(nn.Module):
    """
    Tự cài đặt một Recurrent Cell đơn giản (Simple RNN Cell / Elman RNN Cell)
    
    Công thức toán học:
        a_t = x_t @ W_xh.T + h_{t-1} @ W_hh.T + b_h
        h_t = tanh(a_t)
        
    Kích thước các tensor:
        - x_t: (batch_size, input_size) hoặc (input_size,)
        - h_{t-1}: (batch_size, hidden_size) hoặc (hidden_size,)
        - W_xh: (hidden_size, input_size)
        - W_hh: (hidden_size, hidden_size)
        - b_h: (hidden_size,)
        - h_t: (batch_size, hidden_size)
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        # Khởi tạo trọng số tuân theo chuẩn phân phối đều Xavier/Uniform của PyTorch
        k = (1.0 / hidden_size) ** 0.5
        self.W_xh = nn.Parameter(torch.empty(hidden_size, input_size).uniform_(-k, k))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(-k, k))
        
        if bias:
            self.b_h = nn.Parameter(torch.empty(hidden_size).uniform_(-k, k))
        else:
            self.register_parameter('b_h', None)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor = None) -> torch.Tensor:
        """
        Thực hiện một bước forward qua RNN Cell tại timestep t.
        
        Args:
            x_t: Tensor đầu vào tại thời điểm t, kích thước (batch_size, input_size) hoặc (input_size,)
            h_prev: Hidden state từ bước trước (t-1), kích thước tương ứng. Nếu None, khởi tạo tensor 0.
            
        Returns:
            h_t: Hidden state mới tại thời điểm t.
        """
        is_1d = False
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            is_1d = True

        batch_size = x_t.size(0)

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x_t.device, dtype=x_t.dtype)
        elif h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        # Tính affine combination: a_t = x_t @ W_xh^T + h_prev @ W_hh^T + b_h
        affine = torch.matmul(x_t, self.W_xh.t()) + torch.matmul(h_prev, self.W_hh.t())
        if self.b_h is not None:
            affine = affine + self.b_h

        # Phi tuyến hóa qua hàm tanh
        h_t = torch.tanh(affine)

        if is_1d:
            return h_t.squeeze(0)
        return h_t


def verify_with_pytorch():
    """
    Kiểm tra độ chính xác của SimpleRNNCell bằng cách so sánh trực tiếp
    với nn.RNNCell chuẩn của PyTorch khi sao chép cùng trọng số.
    """
    print("=" * 60)
    print("1. Kiểm chứng SimpleRNNCell so với torch.nn.RNNCell")
    print("=" * 60)
    
    torch.manual_seed(42)
    input_size = 8
    hidden_size = 4
    batch_size = 3

    # Khởi tạo cell tự xây dựng và cell chuẩn của PyTorch
    custom_cell = SimpleRNNCell(input_size, hidden_size)
    torch_cell = nn.RNNCell(input_size, hidden_size, nonlinearity='tanh', bias=True)

    # Sao chép chuẩn xác toàn bộ tham số
    with torch.no_grad():
        torch_cell.weight_ih.copy_(custom_cell.W_xh)
        torch_cell.weight_hh.copy_(custom_cell.W_hh)
        torch_cell.bias_ih.copy_(custom_cell.b_h)
        torch_cell.bias_hh.zero_()  # Đặt bias_hh bằng 0 vì custom_cell dùng 1 vector bias gộp

    # Dữ liệu thử nghiệm
    x_test = torch.randn(batch_size, input_size)
    h_test = torch.randn(batch_size, hidden_size)

    # Tính toán đầu ra
    out_custom = custom_cell(x_test, h_test)
    out_torch = torch_cell(x_test, h_test)

    # Đo sai số tuyệt đối lớn nhất
    diff = torch.max(torch.abs(out_custom - out_torch)).item()
    print(f"Kích thước x_t: {list(x_test.shape)}")
    print(f"Kích thước h_prev: {list(h_test.shape)}")
    print(f"Đầu ra custom cell:\n{out_custom.detach().numpy()}")
    print(f"Đầu ra PyTorch cell:\n{out_torch.detach().numpy()}")
    print(f"Sai số tuyệt đối cực đại (Max absolute diff): {diff:.2e}")

    assert diff < 1e-6, "Thử nghiệm thất bại: Sai số quá lớn!"
    print("-> KẾT QUẢ: KHỚP 100% VỚI PYTORCH FRAMEWORK (Test Passed!)\n")


def solve_exercise_1():
    """
    Giải bài tập tính toán số 1 (Section 6 trong tài liệu LAB 05):
    Cho:
        x_t = [1, 0]^T
        h_{t-1} = [0.5, 0.2]^T
        W_xh = [[0.5, 0.2],
                [0.1, 0.4]]
        W_hh = [[0.3, 0.1],
                [0.2, 0.5]]
        b_h = [0, 0]^T
    Yêu cầu:
        Tính a_t = W_xh @ x_t + W_hh @ h_{t-1} + b_h
        Sau đó tính h_t = tanh(a_t)
    """
    print("=" * 60)
    print("2. Giải Bài tập tính toán số 1 (Section 6 — W5.pdf)")
    print("=" * 60)

    x_t = np.array([1.0, 0.0])
    h_prev = np.array([0.5, 0.2])
    W_xh = np.array([[0.5, 0.2],
                     [0.1, 0.4]])
    W_hh = np.array([[0.3, 0.1],
                     [0.2, 0.5]])
    b_h = np.array([0.0, 0.0])

    # Bước 1: W_xh @ x_t
    W_xh_x = W_xh @ x_t
    print("Bước 1: Tính W_xh @ x_t:")
    print(f"  [[0.5, 0.2], [0.1, 0.4]] @ [1, 0]^T = [{W_xh_x[0]:.2f}, {W_xh_x[1]:.2f}]^T")

    # Bước 2: W_hh @ h_{t-1}
    W_hh_h = W_hh @ h_prev
    print("\nBước 2: Tính W_hh @ h_{t-1}:")
    print(f"  [[0.3, 0.1], [0.2, 0.5]] @ [0.5, 0.2]^T = [{W_hh_h[0]:.2f}, {W_hh_h[1]:.2f}]^T")

    # Bước 3: a_t
    a_t = W_xh_x + W_hh_h + b_h
    print("\nBước 3: Tính a_t = W_xh @ x_t + W_hh @ h_{t-1} + b_h:")
    print(f"  a_t = [{a_t[0]:.4f}, {a_t[1]:.4f}]^T")

    # Bước 4: h_t = tanh(a_t)
    h_t = np.tanh(a_t)
    print("\nBước 4: Tính h_t = tanh(a_t):")
    print(f"  tanh({a_t[0]:.4f}) = {h_t[0]:.6f}")
    print(f"  tanh({a_t[1]:.4f}) = {h_t[1]:.6f}")
    print(f"  h_t = [{h_t[0]:.6f}, {h_t[1]:.6f}]^T")

    # Kiểm tra lại bằng class SimpleRNNCell
    cell = SimpleRNNCell(input_size=2, hidden_size=2, bias=True)
    with torch.no_grad():
        cell.W_xh.copy_(torch.tensor(W_xh, dtype=torch.float32))
        cell.W_hh.copy_(torch.tensor(W_hh, dtype=torch.float32))
        cell.b_h.copy_(torch.tensor(b_h, dtype=torch.float32))
        x_torch = torch.tensor(x_t, dtype=torch.float32)
        h_prev_torch = torch.tensor(h_prev, dtype=torch.float32)
        h_t_torch = cell(x_torch, h_prev_torch)

    print("\nKiểm tra lại bằng class SimpleRNNCell:")
    print(f"  h_t (PyTorch Tensor): {h_t_torch.numpy()}")
    np.testing.assert_allclose(h_t, h_t_torch.numpy(), atol=1e-6)
    print("-> KẾT QUẢ TÍNH TAY VÀ CELL CODE TRÙNG KHỚP TUYỆT ĐỐI!\n")


if __name__ == "__main__":
    verify_with_pytorch()
    solve_exercise_1()
