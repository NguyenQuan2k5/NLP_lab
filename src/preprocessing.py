"""Tiện ích tải và kiểm tra tính hợp lệ của tập dữ liệu ý định (intents dataset)."""
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "intents.csv"
REQUIRED_COLUMNS = {"text", "intent"}


def load_data(path=DATA_PATH):
    """Tải tập dữ liệu ý định và kiểm tra các trường bắt buộc."""
    df = pd.read_csv(path)
    # Hỗ trợ tự động chuẩn hóa tiêu đề số (0, 1) hoặc xóa khoảng trắng
    if list(df.columns) in [["0", "1"], [0, 1]]:
        df = df.rename(columns={df.columns[0]: "text", df.columns[1]: "intent"})
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Thiếu các cột bắt buộc: {sorted(missing_columns)}")

    df = df[["text", "intent"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["intent"] = df["intent"].fillna("").astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]
    if df.empty:
        raise ValueError("Không còn hàng dữ liệu hợp lệ nào sau khi loại bỏ văn bản/nhãn trống.")
    return df


def dataset_report(df):
    """Trả về thống kê tổng hợp rút gọn của tập dữ liệu."""
    return {
        "samples (số mẫu)": int(len(df)),
        "classes (số lớp)": int(df["intent"].nunique()),
        "class_distribution (phân phối lớp)": df["intent"].value_counts().sort_index().to_dict(),
        "empty_texts (văn bản trống)": int((df["text"].str.strip() == "").sum()),
        "missing_labels (thiếu nhãn)": int((df["intent"].str.strip() == "").sum()),
        "exact_duplicate_texts (văn bản trùng lặp)": int(df["text"].duplicated().sum()),
    }
