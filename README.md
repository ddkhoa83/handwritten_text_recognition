# Handwritten Text Recognition

## Thiết lập môi trường trên Windows

### 1. Cài Python 3.12

Tải Python 3.12 từ [python.org](https://www.python.org/downloads/windows/). Khi cài đặt, chọn **Add python.exe to PATH** và cài Python Launcher nếu có tùy chọn.

Mở **Command Prompt (CMD)** và kiểm tra:

```bat
py -3.12 --version
```

### 2. Mở thư mục dự án

Tải hoặc clone dự án, sau đó mở CMD tại thư mục chứa file `requirement.txt`:

```bat
cd /d "C:\duong-dan\handwritten_text_recognition"
```

Thay đường dẫn trên bằng vị trí dự án trên máy của bạn.

### 3. Tạo và kích hoạt môi trường ảo

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

Khi kích hoạt thành công, đầu dòng lệnh sẽ có `(.venv)`.

### 4. Cài thư viện

```bat
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
python -m pip check
```

File `requirement.txt` cần sử dụng tên và phiên bản thư viện (ví dụ `numpy==...` với phiên bản thực tế), không chứa đường dẫn cục bộ `file:///...` từ máy khác.

### 5. Sử dụng lại môi trường

Mỗi lần mở CMD mới, vào thư mục dự án và kích hoạt lại:

```bat
.venv\Scripts\activate.bat
```

Thoát môi trường khi dùng xong:

```bat
deactivate
```

> Các lệnh kích hoạt trên dành cho CMD. Nếu dùng PowerShell, chạy `.\.venv\Scripts\Activate.ps1`. Nếu PowerShell chặn script, bạn có thể chuyển sang CMD.

## Lưu ý

- Không cần cài Conda để sử dụng hướng dẫn này.
- Không đưa thư mục `.venv/` lên GitHub; thêm `.venv/` vào `.gitignore`.
- Các thư viện chỉ có trên Conda hoặc phụ thuộc GPU/CUDA có thể cần hướng dẫn cài riêng.
