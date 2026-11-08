# Handwritten Text Recognition

Dự án nhận dạng **từng chữ số viết tay từ 0 đến 9** trên MNIST. Notebook chính trình bày quá trình huấn luyện, khảo sát siêu tham số, đánh giá và xuất mô hình. Repo cũng có thí nghiệm phân loại Fashion-MNIST và mã tiền xử lý EMNIST.

Hướng dẫn dưới đây được đối chiếu với [commit `5da21f0`](https://github.com/ddkhoa83/handwritten_text_recognition/tree/5da21f0ac26c2e00e03d57984945c6e56979c50f).

## Cấu trúc dự án

| Đường dẫn | Nội dung |
|---|---|
| [`notebooks/main.ipynb`](notebooks/main.ipynb) | Thí nghiệm MNIST và Fashion-MNIST, biểu đồ, báo cáo phân loại, ma trận nhầm lẫn và cell lưu mô hình. |
| [`data/MNIST/raw/`](data/MNIST/raw/) | Ảnh và nhãn MNIST gốc ở định dạng IDX. |
| [`data/preprocessing.py`](data/preprocessing.py) | Đọc MNIST gốc, trải phẳng ảnh thành 784 đặc trưng, chuẩn hóa pixel về `[0, 1]` và tách validation. |
| [`data/preprocessing_mnist_400k.py`](data/preprocessing_mnist_400k.py) | Đọc MNIST mở rộng và dùng metadata để giữ ảnh gốc cùng các bản xoay trong một tập. |
| [`data/FashionMNIST/`](data/FashionMNIST/) và [`data/EMNIST/`](data/EMNIST/) | Dữ liệu Fashion-MNIST, mã tiền xử lý Fashion-MNIST và EMNIST. Dữ liệu thô EMNIST cần chuẩn bị riêng nếu dùng. |
| [`src/algorithm/`](src/algorithm/) | Naive Bayes, Logistic Regression, SVM tuyến tính, MLP, CNN, stacking và phép tăng cường dữ liệu. Notebook chính dùng `MLP.py` với PyTorch. |
| [`saved_models/`](saved_models/) | Tám mô hình đã lưu bằng Joblib, có thể tải để dự đoán trực tiếp. |
| [`figures/`](figures/) | Thư mục dành cho hình xuất từ thí nghiệm. Nhiều hình đã lưu trực tiếp trong notebook. |
| [`app_buil/`](app_buil/) và `app_buil.zip` | Bản sao notebook, mã mô hình và checkpoint dùng cho khâu đóng gói. Quy trình tái lập dưới đây sử dụng notebook ở thư mục gốc. |
| [`requirement.txt`](requirement.txt) | Danh sách phiên bản phụ thuộc của dự án, gồm cả các gói dành riêng cho Windows. |

## 1. Chuẩn bị môi trường

Clone repo và vào thư mục dự án:

```bash
git clone https://github.com/ddkhoa83/handwritten_text_recognition.git
cd handwritten_text_recognition
```

Để dùng đúng mã nguồn của các số liệu tham chiếu trong README, chạy `git checkout --detach 5da21f0` ngay sau khi clone.

Dùng **Python 3.12** theo file `.python-version`. Trên Linux hoặc macOS, dùng `python3.12` nếu lệnh này có trên máy:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Trên Windows CMD:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

Cài nhóm thư viện cần cho notebook. Các phiên bản bên dưới lấy từ `requirement.txt`. Cách cài theo nhóm bỏ qua các phụ thuộc ngoài notebook, ví dụ `pywinpty` trên Linux và macOS.

```bash
python -m pip install --upgrade pip
python -m pip install numpy==2.5.4 scipy==1.18.1 scikit-learn==1.9.1 pandas==3.0.6 matplotlib==3.11.2 joblib==1.6.0 jupyterlab==4.6.4 ipykernel==7.4.0
```

Trên Linux và Windows, cài PyTorch và torchvision từ kho CPU:

```bash
python -m pip install --timeout 120 --retries 5 torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

Trên macOS, dùng kho mặc định:

```bash
python -m pip install torch==2.14.1 torchvision==0.29.1
```

Sau đó kiểm tra môi trường và đăng ký kernel cho notebook:

```bash
python -m pip check
python -c "import numpy, sklearn, torch, torchvision; print(numpy.__version__, sklearn.__version__, torch.__version__)"
python -m ipykernel install --user --name htr-mnist --display-name "Python 3.12 (HTR MNIST)"
```

Nếu pip báo một phiên bản PyTorch không có wheel cho kiến trúc máy bạn, dùng môi trường có cặp phiên bản trên, hoặc chọn cặp torch/torchvision tương thích rồi ghi lại phiên bản thực tế khi báo cáo kết quả. Trên Mac Intel, hãy kiểm tra kỹ khả năng cài đúng cặp đã ghim trước khi chạy cả notebook.

## 2. Chuẩn bị dữ liệu

**MNIST gốc và Fashion-MNIST.** Hai tập IDX đã có trong repo. MNIST gốc được chia thành **50.000 train, 10.000 validation, 10.000 test**. Notebook dùng Fashion-MNIST với `val_size=0.2`, tương ứng **48.000 train, 12.000 validation, 10.000 test**. Mỗi lần chia giữ tỷ lệ lớp và dùng seed 42.

**MNIST mở rộng.** Phần CNN tăng cường mạnh yêu cầu gói `MNIST_expanded_400k_360k_train_40k_test_pm15.zip`, cần lấy riêng từ người duy trì dự án. Giải nén gói sao cho có đúng hai đường dẫn:

- `data/MNIST_expanded_400k_360k_train_40k_test_pm15/MNIST/raw/` chứa bốn tệp IDX ảnh và nhãn train/test đã giải nén.
- `data/MNIST_expanded_400k_360k_train_40k_test_pm15/MNIST/metadata/` chứa `train_rotations.csv` và `t10k_rotations.csv`.

Loader kiểm tra **360.000 ảnh train trước khi chia validation** và **40.000 ảnh test**, sau đó tách thành **299.989 train, 60.011 validation, 40.000 test** với bộ dữ liệu tham chiếu. Ảnh số 6 và 9 được xoay trong khoảng **−15° đến +15°**. Metadata giữ ảnh gốc cùng các bản xoay của nó trong cùng một tập. Kiểm tra đường dẫn nếu gặp `FileNotFoundError` ở mục **Data augement** của notebook.

## 3. Kiểm tra nhanh với mô hình đã lưu

Chạy đoạn Python sau từ **thư mục gốc repo**, trong môi trường ảo vừa tạo:

```python
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score
from data.preprocessing import X_test_scaled, y_test

model = joblib.load(Path("saved_models") / "logistic_regression.joblib")
predictions = model.predict(X_test_scaled)

print(f"Test samples: {len(y_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")
```

Với checkpoint tại commit tham chiếu, bước kiểm tra cho **10.000 mẫu**, **accuracy 0.9241** và **macro F1 0.9229**. Chỉ tải checkpoint Joblib từ nguồn đáng tin cậy.

## 4. Chạy lại thí nghiệm

Từ thư mục gốc, mở JupyterLab với thư mục làm việc là `notebooks/`. Cell đầu dùng `Path.cwd().parent` để tìm mã và thư mục `saved_models/`.

```bash
cd notebooks
python -m jupyterlab main.ipynb
```

1. Chọn kernel **Python 3.12 (HTR MNIST)**. Kiểm tra `Path.cwd()` trỏ đến thư mục `notebooks/`.
2. Khởi động lại kernel và chạy tuần tự các mục **Import** và **Train Model and Classification report**. Logistic Regression và SVM chọn cấu hình theo validation. MLP có baseline, các lượt khảo sát và cấu hình cuối dùng early stopping.
3. Chạy phần **CNN** trên MNIST gốc. Khi có bộ mở rộng, chạy tiếp **Data augement → CNN with data agument → Confusion matrix**. Phần import bộ mở rộng gán lại các biến dữ liệu. Nếu quay lại thí nghiệm MNIST gốc, hãy chạy lại cell import ban đầu.
4. Để xuất checkpoint mới, tại cell ngay trước **Save weights of models**, đặt `model_dir = project_dir / "saved_models_reproduced"` rồi chạy `model_dir.mkdir(parents=True, exist_ok=True)`. Sau đó chạy phần lưu mô hình. Cách này tạo một thư mục kết quả riêng để đối chiếu checkpoint đi kèm.
5. Chạy **Test with FS-MNIST** nếu cần tái lập phần Fashion-MNIST. Nếu chưa có MNIST mở rộng, bạn vẫn có thể chạy phần này sau các cell import, bỏ qua khối CNN tăng cường mạnh và cell xuất toàn bộ checkpoint.

Thời gian huấn luyện phụ thuộc CPU, thư viện đại số tuyến tính và số luồng. CNN mở rộng, stacking và các lượt khảo sát MLP chạy lâu hơn các thử nghiệm đơn lẻ.

## 5. Đối chiếu kết quả

Các số liệu dưới đây được lưu trong output của [notebook tham chiếu](https://github.com/ddkhoa83/handwritten_text_recognition/blob/5da21f0ac26c2e00e03d57984945c6e56979c50f/notebooks/main.ipynb).

| Thí nghiệm | Tập đánh giá | Accuracy |
|---|---|---:|
| Multinomial Naive Bayes | MNIST gốc · test | 83,51% |
| Logistic Regression, learning rate 0,3 và 60 epoch | MNIST gốc · validation | 92,04% |
| Linear SVM, learning rate 0,03 và 60 epoch | MNIST gốc · validation | 92,11% |
| MLP cấu hình cuối với early stopping | MNIST gốc · test | 98,25% |
| KNN | MNIST gốc · test | 97,00% |
| Random Forest | MNIST gốc · test | 96,68% |
| Stacking Ensemble | MNIST gốc · test theo notebook | 97,77% |
| AdaBoost | MNIST gốc · test | 85,82% |
| CNN trên MNIST gốc | MNIST gốc · validation | 99,24% |
| CNN tăng cường mạnh | MNIST mở rộng · validation | 98,8952% |
| CNN tăng cường mạnh | MNIST mở rộng · test | 98,9250% |

Đối chiếu từng hàng trên đúng tập đánh giá. Cell ma trận nhầm lẫn của CNN mở rộng in chữ `Validation accuracy`, nhưng dùng `X_test_scaled_au_cnn` và `y_test_cnn`, nên **98,9250% là accuracy test**. Notebook truyền test MNIST vào `ensemble.fit` để MLP thành phần dùng làm validation. Vì vậy **97,77% là kết quả của quy trình đã ghi**, có khả năng chịu ảnh hưởng từ việc dùng test trong huấn luyện. Khi cần đánh giá độc lập, dùng validation trong `fit` và chỉ đánh giá một lần trên test.

Khi báo cáo kết quả chạy lại, ghi kèm commit, phiên bản thư viện, tập dữ liệu, seed, cấu hình và số lượng mẫu đánh giá. Kết quả có thể dao động giữa các môi trường. Thời gian chạy chỉ nên so sánh trên cùng phần cứng và thiết lập số luồng.

## Giấy phép

Mã nguồn được phân phối theo [MIT License](LICENSE).
