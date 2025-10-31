# Trích Xuất Thông Tin Hóa Đơn từ PDF sử dụng NER

Trích xuất **Tên** và **Email** từ file PDF hóa đơn bằng spaCy + Rule-Based.

---

## Bắt Đầu Nhanh

### 1. Cài Đặt

```bash
conda create -n ner_resume python=3.10
conda activate ner_resume
pip install -r requirements.txt
```

### 2. Chuẩn Bị Dữ Liệu

**Bước 1:** Trích xuất text từ PDF

```bash
python extract_with_docling.py
```

- Input: `data/raw/resumes/*.pdf`
- Output: `data/text/*.txt`

**Bước 2:** Gán nhãn với [NER Annotator](https://arunmozhi.in/ner-annotator/)

- Export JSON vào: `data/annotations/`

**Bước 3:** Gộp dữ liệu gán nhãn

```bash
python merge_annotations.py
```

- Output: `data/train_data.json`

### 3. Huấn Luyện Model

```bash
python ner_trainer.py
```

- Model được lưu: `data/models/ner_resume/`

### 4. Dự Đoán

**Một file:**

```bash
python ner_predictor.py
```

---

## Cấu Trúc Thư Mục

```
resume_ner/
├── ner_trainer.py                    # Script huấn luyện
├── ner_predictor.py                  # Script dự đoán
├── extract_with_docling.py            # Chuyển đổi PDF → Text
├── merge_annotations.py        # Gộp file gán nhãn
│
├── data/
│   ├── raw/resumes/           # PDF gốc
│   ├── text/                  # File text đã trích xuất
│   ├── annotations/           # File JSON từ NER Annotator
│   ├── train_data.json        # Dữ liệu huấn luyện đã gộp
│   ├── samples/               # Mẫu để kiểm thử
│   ├── models/ner_resume/     # Model đã huấn luyện
│   └── predictions/           # Kết quả dự đoán
├── requirements.txt            # Các thư viện cần dùng
└── README.md                   # File này
```

---

## Các Script

### `extract_with_docling.py`

Chuyển PDF thành text bằng Docling

```python
# Input: data/raw/resumes/*.pdf
# Output: data/text/*.txt
python extract_text.py
```

### `merge_annotations.py`

Gộp các file JSON gán nhãn thành dữ liệu huấn luyện

```python
# Input: data/annotations/*.json (từ NER Annotator)
# Output: data/train_data.json
python merge_annotations.py
```

### `ner_trainer.py`

Huấn luyện model spaCy NER

```python
# Input: data/train_data.json
# Output: data/models/ner_resume/
python train.py
```

### `ner_predictor.py`

Dự đoán TÊN và EMAIL từ PDF

```python
# Một file
python predict.py
```

---

## Quy Trình Xử Lý

```
PDF gốc
  ↓ [extract_with_docling.py]
File text
  ↓ [Gán nhãn với NER Annotator]
File JSON gán nhãn
  ↓ [merge_annotations.py]
train_data.json
  ↓ [ner_trainer.py]
Model được huấn luyện
  ↓ [ner_predictor.py]
TÊN, EMAIL
```

## Các Thư Viện Cần Dùng

```
spacy==3.7.2
pdfplumber==0.10.3
docling>=1.0.0
numpy==1.24.3
tqdm==4.66.1
```
