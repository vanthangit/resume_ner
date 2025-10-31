from docling.document_converter import DocumentConverter
from pathlib import Path
import os

# Thư mục đầu vào & đầu ra
input_dir = Path("data/raw/resumes")
output_dir = Path("data/text")
output_dir.mkdir(parents=True, exist_ok=True)

# Khởi tạo converter 1 lần
converter = DocumentConverter()

# Duyệt qua tất cả file PDF
for pdf_file in input_dir.glob("*.pdf"):
    try:
        # Chuyển đổi PDF → Document
        result = converter.convert(pdf_file)
        doc = result.document

        # Lấy toàn bộ nội dung dạng text
        text = doc.export_to_markdown()

        # Ghi ra file .txt
        output_path = output_dir / f"{pdf_file.stem}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Extracted: {pdf_file.name} → {output_path.name}")

    except Exception as e:
        print(f"Lỗi khi xử lý {pdf_file.name}: {e}")
