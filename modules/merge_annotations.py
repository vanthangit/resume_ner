import json
import os

input_dir = "data/annotations"
output_file = "data/train_data.json"

train_data = []

# Lặp qua tất cả file .json trong thư mục
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Lọc các annotation hợp lệ (bỏ null)
        for ann in data.get("annotations", []):
            if ann:
                text, labels = ann
                train_data.append((text, labels))

# Xuất ra file train_data.json
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

print(f"Đã gộp {len(train_data)} mẫu có nhãn từ {len(os.listdir(input_dir))} file JSON!")
print(f"File đầu ra: {output_file}")
