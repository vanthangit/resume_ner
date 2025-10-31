import json
import spacy
from spacy.training import Example
from spacy.util import minibatch
from tqdm import tqdm
import random
import os

print("="*70)
print("RESUME NER TRAINING")
print("="*70)

# ========== STEP 1: LOAD DATA ==========
print("\n[1/5] Loading training data...")

try:
    with open("data/train_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"[SUCCESS] Loaded {len(train_data)} training examples")
except FileNotFoundError:
    print("[ERROR] data/train_data.json not found")
    exit(1)

# ========== STEP 2: LOAD OR CREATE MODEL ==========
print("\n[2/5] Loading/Creating model...")

# Try to load pre-trained model, fallback to blank
model_options = [
    ("xx_core_web_sm", "Pre-trained multilingual model"),
    ("xx_sent_ud_sm", "Sentence segmentation model"),
]

nlp = None
for model_name, description in model_options:
    try:
        nlp = spacy.load(model_name)
        print(f"[SUCCESS] Loaded {description}: {model_name}")
        break
    except OSError:
        print(f"[INFO] {model_name} not found (optional)")

if nlp is None:
    print("[INFO] Using blank multilingual model (no download needed)")
    nlp = spacy.blank("xx")
    print("[SUCCESS] Created blank multilingual model")

# ========== STEP 3: SETUP NER PIPE ==========
print("\n[3/5] Setting up NER pipe...")

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
    print("[SUCCESS] Added NER pipe")
else:
    ner = nlp.get_pipe("ner")
    print("[INFO] NER pipe already exists")

# Get labels from training data
labels = set()
for _, ann in train_data:
    for start, end, label in ann["entities"]:
        labels.add(label)

# Add labels to model
for label in labels:
    ner.add_label(label)

print(f"[INFO] Labels: {sorted(labels)}")
print(f"[INFO] Total training examples: {len(train_data)}")

# ========== STEP 4: PREPARE TRAINING DATA ==========
print("\n[4/5] Preparing training data...")

# Validate and prepare examples
examples = []
invalid_count = 0

for text, ann in train_data:
    try:
        # Skip empty text
        if not text or len(text.strip()) < 3:
            invalid_count += 1
            continue
        
        # Skip if no entities
        if not ann.get("entities"):
            continue
        
        # Validate entity positions
        valid_entities = []
        for start, end, label in ann["entities"]:
            # Check if positions are valid
            if 0 <= start < end <= len(text):
                # Check if label is registered
                if label in labels:
                    valid_entities.append((start, end, label))
                else:
                    invalid_count += 1
            else:
                invalid_count += 1
        
        # Only add if has valid entities
        if valid_entities:
            ann_dict = {"entities": valid_entities}
            try:
                example = Example.from_dict(nlp.make_doc(text), ann_dict)
                examples.append(example)
            except Exception as e:
                invalid_count += 1
    
    except Exception as e:
        invalid_count += 1

print(f"[INFO] Valid examples: {len(examples)}")
if invalid_count > 0:
    print(f"[WARNING] Skipped {invalid_count} invalid examples")

if len(examples) < 10:
    print("[ERROR] Not enough valid training examples")
    exit(1)

# ========== STEP 5: TRAINING ==========
print("\n[5/5] Training model...")

# Disable other pipes to speed up training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    # Initialize optimizer
    optimizer = nlp.begin_training()
    
    # Training configuration
    n_epochs = 50
    batch_size = 8
    dropout = 0.5
    
    print(f"\n[CONFIG]")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dropout: {dropout}")
    print(f"  Examples: {len(examples)}\n")
    
    losses_history = []
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(n_epochs):
        random.shuffle(examples)
        losses = {}
        
        # Training progress bar
        pbar = tqdm(
            minibatch(examples, size=batch_size),
            total=len(examples) // batch_size,
            desc=f"Epoch {epoch+1:2d}/{n_epochs}",
            leave=False
        )
        
        for batch in pbar:
            nlp.update(
                batch,
                sgd=optimizer,
                losses=losses,
                drop=dropout
            )
            
            loss_val = losses.get('ner', 0)
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        loss_val = losses.get('ner', 0)
        losses_history.append(loss_val)
        
        # Track best model
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch + 1
        
        # Print epoch summary
        avg_loss = sum(losses_history[-5:]) / len(losses_history[-5:]) if len(losses_history) >= 5 else loss_val
        
        trend = "↓" if (len(losses_history) > 1 and loss_val < losses_history[-2]) else "↑"
        
        print(f"Epoch {epoch+1:2d}/{n_epochs} | Loss: {loss_val:.6f} | Avg-5: {avg_loss:.6f} {trend}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = f"data/models/ner_resume_epoch{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            nlp.to_disk(checkpoint_dir)
            print(f"  [CHECKPOINT] Saved at: {checkpoint_dir}")

# ========== STEP 6: SAVE MODEL ==========
print("\n[SAVING] Saving final model...")

output_dir = "data/models/ner_resume"
os.makedirs(output_dir, exist_ok=True)
nlp.to_disk(output_dir)
print(f"[SUCCESS] Model saved to: {output_dir}")

# ========== STEP 7: TRAINING SUMMARY ==========
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

print(f"\nLoss Statistics:")
print(f"  Initial loss: {losses_history[0]:.6f}")
print(f"  Final loss:   {losses_history[-1]:.6f}")
print(f"  Best loss:    {best_loss:.6f} (Epoch {best_epoch})")
print(f"  Reduction:    {((losses_history[0] - losses_history[-1]) / losses_history[0] * 100):.1f}%")

print(f"\nModel Info:")
print(f"  Labels: {sorted(labels)}")
print(f"  Training examples: {len(examples)}")
print(f"  Epochs: {n_epochs}")
print(f"  Final model path: {output_dir}")

# ========== STEP 8: TESTING ==========
print("\n" + "="*70)
print("MODEL TESTING")
print("="*70)

# Reload model fresh
nlp = spacy.load(output_dir)

test_texts = [
    "Email: hoangthang@gmail.com | Tên: Nguyễn Văn Thắng",
    "Công ty Apple Inc, địa chỉ: 123 Đường Lê Lợi, TP.HCM",
    "Người liên hệ: Trần Thị Mai, số điện thoại: 0909123456",
    "Contact: John Smith, Email: john@example.com",
    "Working at Google in Mountain View, California",
    "Phone: +1-555-0123, Address: 456 Main Street"
]

total_entities = 0
for test_text in test_texts:
    doc = nlp(test_text)
    print(f"\n[TEST] {test_text}")
    
    if doc.ents:
        for ent in doc.ents:
            print(f"       [{ent.label_:8s}] {ent.text}")
            total_entities += 1
    else:
        print("       [NO ENTITIES FOUND]")

print(f"\n[INFO] Total entities detected in test: {total_entities}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nNext step: python predict.py data/samples/resume.pdf\n")