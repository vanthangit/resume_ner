import os
import re
import json
import pdfplumber
import spacy
from pathlib import Path

class ResumeNERExtractor:
    """Resume NER Extractor using spaCy + Rule-Based approach"""
    
    def __init__(self, model_path="data/models/ner_resume"):
        """Initialize with trained spaCy model"""
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            print("Train model first using: python train.py")
            exit()
        
        self.nlp = spacy.load(model_path)
        print(f"[SUCCESS] Loaded model from: {model_path}")
    
    # ========== RULE-BASED PATTERNS ==========
    
    @staticmethod
    def extract_emails_by_rules(text):
        """Extract emails using regex patterns"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        matches = re.finditer(email_pattern, text)
        
        emails = []
        for match in matches:
            email = match.group(0)
            
            # Filter out false positives
            if any(x in email.lower() for x in ['http://', 'https://', 'github', 'linkedin']):
                continue
            if email.count('@') != 1 or '.git' in email:
                continue
            
            emails.append({
                "text": email,
                "start": match.start(),
                "end": match.end(),
                "source": "rule-based"
            })
        
        # Remove duplicates
        return list({e["text"]: e for e in emails}.values())
    
    @staticmethod
    def extract_names_by_rules(text):
        """Extract names using heuristic patterns"""
        names = []
        
        # Pattern 1: "Name: [NAME]" or "Full name: [NAME]"
        patterns = [
            (r'(?:Full\s+Name|Full\s+name|Name|Tên)\s*:?\s*([A-ZÀ-ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-ỹ][a-zà-ỹ]+)*)', 0.9),
            (r'^##\s+([A-ZÀ-ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-ỹ][a-zà-ỹ]+)*)$', 0.8),
        ]
        
        for pattern, confidence in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                name = match.group(1).strip()
                
                # Validation
                if len(name) < 3 or len(name) > 50:
                    continue
                if '@' in name or 'http' in name.lower():
                    continue
                if any(x in name.lower() for x in ['email', 'phone', 'address']):
                    continue
                
                # Check if already exists
                if not any(n["text"].lower() == name.lower() for n in names):
                    names.append({
                        "text": name,
                        "start": match.start(1),
                        "end": match.end(1),
                        "source": "rule-based",
                        "confidence": confidence
                    })
        
        return names
    
    # ========== SPACY NER ==========
    
    def extract_by_spacy(self, text):
        """Extract entities using trained spaCy model"""
        doc = self.nlp(text)
        entities = {
            "NAME": [],
            "EMAIL": []
        }
        
        for ent in doc.ents:
            label = ent.label_
            entity_text = ent.text
            
            # Filter invalid emails
            if label == "EMAIL":
                if any(x in entity_text.lower() for x in ['http://', 'github', '.git']):
                    continue
                if entity_text.count('@') != 1:
                    continue
            
            entities[label].append({
                "text": entity_text,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy"
            })
        
        return entities
    
    # ========== FUSION STRATEGY ==========
    
    def extract_entities(self, text):
        """
        Combine spaCy + Rule-Based approaches
        Strategy:
        - Use Rule-Based for high-confidence patterns
        - Use spaCy for remaining detections
        - Merge results avoiding duplicates
        """
        
        # Step 1: Rule-based extraction
        rule_emails = self.extract_emails_by_rules(text)
        rule_names = self.extract_names_by_rules(text)
        
        # Step 2: spaCy extraction
        spacy_entities = self.extract_by_spacy(text)
        
        # Step 3: Merge & deduplicate
        final_entities = {
            "NAME": self._merge_entities(rule_names, spacy_entities["NAME"]),
            "EMAIL": self._merge_entities(rule_emails, spacy_entities["EMAIL"])
        }
        
        return final_entities
    
    @staticmethod
    def _merge_entities(rule_ents, spacy_ents):
        """Merge rule-based and spaCy entities, removing duplicates"""
        merged = {}
        
        # Add rule-based (higher priority)
        for ent in rule_ents:
            key = ent["text"].lower()
            merged[key] = ent
        
        # Add spaCy (if not already in merged)
        for ent in spacy_ents:
            key = ent["text"].lower()
            if key not in merged:
                merged[key] = ent
        
        return list(merged.values())
    
    # ========== PDF PROCESSING ==========
    
    @staticmethod
    def pdf_to_text(pdf_path):
        """Extract text from PDF"""
        if not os.path.exists(pdf_path):
            return None
        
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"[ERROR] {e}")
            return None
    
    # ========== MAIN PREDICT ==========
    
    def predict(self, pdf_path, output_json=None):
        """
        Predict NAME and EMAIL from resume PDF
        
        Args:
            pdf_path: Path to resume PDF
            output_json: Optional output JSON file
        
        Returns:
            dict with NAME and EMAIL
        """
        print(f"\n[PROCESSING] {pdf_path}")
        
        # Extract text
        text = self.pdf_to_text(pdf_path)
        if not text:
            return None
        
        print(f"[INFO] PDF length: {len(text)} characters")
        
        # Extract entities
        print("[INFO] Extracting entities (spaCy + Rule-Based)...")
        entities = self.extract_entities(text)
        
        # Format result
        result = {
            "file": pdf_path,
            "name": [e["text"] for e in entities["NAME"]],
            "email": [e["text"] for e in entities["EMAIL"]],
            "details": {
                "names": entities["NAME"],
                "emails": entities["EMAIL"]
            }
        }
        
        # Print results
        print("\n[RESULT]")
        print(f"Names found ({len(result['name'])}):")
        for name in result['name']:
            print(f"  • {name}")
        
        print(f"\nEmails found ({len(result['email'])}):")
        for email in result['email']:
            print(f"  • {email}")
        
        # Save JSON
        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n[SAVED] {output_json}")
        
        return result

# ========== MAIN ==========

if __name__ == "__main__":
    import sys
    
    # Initialize extractor
    extractor = ResumeNERExtractor()
    
    # Process PDF
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        # Default test
        pdf_file = "data/samples/NguyenVanThang_AI_Engineer.pdf"
    
    # Predict
    output_file = f"data/predictions/{Path(pdf_file).stem}_result.json"
    result = extractor.predict(pdf_file, output_file)
    
    if not result:
        print("[ERROR] Failed to process PDF")
        exit(1)