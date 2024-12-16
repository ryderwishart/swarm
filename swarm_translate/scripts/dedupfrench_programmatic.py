import json
from pathlib import Path
import glob
import re
from difflib import SequenceMatcher

def normalize_text(text: str) -> str:
    """Normalize text by removing extra spaces, standardizing quotes, and articles."""
    text = re.sub(r'\s+', ' ', text)
    # Standardize quotes
    text = text.replace('« ', '«').replace(' »', '»')
    text = text.replace('" ', '"').replace(' "', '"')
    # Normalize articles and common variations
    text = re.sub(r'\b(la|le|les|un|une|des)\b', '', text)
    
    return text.strip()

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, handling French punctuation."""
    # First normalize the text
    text = normalize_text(text)
    
    # Split on sentence endings but keep the delimiter
    sentences = re.split(r'([.!?] )', text)
    
    # Recombine sentences with their delimiters and handle the last one
    result = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i+1].strip() in ['. ', '! ', '? ']:
            result.append(sentences[i] + sentences[i+1].strip())
            i += 2
        else:
            if sentences[i].strip():
                result.append(sentences[i].strip())
            i += 1
    
    return result

def similarity_ratio(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def clean_translations(file_path: Path):
    """Remove duplicated content from verses that are suspiciously long."""
    
    # Read all lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    cleaned_lines = []
    window_size = 6  # Increased window size
    
    for i in range(0, len(lines)):
        current_line = lines[i].copy()
        
        # Check if translation is more than 1.5x longer than original
        if (len(current_line["translation"]) > 1.5 * len(current_line["original"]) 
            and i >= 1):  # Only need 1 previous verse
            # Get all sentences from previous verses
            prev_sentences = []
            for j in range(max(0, i - window_size), i):
                prev_sentences.extend(split_into_sentences(lines[j]["translation"]))
            
            # Split current translation into sentences
            current_sentences = split_into_sentences(current_line["translation"])
            
            # Keep only unique sentences
            unique_sentences = []
            for current_sent in current_sentences:
                is_duplicate = False
                for prev_sent in prev_sentences:
                    if similarity_ratio(current_sent, prev_sent) > 0.7:  # Lowered threshold
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_sentences.append(current_sent)
            
            # Rejoin unique sentences with proper spacing
            current_line["translation"] = " ".join(unique_sentences)
        
        cleaned_lines.append(current_line)
    
    # Write cleaned lines back to the same file
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    translations_dir = Path("scenarios/translations")
    
    # Find all eng-fra files (excluding progress files)
    eng_fra_files = [
        Path(f) for f in glob.glob(str(translations_dir / "eng-fra_bible_*.jsonl"))
        if not f.endswith("_progress.json")
    ]
    
    for file_path in eng_fra_files:
        clean_translations(file_path)
        print(f"Cleaned translations in {file_path}")