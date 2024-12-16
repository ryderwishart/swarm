import json
from pathlib import Path
import glob
import re
from difflib import SequenceMatcher
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / '.env')

def normalize_text(text: str) -> str:
    """Normalize text by removing extra spaces, standardizing quotes, and articles."""
    text = re.sub(r'\s+', ' ', text)
    # Standardize quotes
    text = text.replace('« ', '«').replace(' »', '»')
    text = text.replace('" ', '"').replace(' "', '"')
    # Normalize articles and common variations
    text = re.sub(r'\b(la|le|les|un|une|des)\b', '', text)
    return text.strip()

def get_llm_deduped_translation(client: OpenAI, original: str, translation: str, prev_translations: list[str]) -> str:
    """Use LLM to fix a translation that's too long."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "You are a French translation expert. Your task is to identify and remove any duplicated content from translations while preserving the actual content for the current verse."
        }, {
            "role": "user",
            "content": f"""Please fix this French translation that seems to contain duplicated content from previous verses.

Original English text:
{original}

Current French translation (may contain duplicates):
{translation}

Previous verses' translations (for reference):
{' '.join(prev_translations)}

Return ONLY the corrected French translation with duplicates removed. Do not include any explanations."""
        }],
        temperature=0.3,
    )
    
    return response.choices[0].message.content.strip()

def clean_translations(file_path: Path):
    """Remove duplicated content from verses that are suspiciously long."""
    
    # Set up OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=api_key)
    
    # Read all lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    cleaned_lines = []
    window_size = 6
    number_of_llm_calls = 0
    
    for i in range(0, len(lines)):
        current_line = lines[i].copy()
        
        # Check if translation is more than 2x longer than original
        if (len(current_line["translation"]) > 2 * len(current_line["original"]) 
            and i >= 1):
            # Get previous translations for context
            prev_translations = [
                lines[j]["translation"] 
                for j in range(max(0, i - window_size), i)
            ]
            
            # Use LLM to fix the translation
            try:
                fixed_translation = get_llm_deduped_translation(
                    client,
                    current_line["original"],
                    current_line["translation"],
                    prev_translations
                )
                current_line["translation"] = fixed_translation
                print(f"Fixed translation for verse {current_line.get('id', i)}")
                number_of_llm_calls += 1
            except Exception as e:
                print(f"Error fixing translation: {e}")
        
        cleaned_lines.append(current_line)
    
    # Write cleaned lines back to the same file
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    print(f"Total LLM calls: {number_of_llm_calls}")

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