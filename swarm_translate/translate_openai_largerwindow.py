from typing import List, Dict, Optional, Tuple
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
import multiprocessing as mp
from functools import partial
from difflib import SequenceMatcher
import csv

# Load environment variables
load_dotenv()

class TranslationScenario:
    def __init__(self, scenario_path: str):
        self.scenario_path = Path(scenario_path)
        with open(scenario_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Set up paths relative to scenario file
        self.base_path = self.scenario_path.parent
        self.input_path = self.base_path / self.config["input"]["file"]
        self.output_dir = self.base_path / self.config["output"]["directory"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Language settings
        self.source_code = self.config["source"]["code"]
        self.source_label = self.config["source"].get("label", self.source_code)
        self.target_code = self.config["target"]["code"]
        self.target_label = self.config["target"].get("label", self.target_code)
        
        # Model settings
        self.routing_model = self.config["models"]["routing"]
        self.translation_model = self.config["models"]["translation"]
        
        # Batch settings
        self.batch_size = self.config["batch_settings"]["batch_size"]
        self.save_frequency = self.config["batch_settings"]["save_frequency"]
        self.resume_from_last = self.config["batch_settings"]["resume_from_last"]
        
        # Load book ID mapping
        self.id_mapping_path = self.base_path / "data" / "id_mapping.json"
        self.book_id_mapping = self._load_book_id_mapping()
    
    def _load_book_id_mapping(self) -> Dict[str, str]:
        """Load the mapping of USFM book codes to full names."""
        try:
            with open(self.id_mapping_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load book ID mapping: {e}")
            return {}
    
    def map_reference_id(self, ref_id: str) -> Optional[str]:
        """Convert USFM reference (e.g., '1CH 10:1') to full name (e.g., '1 Chronicles 10:1')."""
        if not ref_id or not self.book_id_mapping:
            return ref_id
            
        try:
            # Split into book code and verse reference
            parts = ref_id.strip().split(' ', 1)
            if len(parts) != 2:
                return ref_id
                
            book_code, verse_ref = parts
            book_code = book_code.strip()
            verse_ref = verse_ref.strip()
            
            if book_code in self.book_id_mapping:
                return f"{self.book_id_mapping[book_code]} {verse_ref}"
            
            return ref_id
        except Exception as e:
            print(f"Warning: Error mapping reference ID '{ref_id}': {e}")
            return ref_id
    
    def get_style_prompt(self) -> str:
        style = self.config["style"]
        return f"""Style requirements:
        - Formality: {style['formality']}
        - Register: {style['register']}
        - Notes: {style.get('notes', '')}"""
    
    def get_output_path(self) -> Path:
        template = self.config["output"]["filename_template"]
        date = datetime.now().strftime("%Y%m%d")
        
        # Generate a scenario identifier from the scenario file name
        scenario_id = Path(self.scenario_path).stem
        
        filename = template.format(
            source_code=self.source_code,
            target_code=self.target_code,
            date=date,
            scenario=scenario_id
        ) + ".jsonl"
        return self.output_dir / filename
    
    def get_progress_path(self) -> Path:
        return self.output_dir / f"{self.get_output_path().stem}_progress.json"
    
    def load_progress(self) -> Dict:
        progress_path = self.get_progress_path()
        if progress_path.exists() and self.resume_from_last:
            with open(progress_path, 'r') as f:
                return json.load(f)
        return {"last_processed_id": None, "processed_count": 0}
    
    def save_progress(self, progress: Dict):
        with open(self.get_progress_path(), 'w') as f:
            json.dump(progress, f)
    
    def find_existing_translations(self) -> Dict[str, Dict]:
        """Find all existing translations and validate them against source."""
        translated_items = {}
        invalid_translations = []
        
        # First, build a map of source content by ID
        source_items = {}
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                if self.config["input"]["format"] == "jsonl":
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        item_id = item.get(self.config["input"]["id_field"])
                        content = item.get(self.config["input"]["content_field"])
                        if item_id and content:
                            source_items[item_id] = content
                else:
                    for i, line in enumerate(f):
                        if line.strip():
                            source_items[str(i)] = line.strip()
        except Exception as e:
            print(f"Warning: Error reading source file: {e}")
            return {}
        
        # Create pattern for matching output files (ignoring date)
        scenario_id = Path(self.scenario_path).stem
        base_pattern = self.config["output"]["filename_template"].format(
            source_code=self.source_code,
            target_code=self.target_code,
            scenario=scenario_id,
            date="*"
        ) + ".jsonl"
        
        # Find and validate all matching files
        for file_path in self.output_dir.glob(base_pattern):
            print(f"Validating existing translations in: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    translations = []
                    for line_num, line in enumerate(f, 1):
                        try:
                            translation = json.loads(line)
                            if "id" in translation and "original" in translation:
                                item_id = translation["id"]
                                # Also check for error field to mark for retry
                                if "error" in translation:
                                    invalid_translations.append((file_path, line_num))
                                    continue
                                # Validate against source
                                if (item_id in source_items and 
                                    source_items[item_id] == translation["original"]):
                                    translated_items[item_id] = translation
                                else:
                                    invalid_translations.append((file_path, line_num))
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON at {file_path}:{line_num}")
                            invalid_translations.append((file_path, line_num))
                            continue
                
                # If we found invalid translations, create a new file without them
                if invalid_translations:
                    print(f"Found {len(invalid_translations)} invalid translations in {file_path}")
                    # Create new file without invalid translations
                    new_content = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if not any(inv[0] == file_path and inv[1] == line_num 
                                     for inv in invalid_translations):
                                new_content.append(line.strip())
                    
                    # Write back valid translations only
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for line in new_content:
                            f.write(line + '\n')
                    
                    print(f"Cleaned {file_path} - removed {len(invalid_translations)} invalid translations")
            
            except Exception as e:
                print(f"Warning: Error processing {file_path}: {e}")
                continue
        
        if translated_items:
            print(f"Found {len(translated_items)} valid existing translations")
        
        return translated_items

class TranslationMemory:
    def __init__(self, scenario: TranslationScenario):
        self.memory_path = scenario.base_path / f"{scenario.source_code}_{scenario.target_code}_memory_{os.getpid()}.jsonl"
        self.memory = {}
    
    def find_matches(self, analysis: Dict) -> List[Dict]:
        """Find similar translations in memory."""
        matches = []
        for source_text, entry in self.memory.items():
            # Simple similarity check based on key terms
            if any(term in source_text for term in analysis.get('terms', [])):
                matches.append(entry)
        return matches[:3]  # Return top 3 matches
    
    def add_entry(self, analysis: Dict, translation: Dict):
        """Add a new translation to memory."""
        entry = {
            'source': json.dumps(analysis, ensure_ascii=False),
            'translation': translation
        }
        key = json.dumps(analysis.get('terms', []), ensure_ascii=False)
        self.memory[key] = entry

def create_memory_context(matches: List[Dict]) -> str:
    """Create context string from translation memory matches."""
    if not matches:
        return ""
    
    context_parts = []
    for match in matches:
        try:
            source = json.loads(match['source'])
            context_parts.append(
                f"Similar translation:\n"
                f"Terms: {', '.join(source.get('terms', []))}\n"
                f"Translation: {json.dumps(match['translation'], ensure_ascii=False)}\n"
            )
        except Exception as e:
            print(f"Warning: Error creating memory context: {e}")
            continue
        
        if context_parts:
            return "Previous similar translations for reference:\n" + "\n".join(context_parts)
        return ""

def get_previous_verses(text_id: Optional[str], scenario: TranslationScenario, 
                       num_verses: int = 2) -> List[Dict]:
    """Get previous verses for context."""
    if not text_id:
        return []
    
    try:
        # Parse book and chapter from text_id (e.g., "Genesis 1:5")
        parts = text_id.split()
        if len(parts) != 2:
            return []
            
        book = parts[0]
        chapter_verse = parts[1].split(':')
        if len(chapter_verse) != 2:
            return []
            
        chapter, verse = chapter_verse
        verse = int(verse)
        
        # Read output file to find previous verses
        output_path = scenario.get_output_path()
        if not output_path.exists():
            return []
            
        previous_verses = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry_id = entry.get('id', '')
                    if entry_id.startswith(f"{book} {chapter}:"):
                        entry_verse = int(entry_id.split(':')[1])
                        if entry_verse < verse and entry_verse >= verse - num_verses:
                            previous_verses.append(entry)
                except Exception:
                    continue
                    
        return sorted(previous_verses, 
                     key=lambda x: int(x['id'].split(':')[1]) if x.get('id') else 0)
                     
    except Exception as e:
        print(f"Warning: Error getting previous verses: {e}")
        return []

def is_complete_thought(text: str) -> bool:
    """Check if text ends with terminal punctuation."""
    if not text:
        return False
    # Check last or second-to-last char for terminal punctuation
    # (accounts for possible trailing whitespace)
    last_chars = text.strip()[-2:]
    return any(punct in last_chars for punct in '.!?')

def batch_translations(translations: List[Dict], scenario: TranslationScenario, qa_bot: OpenAI, swarm_client: OpenAI) -> str:
    """Combine multiple translations into a cohesive paragraph."""
    # Only use the translations from the current batch
    originals = [t["original"] for t in translations]
    translated = [t["translation"] for t in translations]
    
    response = swarm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": f"""Combine ONLY these {len(translations)} translations into natural, flowing {scenario.target_label} text.
            Do not repeat or include any previous verses. Only combine these specific verses:
            
            Original {scenario.source_label} texts:
            {' '.join(originals)}
            
            Individual translations:
            {' '.join(translated)}
            
            Important: Return ONLY the combined translation of these specific verses, without including any previous verses."""
        }]
    )
    
    return response.choices[0].message.content.strip()

def setup_agents(scenario: TranslationScenario) -> tuple:
    """Set up the translation agents based on scenario configuration."""
    
    # Configure DeepSeek clients
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Create OpenAI client instances 
    decision_client = OpenAI(
        api_key=api_key,
        # base_url="https://api.deepseek.com"
    )
    translation_client = OpenAI(
        api_key=api_key,
        # base_url="https://api.deepseek.com"
    )

    # Create Swarm instance
    swarm_client = OpenAI(
        api_key=api_key,
        # base_url="https://api.deepseek.com"
    )
    
    def transfer_to_translator(*args, **kwargs):
        return translator_bot
        
    def transfer_to_qa(*args, **kwargs):
        return qa_bot
        
    def transfer_to_consolidator(*args, **kwargs):
        return consolidation_bot

    # Create agents with proper transfer functions
    linguist_bot = OpenAI(
        api_key=api_key,
        # base_url="https://api.deepseek.com"
    )

    translator_bot = OpenAI(
        api_key=api_key,
        # base_url="https://api.deepseek.com"
    )

    consolidation_bot = OpenAI(
        api_key=api_key,
        # base_url="https://api.deepseek.com"
    )

    qa_bot = OpenAI(
        api_key=api_key,
        # base_url="https://api.deepseek.com"
    )
    
    return swarm_client, linguist_bot, translator_bot, qa_bot, consolidation_bot

def call_llm(client: OpenAI, prompt: str, expect_json: bool = False, model: str = None) -> str:
    """Make a single call to the LLM with retries and parsing."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user", 
                    "content": f"{prompt}\n\nIMPORTANT: Return ONLY the requested output, nothing else."
                }]
            )
            content = response.choices[0].message.content.strip()
            
            if expect_json:
                # Multiple approaches to extract valid JSON
                json_result = extract_json(content)
                if json_result:
                    return json_result
                
                print(f"Attempt {attempt+1}: Failed to parse JSON. Content: {content[:200]}...")
                if attempt == max_retries - 1:
                    # Return a minimal valid JSON as fallback on last attempt
                    print(f"Returning minimal valid JSON after {max_retries} failed attempts")
                    return {"phrases": [], "terms": [], "idioms": [], "cultural": []}
                continue
            
            return content
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                print(f"Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                continue
            print(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt == max_retries - 1:
                if expect_json:
                    print(f"Returning minimal valid JSON after {max_retries} failed attempts")
                    return {"phrases": [], "terms": [], "idioms": [], "cultural": []}
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from text using multiple approaches."""
    # Approach 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Approach 2: Find JSON between curly braces
    try:
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            extracted = text[json_start:json_end]
            return json.loads(extracted)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Approach 3: Find JSON between code blocks
    try:
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(code_block_pattern, text)
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    # Approach 4: Try to fix common JSON errors
    try:
        # Replace single quotes with double quotes
        fixed_text = re.sub(r"'([^']*)':", r'"\1":', text)
        # Add quotes to unquoted keys
        fixed_text = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', fixed_text)
        return json.loads(fixed_text)
    except (json.JSONDecodeError, Exception):
        pass
    
    # Approach 5: Construct JSON from key-value pairs
    try:
        result = {}
        # Look for patterns like "key": value or key: value
        kv_pattern = r'["\']?([^"\']+)["\']?\s*:\s*(?:\[([^\]]+)\]|["\']([^"\']+)["\']|(\w+))'
        matches = re.findall(kv_pattern, text)
        for match in matches:
            key = match[0].strip()
            # Determine which value group matched
            if match[1]:  # List value
                value_str = match[1].strip()
                # Split by commas, handle quoted items
                items = re.findall(r'["\']([^"\']+)["\']|(\w+)', value_str)
                value = [item[0] if item[0] else item[1] for item in items if any(item)]
            elif match[2]:  # String value
                value = match[2].strip()
            elif match[3]:  # Word value
                value = match[3].strip()
            else:
                continue
                
            result[key] = value
            
        # Only return if we found something
        if result:
            return result
    except Exception:
        pass
    
    # No valid JSON found
    return None

def analyze_text(client: OpenAI, text: str, source_label: str, target_label: str, book_context: str = "", model: str = None) -> Dict:
    """Analyze text structure (formerly Linguist bot)."""
    prompt = f"""Analyze this {source_label} sentence for translation to {target_label}: '{text}'{book_context}
    
    Return ONLY a JSON object with these components:
    {{
        "phrases": [list of main phrases],
        "terms": [key terms],
        "idioms": [idiomatic expressions],
        "cultural": [cultural references]
    }}"""
    
    analysis = call_llm(client, prompt, expect_json=True, model=model)
    return {
        'phrases': analysis.get('phrases', []),
        'terms': analysis.get('terms', []),
        'idioms': analysis.get('idioms', []),
        'cultural': analysis.get('cultural', [])
    }

def translate_components(client: OpenAI, analysis: Dict, source_label: str, target_label: str, 
                       style_prompt: str, memory_context: str = "", model: str = None) -> Dict:
    """Translate analyzed components (formerly Translator bot)."""
    prompt = f"""Translate these components from {source_label} to {target_label}:
    
    Components to translate:
    {json.dumps(analysis, ensure_ascii=False, indent=2)}
    
    {memory_context}
    {style_prompt}
    
    Return ONLY a JSON object with the translations."""
    
    translations = call_llm(client, prompt, expect_json=True, model=model)
    return {
        'phrases': translations.get('phrases', []),
        'terms': translations.get('terms', []),
        'idioms': translations.get('idioms', []),
        'cultural': translations.get('cultural', [])
    }

def combine_translation(client: OpenAI, original: str, translations: Dict, 
                       source_label: str, target_label: str, style_prompt: str, model: str = None) -> str:
    """Combine translated components (formerly QA bot)."""
    prompt = f"""Provide the final {target_label} translation:
    Original ({source_label}): {original}
    Component translations: {json.dumps(translations, ensure_ascii=False)}
    
    {style_prompt}
    Return ONLY the final translation text."""
    
    return call_llm(client, prompt, model=model)

def consolidate_translation(client: OpenAI, final_translation: str, original: str,
                          book_context: str, previous_verses: List[Dict],
                          source_label: str, target_label: str,
                          style_prompt: str, verse_num: str, model: str = None) -> str:
    """Ensure consistency with previous verses (formerly Consolidation bot)."""
    if not previous_verses:
        return final_translation
        
    context = "\n".join([
        f"Verse {v['id'].split(':')[1]}: {v['translation']}" 
        for v in previous_verses
    ])
    
    prompt = f"""Review and refine this translation in light of the preceding verses:

    {book_context}
    Previous verses from this chapter:
    {context}
    
    Current verse ({verse_num}):
    {final_translation}
    
    Original ({source_label}): {original}
    
    {style_prompt}
    Return ONLY the refined translation of the current verse."""
    
    return call_llm(client, prompt, model=model)

def translate_with_llm(text: str, scenario: TranslationScenario, 
                      client: OpenAI, translation_memory: TranslationMemory,
                      text_id: Optional[str] = None) -> Dict:
    """Main translation function using direct LLM calls."""
    start_time = time.time()
    
    try:
        # Get book context
        book_context = ""
        if text_id:
            try:
                book_name = text_id.split()[0]
                if book_name in scenario.book_id_mapping:
                    book_context = f"\nThis verse is from {scenario.book_id_mapping[book_name]}. "
            except Exception as e:
                print(f"Warning: Error creating book context for '{text_id}': {e}")

        # Get model from scenario config
        model = scenario.config["models"]["translation"]

        # Step 1: Analyze text
        analysis = analyze_text(client, text, scenario.source_label, 
                              scenario.target_label, book_context, model=model)

        # Step 2: Get matching translations and translate components
        matches = translation_memory.find_matches(analysis)
        memory_context = create_memory_context(matches) if matches else ""
        
        translations = translate_components(client, analysis, 
                                         scenario.source_label, scenario.target_label,
                                         scenario.get_style_prompt(), memory_context, model=model)

        # Add to translation memory
        try:
            translation_memory.add_entry(analysis, translations)
        except Exception as e:
            print(f"Warning: Failed to add to translation memory: {e}")

        # Step 3: Combine translations
        final_translation = combine_translation(client, text, translations,
                                             scenario.source_label, scenario.target_label,
                                             scenario.get_style_prompt(), model=model)

        # Step 4: Consolidate with previous verses
        try:
            previous_verses = get_previous_verses(text_id, scenario)
            if previous_verses:
                verse_num = text_id.split(':')[1] if text_id else "unknown"
                final_translation = consolidate_translation(
                    client, final_translation, text, book_context,
                    previous_verses, scenario.source_label, scenario.target_label,
                    scenario.get_style_prompt(), verse_num, model=model
                )
        except Exception as e:
            print(f"Warning: Consolidation step failed: {e}")

        # Create result
        translation_time = time.time() - start_time
        result = {
            "source_lang": scenario.source_code,
            "target_lang": scenario.target_code,
            "original": text,
            "translation": final_translation,
            "translation_time": round(translation_time, 2),
            "calver": datetime.now().strftime("%Y.%m.%d")
        }
        
        if text_id:
            result["id"] = text_id
            
        return result
        
    except Exception as e:
        print(f"Error translating text: '{text}'")
        print(f"Error details: {str(e)}")
        return {
            "source_lang": scenario.source_code,
            "target_lang": scenario.target_code,
            "original": text,
            "translation": f"[Translation failed: {str(e)}]",
            "error": str(e),
            "id": text_id if text_id else None,
            "calver": datetime.now().strftime("%Y.%m.%d")
        }

def process_batch(batch_data: tuple, scenario: TranslationScenario, output_path: Path):
    """Process a single batch of translations."""
    batch_id, lines = batch_data
    
    print(f"Starting batch {batch_id} with {len(lines)} items")
    
    # Set up agents for this process
    swarm_client, linguist_bot, translator_bot, qa_bot, consolidation_bot = setup_agents(scenario)
    
    # Create a process-specific translation memory
    translation_memory = TranslationMemory(scenario)
    
    for item in lines:
        try:
            # Safely fetch the text field name from scenario.config
            source_field = scenario.config["input"].get("content_field", "content")
            # Fallback logic if the chosen field doesn't exist in the item
            if source_field not in item:
                # First try "translation" for daisy chaining translations
                if "translation" in item:
                    content = item["translation"]
                # Then fall back to "original" if translation isn't available
                elif "original" in item:
                    content = item["original"]
                else:
                    # Final fallback - handle gracefully or raise
                    raise KeyError(f"No '{source_field}', 'translation', or 'original' field found in item:\n{item}")
            else:
                content = item[source_field]
            
            text_id = item.get(scenario.config["input"]["id_field"])
        except KeyError as e:
            print(f"Error accessing fields in batch {batch_id}:")
            print(f"Item content: {item}")
            raise
        
        while True:
            try:
                result = translate_with_llm(
                    content, scenario, swarm_client, translation_memory, text_id
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    print(f"Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                raise
        
        # Use a lock for file writing
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"✓ Translated: {text_id or 'unknown'}")
        except Exception as e:
            print(f"Error saving translation to {output_path}: {str(e)}")
            raise

def process_input_file(scenario: TranslationScenario):
    """Process input file according to scenario configuration using parallel processing."""
    try:
        start_time = time.time()
        
        # Find and validate existing translations
        existing_translations = scenario.find_existing_translations()
        
        # Read input file
        lines = []
        with open(scenario.input_path, 'r', encoding='utf-8') as f:
            input_format = scenario.config["input"]["format"].lower()
            
            if input_format == "jsonl":
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        item = json.loads(line)
                        # Map the reference ID if present
                        item_id = item.get(scenario.config["input"]["id_field"])
                        if item_id:
                            item[scenario.config["input"]["id_field"]] = scenario.map_reference_id(item_id)
                        
                        # Skip if we already have a valid translation
                        if item_id and item_id in existing_translations:
                            continue
                        lines.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}:")
                        print(f"Line content: {line}")
                        raise
                        
            elif input_format == "csv":
                csv_reader = csv.DictReader(f)
                for i, row in enumerate(csv_reader, 1):
                    try:
                        # Create item dictionary with mapped fields
                        item = {
                            scenario.config["input"]["id_field"]: scenario.map_reference_id(row["id"]),
                            scenario.config["input"]["content_field"]: row["content"]
                        }
                        
                        # Skip if we already have a valid translation
                        if item[scenario.config["input"]["id_field"]] in existing_translations:
                            continue
                        lines.append(item)
                    except Exception as e:
                        print(f"Error processing CSV row {i}:")
                        print(f"Row content: {row}")
                        raise
                        
            else:
                # Plain text format (unchanged)
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    item_id = str(i)
                    if item_id not in existing_translations:
                        lines.append({"content": line.strip(), "id": item_id})
        
        if not lines:
            print("No new lines to process")
            return
        
        total_lines = len(lines)
        print(f"Processing {total_lines} new lines...")
        
        # Split lines into batches for parallel processing
        num_processes = min(mp.cpu_count(), len(lines))
        batch_size = max(1, len(lines) // num_processes)
        batches = []
        
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            batches.append((i // batch_size, batch))
        
        # Set up multiprocessing pool
        output_path = scenario.get_output_path()
        with mp.Pool(num_processes) as pool:
            try:
                process_batch_partial = partial(process_batch, 
                                             scenario=scenario,
                                             output_path=output_path)
                
                pool.map(process_batch_partial, batches)
            except KeyboardInterrupt:
                print("\nGracefully shutting down workers...")
                pool.terminate()
                pool.join()
                print("Workers shut down successfully")
                return
            finally:
                pool.close()
                pool.join()
        
        total_time = time.time() - start_time
        # Count actual translations completed by counting lines in output file
        with open(output_path, 'r') as f:
            completed_count = sum(1 for _ in f)
        
        print(f"\nCompleted {completed_count}/{total_lines} translations")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per translation: {total_time/completed_count:.1f} seconds")
        
        # Update progress after successful completion
        progress = {
            "last_processed_id": lines[-1].get(scenario.config["input"]["id_field"]) if lines else None,
            "processed_count": len(lines)
        }
        scenario.save_progress(progress)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        return
    except Exception as e:
        print(f"Error processing input file: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate text using agent swarm')
    parser.add_argument('scenario', help='Path to scenario configuration file')
    
    args = parser.parse_args()
    scenario = TranslationScenario(args.scenario)
    process_input_file(scenario)