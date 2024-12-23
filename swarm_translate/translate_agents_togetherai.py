from typing import List, Dict, Optional, Tuple
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
import os
from dotenv import load_dotenv
import re
import multiprocessing as mp
from functools import partial
from openai import OpenAI
from together import Together
import random
from typing import Callable, TypeVar, Any

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
    
    def get_style_prompt(self) -> str:
        style = self.config["style"]
        return f"""Style requirements:
        - Formality: {style['formality']}
        - Register: {style['register']}
        - Notes: {style.get('notes', '')}"""
    
    def get_output_path(self) -> Path:
        template = self.config["output"]["filename_template"]
        date = datetime.now().strftime("%Y%m%d")
        filename = template.format(
            source_code=self.source_code,
            target_code=self.target_code,
            date=date
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
        rate_limited_translations = []  # New list for rate-limited translations
        
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
        base_pattern = self.config["output"]["filename_template"].format(
            source_code=self.source_code,
            target_code=self.target_code,
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
                                
                                # Check for rate limit failures
                                if ("translation" in translation and 
                                    isinstance(translation["translation"], str) and
                                    ("rate limit" in translation["translation"].lower() or
                                     "rate_limit" in translation.get("error", "").lower())):
                                    rate_limited_translations.append((file_path, line_num))
                                    continue
                                
                                # Validate against source
                                if (item_id in source_items and 
                                    source_items[item_id] == translation["original"]):
                                    translated_items[item_id] = translation
                                else:
                                    invalid_translations.append((file_path, line_num))
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON at {file_path}:{line_num}")
                            continue
                
                # If we found invalid or rate-limited translations, create a new file without them
                if invalid_translations or rate_limited_translations:
                    print(f"Found {len(invalid_translations)} invalid translations and "
                          f"{len(rate_limited_translations)} rate-limited translations in {file_path}")
                    
                    # Create new file without invalid translations
                    new_content = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if not any(inv[0] == file_path and inv[1] == line_num 
                                     for inv in invalid_translations + rate_limited_translations):
                                new_content.append(line.strip())
                    
                    # Write back valid translations only
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for line in new_content:
                            f.write(line + '\n')
                    
                    print(f"Cleaned {file_path} - removed {len(invalid_translations)} invalid "
                          f"and {len(rate_limited_translations)} rate-limited translations")
            
            except Exception as e:
                print(f"Warning: Error processing {file_path}: {e}")
                continue
        
        if translated_items:
            print(f"Found {len(translated_items)} valid existing translations")
            if rate_limited_translations:
                print(f"Will retry {len(rate_limited_translations)} rate-limited translations")
        
        return translated_items

def is_complete_thought(text: str) -> bool:
    """Check if text ends with terminal punctuation."""
    if not text:
        return False
    # Check last or second-to-last char for terminal punctuation
    # (accounts for possible trailing whitespace)
    last_chars = text.strip()[-2:]
    return any(punct in last_chars for punct in '.!?')

def setup_client() -> Together:
    """Set up the Together AI client."""
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment variables")
    return Together(api_key=api_key)

T = TypeVar('T')

def exponential_backoff(func: Callable[..., T], *args, max_retries: int = 5, **kwargs) -> T:
    """
    Execute a function with exponential backoff for rate limits.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        The result of the function call
    
    Raises:
        Exception: If max retries are exceeded
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" not in error_msg.lower():
                raise
            
            if attempt == max_retries - 1:
                raise Exception(f"Max retries ({max_retries}) exceeded: {error_msg}")
            
            # Extract wait time from error message if available
            wait_time = None
            try:
                import re
                match = re.search(r'try again in (\d+\.?\d*)s', error_msg)
                if match:
                    wait_time = float(match.group(1))
            except:
                pass
            
            # If we couldn't extract wait time, use exponential backoff with jitter
            if wait_time is None:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
            else:
                # Add small jitter to extracted wait time
                wait_time += random.uniform(0, 2)
            
            print(f"Rate limit hit, waiting {wait_time:.2f} seconds (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)

def make_together_call(client: Together, messages: List[Dict], model: str) -> str:
    """Make a Together AI API call with retry logic."""
    response = exponential_backoff(
        client.chat.completions.create,
        messages=messages,
        model=model,
        max_tokens=1024,  # Adjust as needed
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"]
    )
    
    # Handle streaming response
    full_response = ""
    for token in response:
        if hasattr(token, 'choices'):
            full_response += token.choices[0].delta.content or ""
    
    return full_response.strip()

def analyze_text(client: Together, text: str, scenario: TranslationScenario) -> str:
    """Analyze the text structure and components."""
    messages = [{
        "role": "system",
        "content": f"""You are an expert linguist who excels at breaking down {scenario.source_label} sentences for translation.
        Analyze the given sentence and output ONLY a JSON object with these components:
        {{
            "phrases": [list of main phrases],
            "terms": [key terms],
            "idioms": [idiomatic expressions],
            "cultural": [cultural references]
        }}"""
    }, {
        "role": "user",
        "content": f"Analyze this {scenario.source_label} sentence: '{text}'"
    }]
    
    return make_together_call(client, messages, scenario.routing_model)

def translate_components(client: Together, analysis: str, text: str, scenario: TranslationScenario) -> str:
    """Translate the analyzed components."""
    messages = [{
        "role": "system",
        "content": f"""You are a precise translation expert specializing in {scenario.source_label} to {scenario.target_label} translation.
        {scenario.get_style_prompt()}
        Translate the components into {scenario.target_label} and output ONLY a JSON object with the translations:
        {{
            "phrases": [translated phrases],
            "terms": [translated terms],
            "idioms": [translated expressions],
            "cultural": [translated references]
        }}"""
    }, {
        "role": "user",
        "content": f"Original text: '{text}'\nAnalysis: {analysis}"
    }]
    
    return make_together_call(client, messages, scenario.translation_model)

def finalize_translation(client: Together, original: str, translations: str, scenario: TranslationScenario) -> str:
    """Create the final translation."""
    messages = [{
        "role": "system",
        "content": f"""You are a translation quality assurance expert for {scenario.target_label}.
        {scenario.get_style_prompt()}
        Provide the FINAL TRANSLATION ONLY.
        Do not include any explanations, notes, or additional text."""
    }, {
        "role": "user",
        "content": f"""Original ({scenario.source_label}): {original}
        Component translations: {translations}"""
    }]
    
    return make_together_call(client, messages, scenario.translation_model)

def translate_with_together(text: str, scenario: TranslationScenario, 
                          client: Together, text_id: Optional[str] = None) -> Dict:
    """Main function to coordinate the translation process."""
    start_time = time.time()
    
    try:
        # Step 1: Analyze the text
        analysis = analyze_text(client, text, scenario)
        
        # Step 2: Translate components
        translations = translate_components(client, analysis, text, scenario)
        
        # Step 3: Create final translation
        final_translation = finalize_translation(client, text, translations, scenario)
        
        translation_time = time.time() - start_time
        
        result = {
            "source_lang": scenario.source_code,
            "source_label": scenario.source_label,
            "target_lang": scenario.target_code,
            "target_label": scenario.target_label,
            "original": text,
            "translation": final_translation.strip(),
            "translation_time": round(translation_time, 2),
            "model": scenario.translation_model,
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
    
    # Set up Together AI client
    client = setup_client()
    
    for item in lines:
        try:
            content = item[scenario.config["input"]["content_field"]]
            text_id = item.get(scenario.config["input"]["id_field"])
        except KeyError as e:
            print(f"Error accessing fields in batch {batch_id}:")
            print(f"Item content: {item}")
            raise
        
        while True:
            try:
                result = translate_with_together(
                    content, scenario, client, text_id
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    print(f"Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                raise
        
        # Save individual translation immediately
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
        with open(scenario.input_path, 'r', encoding='utf-8') as f:
            if scenario.config["input"]["format"] == "jsonl":
                lines = []
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        item = json.loads(line)
                        # Skip if we already have a valid translation
                        item_id = item.get(scenario.config["input"]["id_field"])
                        if item_id and item_id in existing_translations:
                            continue
                        lines.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}:")
                        print(f"Line content: {line}")
                        raise
            else:
                lines = []
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