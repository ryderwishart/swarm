from swarm import Swarm, Agent
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
                                # Validate against source
                                if (item_id in source_items and 
                                    source_items[item_id] == translation["original"]):
                                    translated_items[item_id] = translation
                                else:
                                    invalid_translations.append((file_path, line_num))
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON at {file_path}:{line_num}")
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
        self.memory_path = scenario.base_path / f"{scenario.source_code}_{scenario.target_code}_translation_memory.jsonl"
        self.memory = self._load_memory()
        
    def _load_memory(self) -> Dict[str, Dict]:
        """Load existing translation memory."""
        memory = {}
        if self.memory_path.exists():
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        # Use source text as key
                        memory[entry['source']] = entry
        return memory
    
    def _save_entry(self, entry: Dict):
        """Save a new entry to the translation memory file."""
        with open(self.memory_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def similar(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        """Check if two texts are similar using sequence matcher."""
        return SequenceMatcher(None, text1, text2).ratio() > threshold
    
    def find_matches(self, components: Dict[str, List[str]], 
                    threshold: float = 0.85) -> Dict[str, List[Dict]]:
        """Find matching translations for each component."""
        matches = {
            'phrases': [],
            'terms': [],
            'idioms': [],
            'cultural': []
        }
        
        for category in matches.keys():
            for item in components.get(category, []):
                for source, entry in self.memory.items():
                    if self.similar(item, source, threshold):
                        matches[category].append({
                            'source': source,
                            'target': entry['target'],
                            'similarity': SequenceMatcher(None, item, source).ratio()
                        })
        
        return matches
    
    def add_entry(self, source_components: Dict[str, List[str]], 
                 target_components: Dict[str, List[str]]):
        """Add new translations to memory."""
        for category in ['phrases', 'terms', 'idioms', 'cultural']:
            sources = source_components.get(category, [])
            targets = target_components.get(category, [])
            
            for source, target in zip(sources, targets):
                if source not in self.memory:
                    entry = {
                        'source': source,
                        'target': target,
                        'category': category,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.memory[source] = entry
                    self._save_entry(entry)

def is_complete_thought(text: str) -> bool:
    """Check if text ends with terminal punctuation."""
    if not text:
        return False
    # Check last or second-to-last char for terminal punctuation
    # (accounts for possible trailing whitespace)
    last_chars = text.strip()[-2:]
    return any(punct in last_chars for punct in '.!?')

def batch_translations(translations: List[Dict], scenario: TranslationScenario, qa_bot: Agent, swarm_client: Swarm) -> str:
    """Combine multiple translations into a cohesive paragraph."""
    # Only use the translations from the current batch
    originals = [t["original"] for t in translations]
    translated = [t["translation"] for t in translations]
    
    response = swarm_client.run(
        agent=qa_bot,
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
    
    return response.messages[-1]["content"]

def setup_agents(scenario: TranslationScenario) -> tuple:
    """Set up the translation agents based on scenario configuration."""
    
    # Configure DeepSeek clients
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

    # Create OpenAI client instances with DeepSeek base URL
    decision_client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    translation_client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # Create Swarm instance
    swarm_client = Swarm(client=decision_client)
    
    def transfer_to_translator(*args, **kwargs):
        return translator_bot

    def transfer_to_qa(*args, **kwargs):
        return qa_bot

    def transfer_to_consolidator(*args, **kwargs):
        return consolidation_bot

    # Define the specialized agents
    linguist_bot = Agent(
        name="Linguist",
        instructions=f"""You are an expert linguist who excels at breaking down {scenario.source_label} sentences for translation.
        Analyze the given sentence and output ONLY a JSON object with these components:
        {{
            "phrases": [list of main phrases],
            "terms": [key terms],
            "idioms": [idiomatic expressions],
            "cultural": [cultural references]
        }}""",
        functions=[transfer_to_translator],
        client=decision_client,
        model="deepseek-chat"
    )

    translator_bot = Agent(
        name="Translator",
        instructions=f"""You are a precise translation expert specializing in {scenario.source_label} to {scenario.target_label} translation.
        Translate the components provided by the linguist into {scenario.target_label}.
        {scenario.get_style_prompt()}
        Output ONLY a JSON object with the translations:
        {{
            "phrases": [translated phrases],
            "terms": [translated terms],
            "idioms": [translated expressions],
            "cultural": [translated references]
        }}""",
        functions=[transfer_to_qa],
        client=translation_client,
        model="deepseek-chat"
    )

    consolidation_bot = Agent(
        name="Consolidation Editor",
        instructions=f"""You are an Oxford professor with decades of experience editing academic works, poetry, and literature in {scenario.target_label}.
        Your task is to review and refine translations while maintaining consistency with preceding verses.
        
        Consider:
        1. Terminology consistency
        2. Style and tone continuity
        3. Thematic coherence
        4. Natural flow between verses
        5. Cultural and contextual alignment
        
        {scenario.get_style_prompt()}
        
        Output ONLY the refined translation of the current verse, ensuring it flows naturally with previous verses while maintaining accuracy.""",
        client=translation_client,
        model="deepseek-chat"
    )

    qa_bot = Agent(
        name="QA Expert",
        instructions=f"""You are a translation quality assurance expert for {scenario.target_label}.
        Given the original {scenario.source_label} text and component translations, provide the FINAL TRANSLATION ONLY.
        {scenario.get_style_prompt()}
        Do not include any explanations, notes, or additional text.
        Output only the translated text in {scenario.target_label}.""",
        functions=[transfer_to_consolidator],
        client=translation_client,
        model="deepseek-chat"
    )
    
    return swarm_client, linguist_bot, translator_bot, qa_bot, consolidation_bot

def get_previous_verses(current_id: str, scenario: TranslationScenario, n: int = 10) -> List[Dict]:
    """Retrieve the n previous verses from the output file."""
    verses = []
    output_path = scenario.get_output_path()
    
    if not output_path.exists():
        return verses

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            translations = [json.loads(line) for line in f if line.strip()]
            
        # Find the current verse's position
        current_index = next((i for i, t in enumerate(translations) 
                            if t.get('id') == current_id), None)
        
        if current_index is not None:
            # Get up to n previous verses
            start_index = max(0, current_index - n)
            verses = translations[start_index:current_index]
    except Exception as e:
        print(f"Warning: Error reading previous verses: {e}")
    
    return verses

def translate_with_agents(text: str, scenario: TranslationScenario, 
                        swarm_client: Swarm, linguist_bot: Agent, 
                        translator_bot: Agent, qa_bot: Agent,
                        consolidation_bot: Agent,
                        translation_memory: TranslationMemory,
                        text_id: Optional[str] = None) -> Dict:
    """Main function to coordinate the translation process between agents."""
    start_time = time.time()
    
    try:
        # Step 1: Linguist bot analyzes the sentence
        response = swarm_client.run(
            agent=linguist_bot,
            messages=[{
                "role": "user",
                "content": f"Analyze this {scenario.source_label} sentence for translation to {scenario.target_label}: '{text}'"
            }]
        )
        analysis = json.loads(response.messages[-1]["content"])
        
        # Find matching translations from memory
        matches = translation_memory.find_matches(analysis)
        
        # Step 2: Translator bot translates components with memory context
        memory_context = ""
        if any(matches.values()):
            memory_context = "Previous translations from similar content:\n"
            for category, category_matches in matches.items():
                if category_matches:
                    memory_context += f"\n{category.title()}:\n"
                    for match in category_matches:
                        memory_context += f"- {match['source']} → {match['target']}\n"
        
        response = swarm_client.run(
            agent=translator_bot,
            messages=[{
                "role": "user",
                "content": f"""Translate these components from {scenario.source_label} to {scenario.target_label}:
                
                Components to translate:
                {json.dumps(analysis, ensure_ascii=False, indent=2)}
                
                {memory_context}
                
                Please maintain consistency with previous translations where appropriate."""
            }]
        )
        translations = json.loads(response.messages[-1]["content"])
        
        # Add new translations to memory
        translation_memory.add_entry(analysis, translations)
        
        # Step 3: QA bot always provides final translation
        response = swarm_client.run(
            agent=qa_bot,
            messages=[{
                "role": "user",
                "content": f"""Provide the final {scenario.target_label} translation:
                Original ({scenario.source_label}): {text}
                Component translations: {translations}"""
            }]
        )
        final_translation = response.messages[-1]["content"]
        
        # New Step 4: Consolidation with previous verses
        previous_verses = get_previous_verses(text_id, scenario)
        
        if previous_verses:
            context = "\n".join([
                f"Verse {i+1}: {v['translation']}" 
                for i, v in enumerate(previous_verses)
            ])
            
            response = swarm_client.run(
                agent=consolidation_bot,
                messages=[{
                    "role": "user",
                    "content": f"""Review and refine this translation in light of the preceding verses:

                    Previous verses:
                    {context}
                    
                    Current verse translation:
                    {final_translation}
                    
                    Original ({scenario.source_label}): {text}
                    
                    Please provide the refined translation of ONLY the current verse, ensuring it flows naturally with the previous verses while maintaining accuracy."""
                }]
            )
            
            final_translation = response.messages[-1]["content"].strip()
        
        translation_time = time.time() - start_time
        
        result = {
            "source_lang": scenario.source_code,
            "source_label": scenario.source_label,
            "target_lang": scenario.target_code,
            "target_label": scenario.target_label,
            "original": text,
            "translation": final_translation.strip(),  # Clean up any extra whitespace
            "translation_time": round(translation_time, 2),
            "model": translator_bot.model,
            "calver": datetime.now().strftime("%Y.%m.%d")
        }
        
        if text_id:
            result["id"] = text_id
        
        return result
        
    except Exception as e:
        print(f"Error translating text: '{text}'")
        print(f"Error details: {str(e)}")
        # Return a minimal result indicating failure
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
    
    # Set up agents and translation memory for this process
    swarm_client, linguist_bot, translator_bot, qa_bot, consolidation_bot = setup_agents(scenario)
    translation_memory = TranslationMemory(scenario)
    
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
                result = translate_with_agents(
                    content, scenario, swarm_client, 
                    linguist_bot, translator_bot, qa_bot, consolidation_bot,
                    translation_memory, text_id
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