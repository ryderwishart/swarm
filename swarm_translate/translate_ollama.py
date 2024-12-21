#Tailored for use with Ollama specifically, ensuring structured outputs.
#ADDED FEATURE: 
#While the command "python translate_ollama.py PATH/TO/SCENARIO.json" starts from the beginning of the Bible,
#there is an optional parameter to translate a single book, eg "python translate_ollama.py PATH/TO/SCENARIO.json -b Jonah" translates only the book of Jonah.
from typing import List, Dict, Optional
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
from ollama import chat
from pydantic import BaseModel

class ParsedVerse(BaseModel):
    phrases: list[str]
    terms: list[str]
    idioms: list[str]
    cultural: list[str]

class TranslationScenario:
    def __init__(self, scenario_path: str, book: str):
        self.scenario_path = Path(scenario_path)
        self.book = book
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
        filename = template.format(
            source_code=self.source_code,
            target_code=self.target_code,
            date=date
        )
        if self.book is not None:
            filename += "-" + self.book
        filename += ".jsonl"
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
            
def get_linguistic_directives(scenario: TranslationScenario) -> str:
    return scenario.config.get("linguistic_directives", "")

def translate_with_agents(text: str, scenario: TranslationScenario, 
                        text_id: Optional[str] = None) -> Dict:
    """Main function to coordinate the translation process between agents."""
    start_time = time.time()
    # Step 1: Linguist bot analyzes the sentence
    linguist_instructions=f"""You are an expert linguist who excels at breaking down {scenario.source_label} sentences for translation.
        Analyze the given sentence and output ONLY a JSON object with these components:
        {{
            "phrases": [list of main phrases],
            "terms": [key terms],
            "idioms": [idiomatic expressions],
            "cultural": [cultural references]
        }}"""
    messages = [
        {"role": "system", "content": linguist_instructions},
        {"role": "user", "content": f"Analyze this {scenario.source_label} sentence for translation to {scenario.target_label}: '{text}'"}
    ]
    
    if hasattr(scenario, 'linguistic_directives'):
        messages.insert(0, {
            "role": "system",
            "content": get_linguistic_directives(scenario)
        })
    
    response = chat(
        model=scenario.translation_model,
        messages=messages,
        format=ParsedVerse.model_json_schema()
    )
    analysis = ParsedVerse.model_validate_json(response.message.content)
    # Step 2: Translator bot translates components
    translator_instructions=f"""You are a precise translation expert specializing in {scenario.source_label} to {scenario.target_label} translation.
        Translate the components provided by the linguist into {scenario.target_label}.
        {scenario.get_style_prompt()}
        Output ONLY a JSON object with the translations:
        {{
            "phrases": [translated phrases],
            "terms": [translated terms],
            "idioms": [translated expressions],
            "cultural": [translated references]
        }}"""
    messages = [
        {"role": "system", "content": translator_instructions},
        {"role": "user","content": f"Translate these components from {scenario.source_label} to {scenario.target_label}: {analysis}"}
    ]
    
    if hasattr(scenario, 'linguistic_directives'):
        messages.insert(0, {
            "role": "system",
            "content": get_linguistic_directives(scenario)
        })
    response = chat(
        model=scenario.translation_model,
        messages=messages,
        format=ParsedVerse.model_json_schema()
    )
    translations = ParsedVerse.model_validate_json(response.message.content)
    # Step 3: QA bot provides final translation
    qa_instructions=f"""You are a translation quality assurance expert for {scenario.target_label}.
        Given the original {scenario.source_label} text and component translations, provide the FINAL TRANSLATION ONLY.
        {scenario.get_style_prompt()}
        Do not include any explanations, notes, or additional text.
        Output only the translated text in {scenario.target_label}."""
    messages = [
        {"role": "system", "content": qa_instructions},
        {
            "role": "user",
            "content": f"""Provide the final {scenario.target_label} translation:
                Original ({scenario.source_label}): {text}
                Component translations: {translations}"""
        }
    ]
    
    if hasattr(scenario, 'linguistic_directives'):
        messages.insert(0, {
            "role": "system",
            "content": get_linguistic_directives(scenario)
        })
        
    response = chat(
        model=scenario.translation_model,
        messages=messages
    )
    
    translation_time = time.time() - start_time
    
    result = {
        "source_lang": scenario.source_code,
        "source_label": scenario.source_label,
        "target_lang": scenario.target_code,
        "target_label": scenario.target_label,
        "original": text,
        "translation": response.message.content,
        "translation_time": round(translation_time, 2),
        "model": scenario.translation_model,
        "calver": datetime.now().strftime("%Y.%m.%d")
    }
    
    if text_id:
        result["id"] = text_id
    
    return result

def process_input_file(scenario: TranslationScenario):
    """Process input file according to scenario configuration."""
    
    # Load progress if resuming
    progress = scenario.load_progress()
    last_id = progress["last_processed_id"]
    processed_count = progress["processed_count"]
    # Read input file with better error handling
    try:
        with open(scenario.input_path, 'r', encoding='utf-8') as f:
            if scenario.config["input"]["format"] == "jsonl":
                lines = []
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        item = json.loads(line)
                        if scenario.book is not None:
                            if scenario.book not in item['id'] or scenario.book[0] is not item['id'][0]: #Second conditional so that "John" doesn't include "1 John"                              
                                continue
                        lines.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}:")
                        print(f"Line content: {line}")
                        print(f"Error details: {str(e)}")
                        raise
                
                # Skip already processed items if resuming
                if last_id is not None:
                    lines = lines[processed_count:]
            else:
                lines = [{"content": line.strip(), "id": str(i)} 
                        for i, line in enumerate(f) if line.strip()]
    except FileNotFoundError:
        print(f"Input file not found: {scenario.input_path}")
        print(f"Working directory: {Path.cwd()}")
        print(f"Absolute path attempted: {scenario.input_path.absolute()}")
        raise
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        raise
    
    if not lines:
        print("Warning: No lines were read from the input file")
        return
    
    # Process in batches
    output_path = scenario.get_output_path()
    for i, item in enumerate(lines):
        try:
            content = item[scenario.config["input"]["content_field"]]
            text_id = item.get(scenario.config["input"]["id_field"])
        except KeyError as e:
            print(f"Error accessing fields in item {i}:")
            print(f"Item content: {item}")
            print(f"Expected fields: {scenario.config['input']['content_field']} and {scenario.config['input']['id_field']}")
            raise
        
        result = translate_with_agents(content, scenario, text_id)
        
        # Append to output file
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Update and save progress periodically
        if (i + 1) % scenario.save_frequency == 0:
            progress = {
                "last_processed_id": text_id,
                "processed_count": processed_count + i + 1
            }
            scenario.save_progress(progress)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate text using agent swarm')
    parser.add_argument('scenario', help='Path to scenario configuration file')
    parser.add_argument('-b', '--book', help='OPTIONAL: Only translate this book')
    
    args = parser.parse_args()
    scenario = TranslationScenario(args.scenario, args.book)
    process_input_file(scenario) 
