from swarm import Swarm, Agent
from typing import List, Dict, Optional
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
from openai import OpenAI

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

def setup_agents(scenario: TranslationScenario, model: str) -> tuple:
    """Set up three independent translators and a consistency checker."""
    
    # Configure OpenAI client with faster/cheaper model
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed"
    )

    # Create Swarm instance
    swarm_client = Swarm(client=client)
    
    # Create three independent translator agents
    translator_instructions = f"""You are a precise translation expert specializing in {scenario.source_label} to {scenario.target_label} translation.
    Translate ONLY the exact text provided into {scenario.target_label}.
    {scenario.get_style_prompt()}
    Provide ONLY the translation of the given text, no explanations or notes.
    Do not continue translating beyond the provided text."""

    translator1 = Agent(
        name="Translator1",
        instructions=translator_instructions,
        client=client,
        model=model
    )

    translator2 = Agent(
        name="Translator2",
        instructions=translator_instructions,
        client=client,
        model=model
    )

    translator3 = Agent(
        name="Translator3",
        instructions=translator_instructions,
        client=client,
        model=model
    )

    consistency_checker = Agent(
        name="ConsistencyChecker",
        instructions=f"""You are a translation consistency expert.
        Compare the three translations provided and determine:
        1. If they are consistent enough (conveying the same meaning)
        2. Which translation is the best, if they are consistent
        3. If they are inconsistent, explain why and suggest a revised translation
        
        Output a JSON object:
        {{
            "consistent": true/false,
            "best_translation": "selected or revised translation",
            "notes": "explanation if inconsistent"
        }}""",
        client=client,
        model=model
    )
    
    return swarm_client, translator1, translator2, translator3, consistency_checker

def translate_with_agents(text: str, scenario: TranslationScenario, 
                        swarm_client: Swarm, translator1: Agent, 
                        translator2: Agent, translator3: Agent,
                        consistency_checker: Agent,
                        text_id: Optional[str] = None) -> Dict:
    """Perform three independent translations and check consistency."""
    start_time = time.time()
    
    # Get three independent translations
    translations = []
    for translator in [translator1, translator2, translator3]:
        response = swarm_client.run(
            agent=translator,
            messages=[{
                "role": "user",
                "content": f"Translate this exact {scenario.source_label} text to {scenario.target_label} (translate ONLY this text): '{text}'"
            }]
        )
        translations.append(response.messages[-1]["content"].strip())
    
    # Check consistency
    response = swarm_client.run(
        agent=consistency_checker,
        messages=[{
            "role": "user",
            "content": f"""Compare these translations:
            Original ({scenario.source_label}): {text}
            Translation 1: {translations[0]}
            Translation 2: {translations[1]}
            Translation 3: {translations[2]}"""
        }]
    )
    
    try:
        consistency_result = json.loads(response.messages[-1]["content"])
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        consistency_result = {
            "consistent": False,
            "best_translation": translations[0],  # Use first translation as fallback
            "notes": "Error parsing consistency check result"
        }
    
    translation_time = time.time() - start_time
    
    result = {
        "source_lang": scenario.source_code,
        "source_label": scenario.source_label,
        "target_lang": scenario.target_code,
        "target_label": scenario.target_label,
        "original": text,
        "translation": consistency_result["best_translation"],
        "consistent": consistency_result["consistent"],
        "consistency_notes": consistency_result["notes"],
        "all_translations": translations,
        "translation_time": round(translation_time, 2),
        "model": scenario.translation_model,
        "calver": datetime.now().strftime("%Y.%m.%d")
    }
    
    if text_id:
        result["id"] = text_id
    
    return result

def process_input_file(scenario: TranslationScenario):
    """Process input file according to scenario configuration."""
    # Set up agents
    swarm_client, translator1, translator2, translator3, consistency_checker = setup_agents(scenario, scenario.translation_model)
    
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
        
        result = translate_with_agents(
            content, scenario, swarm_client,
            translator1, translator2, translator3,
            consistency_checker, text_id
        )
        
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
    
    args = parser.parse_args()
    scenario = TranslationScenario(args.scenario)
    process_input_file(scenario) 