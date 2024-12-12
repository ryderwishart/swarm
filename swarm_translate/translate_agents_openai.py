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

# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / '.env')

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
    originals = [t["original"] for t in translations]
    translated = [t["translation"] for t in translations]
    
    response = swarm_client.run(
        agent=qa_bot,
        messages=[{
            "role": "user",
            "content": f"""Combine these translations into natural, flowing {scenario.target_label} text.
            Maintain the meaning and style while ensuring the translations connect smoothly.
            
            Original {scenario.source_label} texts:
            {' '.join(originals)}
            
            Individual translations:
            {' '.join(translated)}"""
        }]
    )
    
    return response.messages[-1]["content"]

def setup_agents(scenario: TranslationScenario) -> tuple:
    """Set up the translation agents based on scenario configuration."""
    
    # Configure OpenAI clients
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    decision_client = OpenAI(api_key=api_key)
    translation_client = OpenAI(api_key=api_key)

    # Create Swarm instance
    swarm_client = Swarm(client=decision_client)
    
    def transfer_to_translator(*args, **kwargs):
        return translator_bot

    def transfer_to_qa(*args, **kwargs):
        return qa_bot

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
        client=translation_client,
        model=scenario.translation_model
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
        model=scenario.translation_model
    )

    qa_bot = Agent(
        name="QA Expert",
        instructions=f"""You are a translation quality assurance expert for {scenario.target_label}.
        Given the original {scenario.source_label} text and component translations, provide the FINAL TRANSLATION ONLY.
        {scenario.get_style_prompt()}
        Do not include any explanations, notes, or additional text.
        Output only the translated text in {scenario.target_label}.""",
        client=translation_client,
        model=scenario.translation_model
    )
    
    return swarm_client, linguist_bot, translator_bot, qa_bot

def translate_with_agents(text: str, scenario: TranslationScenario, 
                        swarm_client: Swarm, linguist_bot: Agent, 
                        translator_bot: Agent, qa_bot: Agent,
                        text_id: Optional[str] = None, 
                        skip_qa: bool = False) -> Dict:
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
        analysis = response.messages[-1]["content"]
        
        # Step 2: Translator bot translates components
        response = swarm_client.run(
            agent=translator_bot,
            messages=[{
                "role": "user",
                "content": f"Translate these components from {scenario.source_label} to {scenario.target_label}: {analysis}"
            }]
        )
        translations = response.messages[-1]["content"]
        
        # Step 3: QA bot provides final translation (optional)
        if not skip_qa:
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
        else:
            # Use translator's output directly if skipping QA
            try:
                trans_data = json.loads(translations)
                if isinstance(trans_data, dict):
                    phrases = trans_data.get("phrases", [])
                    terms = trans_data.get("terms", [])
                    final_translation = " ".join(phrases + terms) if phrases or terms else translations
                else:
                    print(f"Warning: Unexpected translation format (not a dict): {translations}")
                    final_translation = translations
            except json.JSONDecodeError:
                print(f"Warning: Could not parse translation as JSON: {translations}")
                final_translation = translations
            except Exception as e:
                print(f"Warning: Error processing translation: {str(e)}")
                print(f"Translation content: {translations}")
                final_translation = translations
        
        translation_time = time.time() - start_time
        
        result = {
            "source_lang": scenario.source_code,
            "source_label": scenario.source_label,
            "target_lang": scenario.target_code,
            "target_label": scenario.target_label,
            "original": text,
            "translation": final_translation,
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

def process_input_file(scenario: TranslationScenario):
    """Process input file according to scenario configuration."""
    # Set up agents
    swarm_client, linguist_bot, translator_bot, qa_bot = setup_agents(scenario)
    
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
    current_batch = []
    batch_size = 5  # Start with groups of 5
    
    for i, item in enumerate(lines):
        try:
            content = item[scenario.config["input"]["content_field"]]
            text_id = item.get(scenario.config["input"]["id_field"])
        except KeyError as e:
            print(f"Error accessing fields in item {i}:")
            print(f"Item content: {item}")
            print(f"Expected fields: {scenario.config['input']['content_field']} and {scenario.config['input']['id_field']}")
            raise
        
        # Translate individual sentence
        result = translate_with_agents(
            content, scenario, swarm_client, 
            linguist_bot, translator_bot, qa_bot, 
            text_id, skip_qa=True  # Skip individual QA
        )
        current_batch.append(result)
        
        # Check if we should process the batch
        should_process = (
            len(current_batch) >= batch_size and 
            is_complete_thought(result["translation"])
        )
        
        if should_process or i == len(lines) - 1:  # Process if batch full or last item
            # Synthesize the batch
            combined_translation = batch_translations(
                current_batch, 
                scenario, 
                qa_bot,
                swarm_client  # Pass the swarm_client
            )
            
            # Update the last translation in the batch with the combined version
            for j, translation in enumerate(current_batch):
                if j == len(current_batch) - 1:
                    translation["translation"] = combined_translation
                
                # Write to output file
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(translation, ensure_ascii=False) + '\n')
            
            # Clear the batch
            current_batch = []
            
            # Update and save progress
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