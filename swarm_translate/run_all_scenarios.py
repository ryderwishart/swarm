import csv
from pathlib import Path
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional
import sys
from dotenv import load_dotenv
import argparse
import pandas as pd

# Add parent directory to path to import from swarm_translate
sys.path.append(str(Path(__file__).parent.parent))

from swarm_translate.translate_dspy import translate_with_dspy
from swarm_translate.translation_scenario import TranslationScenario

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_run.log'),
        logging.StreamHandler()
    ]
)

def load_plan(plan_path: Path) -> List[Dict]:
    """Load scenarios from PLAN.csv."""
    scenarios = []
    with open(plan_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('SCENARIO_FILEPATH'):  # Only include rows with scenario files
                scenarios.append({
                    'language_name': row['LANGUAGE_NAME'],
                    'code': row['CODE'],
                    'country': row['COUNTRY'],
                    'region': row['REGION'],
                    'scenario_path': row['SCENARIO_FILEPATH']
                })
    return scenarios

def load_test_texts() -> List[Dict]:
    """Load test texts to translate."""
    # This function is now used as a fallback if the input file can't be loaded
    return [
        {"id": "Luke 1:1", "text": "Forasmuch as many have taken in hand to set forth in order a declaration of those things which are most surely believed among us,"},
        {"id": "Luke 1:2", "text": "Even as they delivered them unto us, which from the beginning were eyewitnesses, and ministers of the word;"},
        {"id": "Luke 1:3", "text": "It seemed good to me also, having had perfect understanding of all things from the very first, to write unto thee in order, most excellent Theophilus,"}
    ]

def load_texts_from_scenario(scenario: TranslationScenario, limit: Optional[int] = None) -> List[Dict]:
    """Load texts from the input file specified in the scenario."""
    try:
        input_config = scenario.config.get('input', {})
        input_file = input_config.get('file')
        if not input_file:
            logging.warning(f"No input file specified in scenario {scenario.name}")
            return load_test_texts()
        
        input_format = input_config.get('format', 'csv')
        id_field = input_config.get('id_field', 'id')
        content_field = input_config.get('content_field', 'content')
        
        # Resolve path - if it's absolute, use it; otherwise, resolve relative to base_path
        file_path = Path(input_file)
        if not file_path.is_absolute():
            file_path = scenario.base_path / file_path
        
        if not file_path.exists():
            logging.warning(f"Input file {file_path} not found for scenario {scenario.name}")
            return load_test_texts()
        
        texts = []
        if input_format.lower() == 'csv':
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                if limit and len(texts) >= limit:
                    break
                texts.append({
                    "id": str(row[id_field]) if id_field in row else "",
                    "text": str(row[content_field]) if content_field in row else ""
                })
        elif input_format.lower() == 'jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if limit and len(texts) >= limit:
                        break
                    try:
                        data = json.loads(line)
                        texts.append({
                            "id": str(data.get(id_field, "")),
                            "text": str(data.get(content_field, ""))
                        })
                    except Exception as e:
                        logging.error(f"Error parsing JSONL line: {e}")
        else:
            logging.warning(f"Unsupported input format {input_format} for scenario {scenario.name}")
            return load_test_texts()
        
        if not texts:
            logging.warning(f"No texts loaded from {file_path} for scenario {scenario.name}")
            return load_test_texts()
        
        logging.info(f"Loaded {len(texts)} texts from {file_path} for scenario {scenario.name}")
        return texts
        
    except Exception as e:
        logging.error(f"Error loading texts from scenario {scenario.name}: {e}")
        return load_test_texts()

def get_completed_scenarios(output_dir: Path) -> set:
    """Get set of completed scenario codes."""
    completed = set()
    for lang_dir in output_dir.iterdir():
        if lang_dir.is_dir():
            # Check if there's a completed marker file
            if (lang_dir / 'completed.txt').exists():
                completed.add(lang_dir.name)
    return completed

def mark_scenario_completed(output_dir: Path, lang_code: str):
    """Mark a scenario as completed."""
    lang_dir = output_dir / lang_code
    with open(lang_dir / 'completed.txt', 'w') as f:
        f.write(f"Completed at {datetime.now().isoformat()}\n")

def process_scenario(scenario_info: Dict, test_mode: bool, output_dir: Path, limit: Optional[int] = None) -> bool:
    """Process a single scenario."""
    try:
        # Create scenario instance
        scenario_path = scenario_info['scenario_path']
        if scenario_path.startswith('swarm_translate/'):
            # Use the path relative to the parent directory
            scenario_path = Path(__file__).parent.parent / scenario_path
        else:
            # Just use the path directly
            scenario_path = Path(scenario_path)
            
        scenario = TranslationScenario.from_file(scenario_path)
        
        # Create output directory for this language
        lang_output_dir = output_dir / scenario_info['code']
        lang_output_dir.mkdir(exist_ok=True)
        
        # Create results file
        results_file = lang_output_dir / f"translations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        logging.info(f"Processing {scenario_info['language_name']} ({scenario_info['code']})")
        
        # Load texts from scenario file or use test texts
        texts = load_test_texts() if test_mode else load_texts_from_scenario(scenario, limit)
        
        # Process each test text
        successful = 0
        for i, text_info in enumerate(texts):
            try:
                # Calculate max_tokens based on input length - approximately 2x
                input_token_estimate = len(text_info['text'].split()) * 1.5  # rough estimate
                max_tokens = max(500, int(input_token_estimate * 2))  # at least 500 tokens, up to 2x input
                
                result = translate_with_dspy(
                    text=text_info['text'],
                    scenario=scenario,
                    text_id=text_info['id'],
                    max_tokens=max_tokens
                )
                
                # Save result
                with open(results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                logging.info(f"Translated {text_info['id']} to {scenario_info['code']}")
                successful += 1
                
                # Small delay to avoid rate limits (reduced to 0.2 seconds)
                time.sleep(0.2)
                
            except Exception as e:
                logging.error(f"Error translating {text_info['id']} to {scenario_info['code']}: {str(e)}")
                continue
            
            # Log progress every 10 items
            if (i + 1) % 10 == 0:
                logging.info(f"Progress: {i+1}/{len(texts)} texts processed for {scenario_info['code']}")
        
        # Mark scenario as completed
        mark_scenario_completed(output_dir, scenario_info['code'])
        logging.info(f"Completed {scenario_info['language_name']} with {successful}/{len(texts)} successful translations")
        return True
        
    except Exception as e:
        logging.error(f"Error processing scenario {scenario_info['language_name']}: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run translation scenarios')
    parser.add_argument('--resume', action='store_true', help='Resume from last completed scenario')
    parser.add_argument('--test', action='store_true', help='Use test texts instead of full input files')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of texts to process per scenario')
    parser.add_argument('--scenario', type=str, default=None, help='Process only a specific scenario code')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Set up paths
    base_path = Path(__file__).parent  # Use swarm_translate directory
    plan_path = base_path.parent / 'PLAN.csv'  # PLAN.csv is in the parent directory
    output_dir = base_path / 'translation_results'
    output_dir.mkdir(exist_ok=True)
    
    # Load scenarios
    scenarios = load_plan(plan_path)
    
    # Filter for specific scenario if requested
    if args.scenario:
        scenarios = [s for s in scenarios if s['code'].lower() == args.scenario.lower()]
        if not scenarios:
            logging.error(f"No scenario found with code {args.scenario}")
            return
    
    # Get completed scenarios if resuming
    completed_scenarios = get_completed_scenarios(output_dir) if args.resume else set()
    
    # Filter scenarios if resuming
    if args.resume:
        scenarios = [s for s in scenarios if s['code'] not in completed_scenarios]
        logging.info(f"Resuming with {len(scenarios)} remaining scenarios")
    
    # Process each scenario
    total = len(scenarios)
    successful = 0
    
    logging.info(f"Starting translation run with {total} scenarios")
    
    for i, scenario_info in enumerate(scenarios, 1):
        logging.info(f"Processing scenario {i}/{total}: {scenario_info['language_name']}")
        
        if process_scenario(scenario_info, args.test, output_dir, args.limit):
            successful += 1
        
        # Progress update
        logging.info(f"Progress: {i}/{total} scenarios processed ({successful} successful)")
    
    # Final report
    logging.info(f"Translation run complete. {successful}/{total} scenarios processed successfully.")

if __name__ == "__main__":
    main() 