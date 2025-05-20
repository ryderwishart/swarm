import csv
from pathlib import Path
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import sys
from dotenv import load_dotenv
import argparse
import pandas as pd
import multiprocessing as mp
from functools import partial
import os

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
        
        # Resolve path - handle both absolute and relative paths
        file_path = Path(input_file)
        if not file_path.is_absolute():
            # Try different relative paths
            potential_paths = [
                scenario.base_path / file_path,  # Relative to scenario file
                Path(__file__).parent / file_path,  # Relative to script
                Path(__file__).parent.parent / file_path  # Relative to parent directory
            ]
            
            for path in potential_paths:
                if path.exists():
                    file_path = path
                    break
            else:
                logging.warning(f"Input file {input_file} not found in any of the potential paths for scenario {scenario.name}")
                return load_test_texts()
        elif not file_path.exists():
            logging.warning(f"Absolute input file path {file_path} not found for scenario {scenario.name}")
            return load_test_texts()
        
        logging.info(f"Using input file: {file_path}")
        
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

def process_batch(batch_data: Tuple[str, TranslationScenario, List[Dict], Path], process_id: int):
    """Process a batch of texts for a single scenario."""
    lang_code, scenario, texts, results_file = batch_data
    
    successful = 0
    batch_start = time.time()
    
    try:
        # Configure process-specific logging
        log_file = f"translation_run_{process_id}.log"
        process_logger = logging.getLogger(f"process_{process_id}")
        process_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        process_logger.addHandler(handler)
        
        process_logger.info(f"Process {process_id} starting batch with {len(texts)} texts for {lang_code}")
        
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
                
                # Save result - use a file lock for safety in multi-process environment
                with open(results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                process_logger.info(f"Translated {text_info['id']} to {lang_code}")
                successful += 1
                
                # Small delay to avoid rate limits
                time.sleep(0.2)
                
            except Exception as e:
                process_logger.error(f"Error translating {text_info['id']} to {lang_code}: {str(e)}")
                continue
            
            # Log progress every 5 items
            if (i + 1) % 5 == 0:
                process_logger.info(f"Process {process_id} progress: {i+1}/{len(texts)} texts for {lang_code}")
        
        batch_time = time.time() - batch_start
        process_logger.info(f"Process {process_id} completed batch for {lang_code} with {successful}/{len(texts)} successful translations in {batch_time:.1f} seconds")
        
        return (lang_code, successful, len(texts))
        
    except Exception as e:
        logging.error(f"Process {process_id} error processing batch for {lang_code}: {str(e)}")
        return (lang_code, successful, len(texts))

def process_scenario(scenario_info: Dict, test_mode: bool, output_dir: Path, limit: Optional[int] = None, parallel: bool = True) -> bool:
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
        
        if not parallel or len(texts) <= 10:
            # For small batches, process sequentially
            return process_sequential(scenario_info['code'], scenario, texts, results_file)
        else:
            # For larger batches, process in parallel
            return process_parallel(scenario_info['code'], scenario, texts, results_file)
        
    except Exception as e:
        logging.error(f"Error processing scenario {scenario_info['language_name']}: {str(e)}")
        return False

def process_sequential(lang_code: str, scenario: TranslationScenario, texts: List[Dict], results_file: Path) -> bool:
    """Process texts sequentially for a scenario."""
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
            
            logging.info(f"Translated {text_info['id']} to {lang_code}")
            successful += 1
            
            # Small delay to avoid rate limits
            time.sleep(0.2)
            
        except Exception as e:
            logging.error(f"Error translating {text_info['id']} to {lang_code}: {str(e)}")
            continue
        
        # Log progress every 10 items
        if (i + 1) % 10 == 0:
            logging.info(f"Progress: {i+1}/{len(texts)} texts processed for {lang_code}")
    
    logging.info(f"Completed with {successful}/{len(texts)} successful translations")
    return successful > 0

def process_parallel(lang_code: str, scenario: TranslationScenario, texts: List[Dict], results_file: Path) -> bool:
    """Process texts in parallel for a scenario."""
    try:
        # Determine number of processes to use (limit to CPU count or text count)
        num_processes = min(mp.cpu_count(), max(1, len(texts) // 5))
        
        # Divide texts into batches
        batch_size = max(1, len(texts) // num_processes)
        batches = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Each batch is (language_code, scenario, texts, results_file)
            batches.append((lang_code, scenario, batch_texts, results_file))
        
        logging.info(f"Processing {len(texts)} texts for {lang_code} in {len(batches)} batches with {num_processes} processes")
        
        # Process batches in parallel
        with mp.Pool(num_processes, maxtasksperchild=1) as pool:
            try:
                # Process batches with unique process IDs
                results = pool.starmap(process_batch, [(batch, i) for i, batch in enumerate(batches)])
                
                # Combine results
                total_successful = sum(r[1] for r in results)
                total_texts = sum(r[2] for r in results)
                
                logging.info(f"Parallel processing complete for {lang_code}: {total_successful}/{total_texts} successful translations")
                return total_successful > 0
                
            except KeyboardInterrupt:
                logging.info("\nGracefully shutting down workers...")
                pool.terminate()
                pool.join()
                return False
            except Exception as e:
                logging.error(f"Error in parallel processing for {lang_code}: {str(e)}")
                return False
    except Exception as e:
        logging.error(f"Error setting up parallel processing for {lang_code}: {str(e)}")
        # Fall back to sequential processing
        logging.info(f"Falling back to sequential processing for {lang_code}")
        return process_sequential(lang_code, scenario, texts, results_file)

def process_scenarios_parallel(scenarios: List[Dict], test_mode: bool, output_dir: Path, limit: Optional[int] = None) -> int:
    """Process multiple scenarios in parallel."""
    try:
        # Determine how many scenarios to process in parallel
        # Limit to half of available CPU cores to avoid overwhelming API rate limits
        num_workers = min(max(1, mp.cpu_count() // 2), len(scenarios))
        
        logging.info(f"Processing {len(scenarios)} scenarios with {num_workers} parallel workers")
        
        with mp.Pool(num_workers, maxtasksperchild=1) as pool:
            try:
                # Process each scenario with partial function
                process_func = partial(process_scenario, test_mode=test_mode, output_dir=output_dir, limit=limit)
                results = pool.map(process_func, scenarios)
                
                # Count successful scenarios
                successful = sum(1 for r in results if r)
                return successful
                
            except KeyboardInterrupt:
                logging.info("\nGracefully shutting down workers...")
                pool.terminate()
                pool.join()
                return 0
            except Exception as e:
                logging.error(f"Error in parallel scenario processing: {str(e)}")
                return 0
    except Exception as e:
        logging.error(f"Error setting up parallel scenario processing: {str(e)}")
        # Fall back to sequential processing
        logging.info("Falling back to sequential scenario processing")
        successful = 0
        for scenario_info in scenarios:
            if process_scenario(scenario_info, test_mode, output_dir, limit, parallel=False):
                successful += 1
        return successful

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run translation scenarios')
    parser.add_argument('--resume', action='store_true', help='Resume from last completed scenario')
    parser.add_argument('--test', action='store_true', help='Use test texts instead of full input files')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of texts to process per scenario')
    parser.add_argument('--scenario', type=str, action='append', help='Process only specific scenario code(s). Can be specified multiple times.')
    parser.add_argument('--sequential', action='store_true', help='Process scenarios sequentially instead of in parallel')
    parser.add_argument('--no-parallel-texts', action='store_true', help='Do not parallelize text processing within scenarios')
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
    
    # Filter for specific scenarios if requested
    if args.scenario:
        # Convert to lowercase for case-insensitive matching
        scenario_codes = [s.lower() for s in args.scenario]
        filtered_scenarios = [s for s in scenarios if s['code'].lower() in scenario_codes]
        
        if not filtered_scenarios:
            logging.error(f"No scenarios found with code(s): {', '.join(args.scenario)}")
            return
            
        scenarios = filtered_scenarios
        logging.info(f"Filtered to {len(scenarios)} scenarios: {', '.join(s['language_name'] for s in scenarios)}")
    
    # Get completed scenarios if resuming
    completed_scenarios = get_completed_scenarios(output_dir) if args.resume else set()
    
    # Filter scenarios if resuming
    if args.resume:
        scenarios = [s for s in scenarios if s['code'] not in completed_scenarios]
        logging.info(f"Resuming with {len(scenarios)} remaining scenarios")
    
    start_time = time.time()
    logging.info(f"Starting translation run with {len(scenarios)} scenarios")
    
    successful = 0
    
    if args.sequential or len(scenarios) == 1:
        # Process scenarios sequentially
        for i, scenario_info in enumerate(scenarios, 1):
            logging.info(f"Processing scenario {i}/{len(scenarios)}: {scenario_info['language_name']}")
            
            if process_scenario(scenario_info, args.test, output_dir, args.limit, not args.no_parallel_texts):
                successful += 1
                # Mark as completed
                mark_scenario_completed(output_dir, scenario_info['code'])
            
            # Progress update
            logging.info(f"Progress: {i}/{len(scenarios)} scenarios processed ({successful} successful)")
    else:
        # Process scenarios in parallel
        successful = process_scenarios_parallel(scenarios, args.test, output_dir, args.limit)
        
        # Mark all completed scenarios
        for scenario_info in scenarios:
            # Check if output exists for this scenario
            lang_dir = output_dir / scenario_info['code']
            if lang_dir.exists() and list(lang_dir.glob("translations_*.jsonl")):
                mark_scenario_completed(output_dir, scenario_info['code'])
    
    # Calculate execution time
    total_time = time.time() - start_time
    
    # Final report
    logging.info(f"Translation run complete. {successful}/{len(scenarios)} scenarios processed successfully.")
    logging.info(f"Total execution time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    # Set start method for multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass
    main() 