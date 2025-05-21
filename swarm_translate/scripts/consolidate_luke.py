import os
import glob
import json
from datetime import datetime
from typing import Dict, List
import re
from pathlib import Path

def parse_luke_reference(id: str) -> tuple:
    """Parse a Luke reference ID into sortable components."""
    # Extract chapter and verse using regex
    match = re.match(r'Luke\s+(\d+):(\d+)', id)
    if not match:
        return (float('inf'), float('inf'))  # Put invalid formats at the end
        
    chapter, verse = match.groups()
    return (int(chapter), int(verse))

def consolidate_luke_files(source_dir: str, target_dir: str, frontend_dir: str = "../swarm_frontend/public"):
    """Consolidate Luke-only translation files and create manifest."""
    # Convert paths to absolute paths
    source_dir = os.path.abspath(source_dir)
    target_dir = os.path.abspath(target_dir)
    frontend_dir = os.path.abspath(frontend_dir)
    
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Frontend directory: {frontend_dir}")

    # Ensure directories exist
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(frontend_dir, exist_ok=True)

    # Dictionary to hold data for each language
    language_data: Dict[str, List[str]] = {}
    scenarios: List[Dict[str, str]] = []

    # Get all language code directories
    lang_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found {len(lang_dirs)} language directories")

    # Process each language directory
    for lang_code in lang_dirs:
        lang_dir = os.path.join(source_dir, lang_code)
        print(f"\nProcessing language directory: {lang_code}")
        
        # Get all JSONL files in this language directory
        jsonl_files = glob.glob(os.path.join(lang_dir, "*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL files found in {lang_dir}")
            continue

        print(f"Found {len(jsonl_files)} JSONL files for {lang_code}")
        
        # Read and combine all files for this language
        for file_path in jsonl_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if lang_code not in language_data:
                    language_data[lang_code] = []
                language_data[lang_code].extend(lines)

    # For each language, sort the lines and create scenario entry
    for lang_code, data in language_data.items():
        print(f"\nProcessing translations for {lang_code}")
        
        # Parse lines into objects, filter out failed translations, then sort
        parsed_lines = []
        for line in data:
            try:
                entry = json.loads(line)
                # Skip failed translations
                if not entry['translation'].startswith("[Translation failed:"):
                    parsed_lines.append(entry)
                else:
                    print(f"Skipping failed translation for {entry['id']}: {entry['translation'][:50]}...")
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line[:50]}...")
                continue
        
        # Check if we have any valid translations after filtering
        if not parsed_lines:
            print(f"No valid translations found for {lang_code} after filtering")
            continue
            
        # Sort by chapter and verse
        parsed_lines.sort(key=lambda x: parse_luke_reference(x['id']))
        sorted_data = [json.dumps(line, ensure_ascii=False) + '\n' for line in parsed_lines]
        
        # Create filenames for both consolidated and frontend
        filename = f'grc-{lang_code}_luke.jsonl'
        consolidated_file_path = os.path.join(target_dir, filename)
        frontend_file_path = os.path.join(frontend_dir, filename)

        # Write sorted data to consolidated directory
        with open(consolidated_file_path, 'w', encoding='utf-8') as file:
            file.writelines(sorted_data)
        print(f'Created consolidated file: {consolidated_file_path}')

        # Copy to frontend directory
        with open(frontend_file_path, 'w', encoding='utf-8') as file:
            file.writelines(sorted_data)
        print(f'Copied to frontend: {frontend_file_path}')

        # Create scenario entry
        scenario_entry = {
            "id": f"grc-{lang_code}_luke",  # Added _luke suffix to avoid conflicts
            "filename": filename,
            "source_lang": "grc",
            "target_lang": lang_code,
            "source_label": "Greek (Luke)",
            "target_label": f"{lang_code.upper()} (Luke)"
        }

        scenarios.append(scenario_entry)
        print(f"Added scenario: {scenario_entry['source_label']} → {scenario_entry['target_label']}")

    # Sort scenarios by ID
    scenarios.sort(key=lambda x: x['id'])

    # Read existing manifest if it exists
    manifest_path = os.path.join(frontend_dir, "manifest.json")
    existing_scenarios = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                existing_manifest = json.load(f)
                existing_scenarios = existing_manifest.get('scenarios', [])
                print(f"Found {len(existing_scenarios)} existing scenarios in manifest")
        except Exception as e:
            print(f"Error reading existing manifest: {e}")
            existing_scenarios = []

    # Combine existing and new scenarios
    all_scenarios = existing_scenarios + scenarios
    
    # Remove any duplicates (in case a language appears in both manifests)
    seen_ids = set()
    unique_scenarios = []
    for scenario in all_scenarios:
        if scenario['id'] not in seen_ids:
            seen_ids.add(scenario['id'])
            unique_scenarios.append(scenario)

    # Sort all scenarios by ID
    unique_scenarios.sort(key=lambda x: x['id'])

    # Write updated manifest.json
    manifest_content = {
        "scenarios": unique_scenarios,
        "updated_at": datetime.now().isoformat()
    }
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_content, f, indent=2, ensure_ascii=False)
    
    print(f'\nUpdated manifest: {manifest_path}')
    print(f'Total number of scenarios: {len(unique_scenarios)}')
    print(f'  - Existing scenarios: {len(existing_scenarios)}')
    print(f'  - New Luke scenarios: {len(scenarios)}')
    for scenario in unique_scenarios:
        print(f"  - {scenario['id']}: {scenario['source_label']} → {scenario['target_label']}")

if __name__ == "__main__":
    source_directory = 'translation_results'
    target_directory = 'scenarios/consolidated/luke-only'
    consolidate_luke_files(source_directory, target_directory) 