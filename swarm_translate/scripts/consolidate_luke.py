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
        
        # Parse lines into objects, sort them, then convert back to strings
        parsed_lines = [json.loads(line) for line in data]
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
            "id": f"grc-{lang_code}",
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

    # Write manifest.json
    manifest_path = os.path.join(frontend_dir, "manifest.json")
    manifest_content = {
        "scenarios": scenarios,
        "updated_at": datetime.now().isoformat()
    }
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_content, f, indent=2, ensure_ascii=False)
    
    print(f'\nCreated manifest: {manifest_path}')
    print(f'Number of scenarios: {len(scenarios)}')
    for scenario in scenarios:
        print(f"  - {scenario['id']}: {scenario['source_label']} → {scenario['target_label']}")

if __name__ == "__main__":
    source_directory = 'translation_results'
    target_directory = 'scenarios/consolidated/luke-only'
    consolidate_luke_files(source_directory, target_directory) 