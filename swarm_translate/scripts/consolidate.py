import os
import glob
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import re
from pathlib import Path

def parse_reference(id: str) -> tuple:
    """Parse a biblical reference ID into sortable components."""
    # Known books in canonical order (Old Testament followed by New Testament)
    BOOK_ORDER = {
        'genesis': 1, 'exodus': 2, 'leviticus': 3, 'numbers': 4, 'deuteronomy': 5,
        'joshua': 6, 'judges': 7, 'ruth': 8, '1 samuel': 9, '2 samuel': 10,
        '1 kings': 11, '2 kings': 12, '1 chronicles': 13, '2 chronicles': 14,
        'ezra': 15, 'nehemiah': 16, 'esther': 17, 'job': 18, 'psalm': 19,
        'proverbs': 20, 'ecclesiastes': 21, 'song of solomon': 22, 'isaiah': 23,
        'jeremiah': 24, 'lamentations': 25, 'ezekiel': 26, 'daniel': 27,
        'hosea': 28, 'joel': 29, 'amos': 30, 'obadiah': 31, 'jonah': 32,
        'micah': 33, 'nahum': 34, 'habakkuk': 35, 'zephaniah': 36,
        'haggai': 37, 'zechariah': 38, 'malachi': 39,
        # New Testament
        'matthew': 40, 'mark': 41, 'luke': 42, 'john': 43, 'acts': 44,
        'romans': 45, '1 corinthians': 46, '2 corinthians': 47, 'galatians': 48,
        'ephesians': 49, 'philippians': 50, 'colossians': 51,
        '1 thessalonians': 52, '2 thessalonians': 53, '1 timothy': 54,
        '2 timothy': 55, 'titus': 56, 'philemon': 57, 'hebrews': 58,
        'james': 59, '1 peter': 60, '2 peter': 61, '1 john': 62, '2 john': 63,
        '3 john': 64, 'jude': 65, 'revelation': 66
    }
    
    # Extract book, chapter, and verse using regex
    # Format: "Book Name 1:1" or "Book of Name 1:1"
    match = re.match(r'((?:\d\s)?[A-Za-z]+(?:\s+(?:of\s+)?[A-Za-z]+)*)\s+(\d+):(\d+)', id)
    if not match:
        return (float('inf'), 0, 0)  # Put invalid formats at the end
        
    book, chapter, verse = match.groups()
    book = book.lower()
    
    # Get book order (default to infinity if book not in list)
    book_num = BOOK_ORDER.get(book, float('inf'))
    
    return (book_num, int(chapter), int(verse))

def get_scenario_config(project_name: str, base_dir: str = "scenarios") -> Optional[Dict]:
    """Find and load the scenario config file for a given project."""
    scenario_dir = Path(base_dir)
    
    # Look for scenario files with matching project name
    for scenario_file in scenario_dir.glob("*.json"):
        try:
            with open(scenario_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Check if this scenario matches our project
                if project_name == config.get("id"):
                    return config
        except Exception as e:
            print(f"Warning: Error reading scenario file {scenario_file}: {e}")
            continue
    return None

def consolidate_files(source_dir: str, target_dir: str, frontend_dir: str = "../swarm_frontend/public"):
    """Consolidate translation files and create manifest."""
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

    # Get all JSONL files from source directory
    jsonl_files = [f for f in glob.glob(os.path.join(source_dir, "*.jsonl"))]
    if not jsonl_files:
        print("No JSONL files found in source directory")
        return

    print(f"Found {len(jsonl_files)} JSONL files")

    # Dictionary to hold data for each project
    project_data: Dict[str, List[str]] = {}
    scenarios: List[Dict[str, str]] = []

    # Read each file and group data by project
    for file_path in jsonl_files:
        project_name = os.path.basename(file_path).split('_')[0]
        print(f"Processing project: {project_name}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if project_name not in project_data:
                project_data[project_name] = []
            project_data[project_name].extend(lines)

    # For each project, sort the lines and create scenario entry
    for project_name, data in project_data.items():
        # Get scenario config first
        scenario_config = get_scenario_config(project_name)
        if not scenario_config:
            print(f"Warning: No scenario config found for {project_name}")
            continue

        # Parse lines into objects, sort them, then convert back to strings
        parsed_lines = [json.loads(line) for line in data]
        parsed_lines.sort(key=lambda x: parse_reference(x['id']))
        sorted_data = [json.dumps(line, ensure_ascii=False) + '\n' for line in parsed_lines]
        
        filename = f'{project_name}_consolidated.jsonl'
        consolidated_file_path = os.path.join(target_dir, filename)
        frontend_file_path = os.path.join(frontend_dir, filename)

        # Write sorted data to both consolidated directory and frontend public directory
        with open(consolidated_file_path, 'w', encoding='utf-8') as file:
            file.writelines(sorted_data)
        # Copy to frontend directory
        shutil.copy2(consolidated_file_path, frontend_file_path)
        print(f'Created consolidated file: {consolidated_file_path}')
        print(f'Copied to frontend: {frontend_file_path}')

        # Create scenario entry from config
        scenario_entry = {
            "id": project_name,
            "filename": filename,
            "source_lang": scenario_config["source"]["code"],
            "target_lang": scenario_config["target"]["code"],
            "source_label": scenario_config["source"]["label"],
            "target_label": scenario_config["target"]["label"]
        }
        scenarios.append(scenario_entry)

    # Sort scenarios
    scenarios.sort(key=lambda x: x['id'])

    # Write manifest.json to frontend directory
    manifest_path = os.path.join(frontend_dir, "manifest.json")
    manifest_content = {
        "scenarios": scenarios,
        "updated_at": datetime.now().isoformat()
    }
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_content, f, indent=2, ensure_ascii=False)
    print(f'Created manifest: {manifest_path}')
    print(f'Number of scenarios: {len(scenarios)}')
    for scenario in scenarios:
        print(f"  - {scenario['id']}: {scenario['source_label']} → {scenario['target_label']}")

if __name__ == "__main__":
    source_directory = 'scenarios/translations'
    target_directory = 'scenarios/consolidated'
    consolidate_files(source_directory, target_directory)