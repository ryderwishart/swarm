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
    # Delete existing consolidated files
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    # Delete existing JSONL files from frontend public dir
    if os.path.exists(frontend_dir):
        for f in glob.glob(os.path.join(frontend_dir, "*.jsonl")):
            os.remove(f)
    os.makedirs(frontend_dir, exist_ok=True)

    # Get all JSONL files in the source directory
    jsonl_files = glob.glob(os.path.join(source_dir, '*.jsonl'))

    # Dictionary to hold data for each project
    project_data: Dict[str, List[str]] = {}
    scenarios: List[Dict[str, str]] = []

    # Read each file and group data by project
    for file_path in jsonl_files:
        project_name = os.path.basename(file_path).split('_')[0]
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if project_name not in project_data:
                project_data[project_name] = []
            project_data[project_name].extend(lines)

    # For each project, sort the lines and create scenario entry
    for project_name, data in project_data.items():
        # Parse lines into objects, sort them, then convert back to strings
        parsed_lines = [json.loads(line) for line in data]
        parsed_lines.sort(key=lambda x: parse_reference(x['id']))
        sorted_data = [json.dumps(line, ensure_ascii=False) + '\n' for line in parsed_lines]
        
        filename = f'{project_name}_consolidated.jsonl'
        consolidated_file_path = os.path.join(target_dir, filename)
        frontend_file_path = os.path.join(frontend_dir, filename)

        # Write sorted data to consolidated directory
        with open(consolidated_file_path, 'w', encoding='utf-8') as file:
            file.writelines(sorted_data)
        print(f'Consolidated file created: {consolidated_file_path}')

        # Get scenario config first
        scenario_config = get_scenario_config(project_name)
        if not scenario_config:
            print(f"Warning: No scenario config found for {project_name}")
            continue

        # Create scenario entry from config
        scenario_entry = {
            "id": project_name,
            "filename": filename,
            "source_lang": scenario_config.get("source", {}).get("code"),
            "target_lang": scenario_config.get("target", {}).get("code"),
            "source_label": scenario_config.get("source", {}).get("label"),
            "target_label": scenario_config.get("target", {}).get("label")
        }

        # Only use translation data as fallback if scenario config is missing info
        if parsed_lines:
            first_line = parsed_lines[0]
            if not scenario_entry["source_lang"]:
                scenario_entry["source_lang"] = first_line.get("source_lang")
            if not scenario_entry["target_lang"]:
                scenario_entry["target_lang"] = first_line.get("target_lang")
            if not scenario_entry["source_label"]:
                scenario_entry["source_label"] = first_line.get("source_label", scenario_entry["source_lang"])
            if not scenario_entry["target_label"]:
                scenario_entry["target_label"] = first_line.get("target_label", scenario_entry["target_lang"])

        scenarios.append(scenario_entry)

    # Sort scenarios by ID
    scenarios.sort(key=lambda x: x['id'])

    # Write manifest.json
    manifest_path = os.path.join(frontend_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({"scenarios": scenarios}, f, indent=2, ensure_ascii=False)
    print(f'Created manifest: {manifest_path}')

if __name__ == "__main__":
    source_directory = 'scenarios/translations'
    target_directory = 'scenarios/consolidated'
    consolidate_files(source_directory, target_directory)