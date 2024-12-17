import os
import glob
import json
import shutil
from datetime import datetime
from typing import Dict, List
import re

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

    # For each project, sort the lines by biblical reference
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

        # Copy to frontend public directory
        shutil.copy2(consolidated_file_path, frontend_file_path)
        print(f'Copied to frontend: {frontend_file_path}')

        # Add to scenarios list
        first_line = json.loads(sorted_data[0])
        scenarios.append({
            "id": project_name,
            "filename": filename,
            "source_lang": first_line["source_lang"],
            "source_label": first_line["source_label"],
            "target_lang": first_line["target_lang"],
            "target_label": first_line["target_label"]
        })

    # Sort scenarios by biblical reference
    scenarios.sort(key=lambda x: parse_reference(x['id']))

    # Write manifest.json
    manifest_path = os.path.join(frontend_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({"scenarios": scenarios}, f, indent=2)
    print(f'Created manifest: {manifest_path}')

if __name__ == "__main__":
    source_directory = 'scenarios/translations'
    target_directory = 'scenarios/consolidated'
    consolidate_files(source_directory, target_directory)