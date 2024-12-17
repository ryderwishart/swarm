import os
import glob
import json
import shutil
from datetime import datetime
from typing import Dict, List

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

    # Write consolidated data to new files and copy to frontend
    for project_name, data in project_data.items():
        filename = f'{project_name}_consolidated.jsonl'
        consolidated_file_path = os.path.join(target_dir, filename)
        frontend_file_path = os.path.join(frontend_dir, filename)

        # Write to consolidated directory
        with open(consolidated_file_path, 'w', encoding='utf-8') as file:
            file.writelines(data)
        print(f'Consolidated file created: {consolidated_file_path}')

        # Copy to frontend public directory
        shutil.copy2(consolidated_file_path, frontend_file_path)
        print(f'Copied to frontend: {frontend_file_path}')

        # Add to scenarios list
        first_line = json.loads(data[0])
        scenarios.append({
            "id": project_name,
            "filename": filename,
            "source_lang": first_line["source_lang"],
            "source_label": first_line["source_label"],
            "target_lang": first_line["target_lang"],
            "target_label": first_line["target_label"]
        })

    # Write manifest.json
    manifest_path = os.path.join(frontend_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({"scenarios": scenarios}, f, indent=2)
    print(f'Created manifest: {manifest_path}')

if __name__ == "__main__":
    source_directory = 'scenarios/translations'
    target_directory = 'scenarios/consolidated'
    consolidate_files(source_directory, target_directory)