import os
import glob
import json
from datetime import datetime

def consolidate_files(source_dir: str, target_dir: str):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Get all JSONL files in the source directory
    jsonl_files = glob.glob(os.path.join(source_dir, '*.jsonl'))

    # Dictionary to hold data for each project
    project_data = {}

    # Read each file and group data by project
    for file_path in jsonl_files:
        project_name = os.path.basename(file_path).split('_')[0]
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if project_name not in project_data:
                project_data[project_name] = []
            project_data[project_name].extend(lines)

    # Write consolidated data to new files
    for project_name, data in project_data.items():
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        consolidated_file_path = os.path.join(target_dir, f'{project_name}_consolidated_{timestamp}.jsonl')
        with open(consolidated_file_path, 'w', encoding='utf-8') as file:
            file.writelines(data)
        print(f'Consolidated file created: {consolidated_file_path}')

if __name__ == "__main__":
    source_directory = 'scenarios/translations'
    target_directory = 'scenarios/consolidated'
    consolidate_files(source_directory, target_directory)