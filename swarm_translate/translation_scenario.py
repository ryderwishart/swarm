import json
from datetime import datetime
from pathlib import Path
from typing import Dict

class TranslationScenario:
    def __init__(self, scenario_path: str):
        self.scenario_path = Path(scenario_path)
        with open(scenario_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Set up scenario name
        self.name = self.config.get("name", Path(scenario_path).stem)
        
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
        
        # Book ID mapping (optional)
        self.id_mapping_path = self.base_path / "data" / "id_mapping.json"
        self.book_id_mapping = self._load_book_id_mapping()

    @classmethod
    def from_file(cls, scenario_path: Path) -> 'TranslationScenario':
        return cls(str(scenario_path))

    def _load_book_id_mapping(self) -> Dict[str, str]:
        try:
            if self.id_mapping_path.exists():
                with open(self.id_mapping_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load book ID mapping: {e}")
        return {}

    def get_style_prompt(self) -> str:
        style = self.config["style"]
        return f"""Style requirements:\n        - Formality: {style['formality']}\n        - Register: {style['register']}\n        - Notes: {style.get('notes', '')}"""

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