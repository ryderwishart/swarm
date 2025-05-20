import dspy
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
from swarm_translate.translation_scenario import TranslationScenario

# Load environment variables
load_dotenv()

class TranslationAnalysis(dspy.Signature):
    """Analyze text for translation components."""
    text: str = dspy.InputField()
    source_label: str = dspy.InputField()
    target_label: str = dspy.InputField()
    book_context: str = dspy.InputField(default="")
    
    phrases: list = dspy.OutputField(desc="List of main phrases")
    terms: list = dspy.OutputField(desc="Key terms")
    idioms: list = dspy.OutputField(desc="Idiomatic expressions")
    cultural: list = dspy.OutputField(desc="Cultural references")

class TranslationComponents(dspy.Signature):
    """Translate analyzed components."""
    analysis: Any = dspy.InputField()
    source_label: str = dspy.InputField()
    target_label: str = dspy.InputField()
    style_prompt: str = dspy.InputField()
    memory_context: str = dspy.InputField(default="")
    
    phrases: list = dspy.OutputField(desc="Translated phrases")
    terms: list = dspy.OutputField(desc="Translated terms")
    idioms: list = dspy.OutputField(desc="Translated expressions")
    cultural: list = dspy.OutputField(desc="Translated references")

class TranslationCombination(dspy.Signature):
    """Combine translated components into final translation."""
    original: str = dspy.InputField()
    translations: Any = dspy.InputField()
    source_label: str = dspy.InputField()
    target_label: str = dspy.InputField()
    style_prompt: str = dspy.InputField()
    
    translation: str = dspy.OutputField(desc="Final translation")

class TranslationConsolidation(dspy.Signature):
    """Ensure consistency with previous verses."""
    final_translation: str = dspy.InputField()
    original: str = dspy.InputField()
    book_context: str = dspy.InputField()
    previous_verses: list = dspy.InputField()
    source_label: str = dspy.InputField()
    target_label: str = dspy.InputField()
    style_prompt: str = dspy.InputField()
    verse_num: str = dspy.InputField()
    
    refined_translation: str = dspy.OutputField(desc="Refined translation")

class TranslationMemory:
    def __init__(self, memory_path: Path):
        self.memory_path = memory_path
        self.memory = {}
        self._load_memory()
    
    def _load_memory(self):
        if self.memory_path.exists():
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        key = json.dumps(entry['analysis'].get('terms', []), ensure_ascii=False)
                        self.memory[key] = entry
                    except Exception as e:
                        print(f"Warning: Error loading memory entry: {e}")
    
    def find_matches(self, analysis: Dict) -> List[Dict]:
        """Find similar translations in memory."""
        matches = []
        for source_text, entry in self.memory.items():
            if any(term in source_text for term in analysis.get('terms', [])):
                matches.append(entry)
        return matches[:3]  # Return top 3 matches
    
    def add_entry(self, analysis: Dict, translation: Dict):
        """Add a new translation to memory."""
        try:
            # Convert prediction objects to dictionaries
            analysis_dict = self._prediction_to_dict(analysis)
            translation_dict = self._prediction_to_dict(translation)
            
            entry = {
                'analysis': analysis_dict,
                'translation': translation_dict
            }
            key = json.dumps(analysis_dict.get('terms', []), ensure_ascii=False)
            self.memory[key] = entry
            
            # Save to file
            with open(self.memory_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"Warning: Failed to add to translation memory: {e}")
    
    def _prediction_to_dict(self, obj) -> Dict:
        """Convert DSPy prediction objects to dictionaries."""
        if hasattr(obj, '__dict__'):
            # For Prediction objects, convert to dict
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    result[key] = self._prediction_to_dict(value)
            return result
        elif isinstance(obj, list):
            return [self._prediction_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._prediction_to_dict(v) for k, v in obj.items()}
        else:
            return obj

class TranslationProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(TranslationAnalysis)
        self.translator = dspy.ChainOfThought(TranslationComponents)
        self.combiner = dspy.ChainOfThought(TranslationCombination)
        self.consolidator = dspy.ChainOfThought(TranslationConsolidation)
    
    def forward(self, text: str, scenario: 'TranslationScenario', 
                translation_memory: TranslationMemory, text_id: Optional[str] = None) -> Dict:
        # Get book context
        book_context = ""
        if text_id:
            try:
                book_name = text_id.split()[0]
                if book_name in scenario.book_id_mapping:
                    book_context = f"\nThis verse is from {scenario.book_id_mapping[book_name]}. "
            except Exception as e:
                print(f"Warning: Error creating book context for '{text_id}': {e}")
        
        # Step 1: Analyze text
        analysis = self.analyzer(
            text=text,
            source_label=scenario.source_label,
            target_label=scenario.target_label,
            book_context=book_context
        )
        
        # Step 2: Get matching translations and translate components
        matches = translation_memory.find_matches(analysis)
        memory_context = self._create_memory_context(matches) if matches else ""
        
        translations = self.translator(
            analysis=analysis,
            source_label=scenario.source_label,
            target_label=scenario.target_label,
            style_prompt=scenario.get_style_prompt(),
            memory_context=memory_context
        )
        
        # Add to translation memory
        try:
            translation_memory.add_entry(analysis, translations)
        except Exception as e:
            print(f"Warning: Failed to add to translation memory: {e}")
        
        # Step 3: Combine translations
        final_translation = self.combiner(
            original=text,
            translations=translations,
            source_label=scenario.source_label,
            target_label=scenario.target_label,
            style_prompt=scenario.get_style_prompt()
        )
        
        # Step 4: Consolidate with previous verses
        try:
            previous_verses = self._get_previous_verses(text_id, scenario)
            if previous_verses:
                verse_num = text_id.split(':')[1] if text_id else "unknown"
                final_translation = self.consolidator(
                    final_translation=final_translation.translation,
                    original=text,
                    book_context=book_context,
                    previous_verses=previous_verses,
                    source_label=scenario.source_label,
                    target_label=scenario.target_label,
                    style_prompt=scenario.get_style_prompt(),
                    verse_num=verse_num
                )
                final_translation = final_translation.refined_translation
            else:
                final_translation = final_translation.translation
        except Exception as e:
            print(f"Warning: Consolidation step failed: {e}")
            final_translation = final_translation.translation
        
        # Create result
        result = {
            "source_lang": scenario.source_code,
            "target_lang": scenario.target_code,
            "original": text,
            "translation": final_translation,
            "calver": datetime.now().strftime("%Y.%m.%d")
        }
        
        if text_id:
            result["id"] = text_id
            
        return result
    
    def _create_memory_context(self, matches: List[Dict]) -> str:
        """Create context string from translation memory matches."""
        if not matches:
            return ""
        
        context_parts = []
        for match in matches:
            try:
                analysis = match['analysis']
                context_parts.append(
                    f"Similar translation:\n"
                    f"Terms: {', '.join(analysis.get('terms', []))}\n"
                    f"Translation: {json.dumps(match['translation'], ensure_ascii=False)}\n"
                )
            except Exception as e:
                print(f"Warning: Error creating memory context: {e}")
                continue
        
        if context_parts:
            return "Previous similar translations for reference:\n" + "\n".join(context_parts)
        return ""
    
    def _get_previous_verses(self, text_id: Optional[str], scenario: 'TranslationScenario', 
                           num_verses: int = 2) -> List[Dict]:
        """Get previous verses for context."""
        if not text_id:
            return []
        
        try:
            # Parse book and chapter from text_id (e.g., "Genesis 1:5")
            parts = text_id.split()
            if len(parts) != 2:
                return []
                
            book = parts[0]
            chapter_verse = parts[1].split(':')
            if len(chapter_verse) != 2:
                return []
                
            chapter, verse = chapter_verse
            verse = int(verse)
            
            # Read output file to find previous verses
            output_path = scenario.get_output_path()
            if not output_path.exists():
                return []
                
            previous_verses = []
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry_id = entry.get('id', '')
                        if entry_id.startswith(f"{book} {chapter}:"):
                            entry_verse = int(entry_id.split(':')[1])
                            if entry_verse < verse and entry_verse >= verse - num_verses:
                                previous_verses.append(entry)
                    except Exception:
                        continue
                        
            return sorted(previous_verses, 
                         key=lambda x: int(x['id'].split(':')[1]) if x.get('id') else 0)
                         
        except Exception as e:
            print(f"Warning: Error getting previous verses: {e}")
            return []

def translate_with_dspy(text: str, scenario: 'TranslationScenario', 
                       text_id: Optional[str] = None, max_tokens: int = 1000) -> Dict:
    """Main translation function using DSPy."""
    start_time = datetime.now()
    
    try:
        # Set up DSPy with the correct LM class for newer versions and the specified max_tokens
        lm = dspy.LM(
            f"openai/{scenario.config['models']['translation']}", 
            max_tokens=max_tokens,
            temperature=0.1  # slight increase from default 0.0 to help with diversity
        )
        dspy.configure(lm=lm)
        
        # Create translation memory
        memory_path = scenario.base_path / f"{scenario.source_code}_{scenario.target_code}_memory.jsonl"
        translation_memory = TranslationMemory(memory_path)
        
        # Create and run translation program
        program = TranslationProgram()
        result = program(text, scenario, translation_memory, text_id)
        
        # Add timing information
        result["translation_time"] = round((datetime.now() - start_time).total_seconds(), 2)
        
        return result
        
    except Exception as e:
        print(f"Error translating text: '{text}'")
        print(f"Error details: {str(e)}")
        return {
            "source_lang": scenario.source_code,
            "target_lang": scenario.target_code,
            "original": text,
            "translation": f"[Translation failed: {str(e)}]",
            "error": str(e),
            "id": text_id if text_id else None,
            "calver": datetime.now().strftime("%Y.%m.%d")
        } 