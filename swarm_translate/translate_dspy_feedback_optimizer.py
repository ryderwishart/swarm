import dspy
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from dspy.teleprompt import BootstrapFewShot, MIPRO
from dspy.evaluate import Evaluate

# Load environment variables
load_dotenv()

class TranslationAnalysis(dspy.Signature):
    """Analyze text for translation components."""
    text: str = dspy.InputField()
    source_label: str = dspy.InputField()
    target_label: str = dspy.InputField()
    book_context: str = dspy.InputField(default="")
    
    phrases: List[str] = dspy.OutputField(desc="List of main phrases")
    terms: List[str] = dspy.OutputField(desc="Key terms")
    idioms: List[str] = dspy.OutputField(desc="Idiomatic expressions")
    cultural: List[str] = dspy.OutputField(desc="Cultural references")

class TranslationComponents(dspy.Signature):
    """Translate analyzed components."""
    analysis: Dict = dspy.InputField()
    source_label: str = dspy.InputField()
    target_label: str = dspy.InputField()
    style_prompt: str = dspy.InputField()
    memory_context: str = dspy.InputField(default="")
    
    phrases: List[str] = dspy.OutputField(desc="Translated phrases")
    terms: List[str] = dspy.OutputField(desc="Translated terms")
    idioms: List[str] = dspy.OutputField(desc="Translated expressions")
    cultural: List[str] = dspy.OutputField(desc="Translated references")

class TranslationCombination(dspy.Signature):
    """Combine translated components into final translation."""
    original: str = dspy.InputField()
    translations: Dict = dspy.InputField()
    source_label: str = dspy.InputField()
    target_label: str = dspy.InputField()
    style_prompt: str = dspy.InputField()
    
    translation: str = dspy.OutputField(desc="Final translation")

class TranslationConsolidation(dspy.Signature):
    """Ensure consistency with previous verses."""
    final_translation: str = dspy.InputField()
    original: str = dspy.InputField()
    book_context: str = dspy.InputField()
    previous_verses: List[Dict] = dspy.InputField()
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
        entry = {
            'analysis': analysis,
            'translation': translation
        }
        key = json.dumps(analysis.get('terms', []), ensure_ascii=False)
        self.memory[key] = entry
        
        # Save to file
        with open(self.memory_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

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

@dataclass
class TranslationFeedback:
    """Feedback for a translation."""
    original: str
    translation: str
    score: float  # 0-1 score
    feedback: str  # Detailed feedback
    source_label: str
    target_label: str

class TranslationOptimizer(dspy.Module):
    """Optimizes translation quality based on feedback."""
    def __init__(self):
        super().__init__()
        self.optimizer = MIPRO(
            metric=dspy.evaluate.answer_exact_match,
            num_trials=3,
            max_bootstrapped_demos=5
        )
        self.few_shot = BootstrapFewShot(
            metric=dspy.evaluate.answer_exact_match,
            max_bootstrapped_demos=5
        )
    
    def forward(self, program: TranslationProgram, 
                feedback_data: List[TranslationFeedback]) -> TranslationProgram:
        """Optimize the translation program using feedback."""
        # Convert feedback to examples
        examples = []
        for fb in feedback_data:
            example = dspy.Example(
                text=fb.original,
                source_label=fb.source_label,
                target_label=fb.target_label,
                translation=fb.translation,
                score=fb.score,
                feedback=fb.feedback
            )
            examples.append(example)
        
        # First optimize with few-shot learning
        program = self.few_shot.compile(program, trainset=examples)
        
        # Then optimize with MIPRO for more complex improvements
        program = self.optimizer.compile(program, trainset=examples)
        
        return program

class TranslationEvaluator(dspy.Module):
    """Evaluates translation quality."""
    def __init__(self):
        super().__init__()
        self.evaluator = dspy.ChainOfThought("translation: str, original: str, source_label: str, target_label: str -> score: float, feedback: str")
    
    def forward(self, translation: str, original: str, 
                source_label: str, target_label: str) -> Tuple[float, str]:
        """Evaluate a translation and provide feedback."""
        result = self.evaluator(
            translation=translation,
            original=original,
            source_label=source_label,
            target_label=target_label
        )
        return result.score, result.feedback

def optimize_translations(scenario: 'TranslationScenario', 
                         feedback_data: List[TranslationFeedback],
                         num_iterations: int = 3) -> TranslationProgram:
    """Optimize translations using feedback data."""
    # Initialize program and optimizer
    program = TranslationProgram()
    optimizer = TranslationOptimizer()
    
    # Optimize program
    optimized_program = optimizer(program, feedback_data)
    
    # Evaluate improvements
    evaluator = TranslationEvaluator()
    scores = []
    
    for fb in feedback_data:
        # Get translation from optimized program
        result = optimized_program(
            text=fb.original,
            scenario=scenario,
            translation_memory=TranslationMemory(
                scenario.base_path / f"{scenario.source_code}_{scenario.target_code}_memory.jsonl"
            )
        )
        
        # Evaluate
        score, feedback = evaluator(
            translation=result["translation"],
            original=fb.original,
            source_label=scenario.source_label,
            target_label=scenario.target_label
        )
        scores.append(score)
    
    print(f"Average improvement: {sum(scores)/len(scores):.2f}")
    return optimized_program

def collect_feedback(scenario: 'TranslationScenario', 
                    texts: List[str],
                    text_ids: Optional[List[str]] = None) -> List[TranslationFeedback]:
    """Collect feedback for a set of translations."""
    feedback_data = []
    
    # Create initial program
    program = TranslationProgram()
    
    # Create translation memory
    memory_path = scenario.base_path / f"{scenario.source_code}_{scenario.target_code}_memory.jsonl"
    translation_memory = TranslationMemory(memory_path)
    
    # Translate and collect feedback
    for i, text in enumerate(texts):
        text_id = text_ids[i] if text_ids else None
        
        # Get translation
        result = program(
            text=text,
            scenario=scenario,
            translation_memory=translation_memory,
            text_id=text_id
        )
        
        # Get feedback (in a real system, this would come from human reviewers)
        evaluator = TranslationEvaluator()
        score, feedback = evaluator(
            translation=result["translation"],
            original=text,
            source_label=scenario.source_label,
            target_label=scenario.target_label
        )
        
        feedback_data.append(TranslationFeedback(
            original=text,
            translation=result["translation"],
            score=score,
            feedback=feedback,
            source_label=scenario.source_label,
            target_label=scenario.target_label
        ))
    
    return feedback_data

def translate_with_optimization(text: str, scenario: 'TranslationScenario',
                              text_id: Optional[str] = None,
                              feedback_data: Optional[List[TranslationFeedback]] = None) -> Dict:
    """Main translation function with optimization support."""
    start_time = datetime.now()
    
    try:
        # Set up DSPy
        dspy.configure(lm=dspy.OpenAI(model=scenario.config["models"]["translation"]))
        
        # Create translation memory
        memory_path = scenario.base_path / f"{scenario.source_code}_{scenario.target_code}_memory.jsonl"
        translation_memory = TranslationMemory(memory_path)
        
        # Get or create optimized program
        if feedback_data:
            program = optimize_translations(scenario, feedback_data)
        else:
            program = TranslationProgram()
        
        # Run translation
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

# Example usage:
"""
# 1. Collect initial feedback
texts = ["Hello world", "How are you?", "Good morning"]
feedback = collect_feedback(scenario, texts)

# 2. Optimize translations using feedback
optimized_program = optimize_translations(scenario, feedback)

# 3. Use optimized program for new translations
result = translate_with_optimization(
    text="New text to translate",
    scenario=scenario,
    feedback_data=feedback
)
""" 