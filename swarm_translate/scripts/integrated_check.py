import re
import math
import zlib
import random
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

@dataclass
class TranslationPair:
    source: str
    target: str
    id: str
    translation_time: float
    model: str

@dataclass
class AlignmentResult:
    unigram_alignments: List[Tuple[str, str, float]]
    trigram_alignments: List[Tuple[str, str, float]]
    composite_score: float

class IntegratedAnalyzer:
    def __init__(self, stop_word_threshold: float = 0.1):
        # Alignment-related attributes
        self.stop_word_threshold = stop_word_threshold
        self.co_occurrences = defaultdict(lambda: defaultdict(int))
        self.source_counts = defaultdict(int)
        self.target_counts = defaultdict(int)
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        self.stop_words = set()
        
        # Anomaly detection attributes
        self.noise_len = 50
        self.trials = 5
        self.std_multiplier = 2.0

    def analyze_translation_pair(self, pair: TranslationPair) -> Dict:
        """Comprehensive analysis of a translation pair."""
        # Get alignment data
        alignment_result = self.align_pair(pair.source, pair.target)
        
        # Get anomaly score
        anomaly_score = self.calculate_anomaly_score(pair)
        
        # Calculate alignment-based features
        alignment_features = self.extract_alignment_features(alignment_result)
        
        # Combine scores
        combined_score = self.combine_scores(anomaly_score, alignment_features)
        
        return {
            'id': pair.id,
            'anomaly_score': anomaly_score,
            'alignment_score': alignment_result.composite_score,
            'combined_score': combined_score,
            'unigram_alignments': alignment_result.unigram_alignments,
            'trigram_alignments': alignment_result.trigram_alignments,
            'alignment_features': alignment_features
        }

    def train(self, pairs: List[TranslationPair]):
        """Train the alignment model."""
        # Reset statistics
        self.co_occurrences.clear()
        self.source_counts.clear()
        self.target_counts.clear()
        self.doc_freq.clear()
        
        # Calculate stop words
        all_text = [p.source + " " + p.target for p in pairs]
        self.stop_words = self.calculate_stop_words(all_text)
        self.total_docs = len(pairs)
        
        # Collect statistics
        for pair in pairs:
            src_tokens = self.tokenize(pair.source)
            tgt_tokens = self.tokenize(pair.target)
            
            src_trigrams = self.get_trigrams(src_tokens)
            tgt_trigrams = self.get_trigrams(tgt_tokens)
            
            self._update_counts(src_tokens, tgt_tokens)
            self._update_counts(src_trigrams, tgt_trigrams)
            
            for token in set(src_tokens + src_trigrams):
                self.doc_freq[token] += 1

    def align_pair(self, source: str, target: str) -> AlignmentResult:
        """Align source and target text using unified n-gram approach."""
        # Get initial alignments
        unigram_alignments = self.get_unigram_alignments(source, target)
        trigram_alignments = self.get_trigram_alignments(source, target)
        
        # Calculate composite score
        unigram_score = sum(score for _, _, score in unigram_alignments) / (len(unigram_alignments) or 1)
        trigram_score = sum(score for _, _, score in trigram_alignments) / (len(trigram_alignments) or 1)
        composite_score = (unigram_score + trigram_score) / 2
        
        return AlignmentResult(
            unigram_alignments=unigram_alignments,
            trigram_alignments=trigram_alignments,
            composite_score=composite_score
        )

    def calculate_anomaly_score(self, pair: TranslationPair) -> float:
        """Calculate anomaly score for a translation pair."""
        base = self.dist_metric(pair.source, pair.target)
        shift_sum = 0
        
        for _ in range(self.trials):
            noise = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=self.noise_len))
            shift = abs(self.dist_metric(pair.source + noise, pair.target + noise) - base)
            shift_sum += shift
        
        return shift_sum / self.trials

    def extract_alignment_features(self, alignment_result: AlignmentResult) -> Dict:
        """Extract features from alignment results."""
        return {
            'unigram_coverage': len(alignment_result.unigram_alignments),
            'trigram_coverage': len(alignment_result.trigram_alignments),
            'avg_unigram_score': np.mean([score for _, _, score in alignment_result.unigram_alignments]) if alignment_result.unigram_alignments else 0,
            'avg_trigram_score': np.mean([score for _, _, score in alignment_result.trigram_alignments]) if alignment_result.trigram_alignments else 0,
            'max_unigram_score': max([score for _, _, score in alignment_result.unigram_alignments]) if alignment_result.unigram_alignments else 0,
            'max_trigram_score': max([score for _, _, score in alignment_result.trigram_alignments]) if alignment_result.trigram_alignments else 0
        }

    def combine_scores(self, anomaly_score: float, alignment_features: Dict) -> float:
        """Combine anomaly and alignment scores into a final score."""
        # Normalize scores
        normalized_anomaly = anomaly_score / 100  # Adjust based on your typical score ranges
        normalized_alignment = (
            alignment_features['avg_unigram_score'] + 
            alignment_features['avg_trigram_score']
        ) / 2
        
        # Higher alignment scores should reduce the anomaly score
        combined = normalized_anomaly * (1 - normalized_alignment)
        return combined

    # Helper methods from the alignment code
    def calculate_stop_words(self, sentences: List[str]) -> Set[str]:
        """Calculate stop words using Zipfian distribution."""
        words = [w for s in sentences for w in self.tokenize(s)]
        freq = Counter(words)
        total = len(sentences)
        return {word for word, count in freq.items() 
               if count / total > self.stop_word_threshold}

    def _update_counts(self, source_tokens: List[str], target_tokens: List[str]):
        """Update co-occurrence and count statistics."""
        for s in source_tokens:
            if s not in self.stop_words:
                self.source_counts[s] += 1
                for t in target_tokens:
                    if t not in self.stop_words:
                        self.co_occurrences[s][t] += 1
        
        for t in target_tokens:
            if t not in self.stop_words:
                self.target_counts[t] += 1

    def get_unigram_alignments(self, source: str, target: str) -> List[Tuple[str, str, float]]:
        """Get unigram alignments between source and target."""
        src_tokens = [t for t in self.tokenize(source) if t not in self.stop_words]
        tgt_tokens = [t for t in self.tokenize(target) if t not in self.stop_words]
        
        alignments = []
        for i, s_token in enumerate(src_tokens):
            best_score = 0
            best_target = None
            
            for j, t_token in enumerate(tgt_tokens):
                score = self._calculate_score(s_token, t_token, i, j, len(src_tokens), len(tgt_tokens))
                if score > best_score:
                    best_score = score
                    best_target = t_token
            
            if best_target and best_score > 0:
                alignments.append((s_token, best_target, best_score))
        
        return sorted(alignments, key=lambda x: x[2], reverse=True)

    def get_trigram_alignments(self, source: str, target: str) -> List[Tuple[str, str, float]]:
        """Get trigram alignments between source and target."""
        src_tokens = self.tokenize(source)
        tgt_tokens = self.tokenize(target)
        
        src_trigrams = self.get_trigrams(src_tokens)
        tgt_trigrams = self.get_trigrams(tgt_tokens)
        
        alignments = []
        for i, s_tri in enumerate(src_trigrams):
            best_score = 0
            best_target = None
            
            for j, t_tri in enumerate(tgt_trigrams):
                score = self._calculate_score(s_tri, t_tri, i, j, len(src_trigrams), len(tgt_trigrams))
                if score > best_score:
                    best_score = score
                    best_target = t_tri
            
            if best_target and best_score > 0:
                alignments.append((s_tri, best_target, best_score))
        
        return sorted(alignments, key=lambda x: x[2], reverse=True)

    def _calculate_score(self, source: str, target: str, src_pos: int, tgt_pos: int,
                        src_len: int, tgt_len: int) -> float:
        """Calculate alignment score using TF-IDF and position."""
        if source in self.stop_words or target in self.stop_words:
            return 0
        
        co_occur = self.co_occurrences[source][target]
        if co_occur == 0:
            return 0
        
        src_count = max(1, self.source_counts[source])
        tgt_count = max(1, self.target_counts[target])
        idf = math.log((self.total_docs + 1) / (self.doc_freq[source] + 1))
        
        tf_idf = (co_occur / src_count) * (co_occur / tgt_count) * idf
        pos_score = 1 - abs((src_pos / max(1, src_len)) - (tgt_pos / max(1, tgt_len)))
        
        return tf_idf * pos_score

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple word tokenization."""
        return re.findall(r'\w+', text.lower())

    @staticmethod
    def get_trigrams(tokens: List[str]) -> List[str]:
        """Generate trigrams from token list."""
        return [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]

    @staticmethod
    def dist_metric(s1: str, s2: str) -> float:
        """Calculate distance metric between two strings."""
        return abs(len(s1) - len(s2)) + abs(len(zlib.compress(s1.encode())) - len(zlib.compress(s2.encode())))

def analyze_translations(pairs: List[TranslationPair]) -> List[Dict]:
    """Analyze a list of translation pairs."""
    analyzer = IntegratedAnalyzer()
    
    # Train the alignment model
    analyzer.train(pairs)
    
    # Analyze each pair
    results = []
    for pair in pairs:
        result = analyzer.analyze_translation_pair(pair)
        results.append(result)
    
    return results

def display_results(results: List[Dict], pairs: List[TranslationPair], top_n: int = 10):
    """Display analysis results."""
    console = Console()
    
    # Sort by combined score
    sorted_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    top_results = sorted_results[:top_n]
    
    # Create results table
    table = Table(title=f"Top {top_n} Anomalous Translations with Alignments")
    table.add_column("ID", style="cyan")
    table.add_column("Combined Score", style="red")
    table.add_column("Alignment Score", style="yellow")
    table.add_column("Top Alignments", style="green")
    table.add_column("Source", style="white", overflow="fold")
    table.add_column("Target", style="white", overflow="fold")
    
    for result in top_results:
        pair = next(p for p in pairs if p.id == result['id'])
        top_alignments = sorted(
            result['unigram_alignments'] + result['trigram_alignments'],
            key=lambda x: x[2],
            reverse=True
        )[:3]
        
        alignment_str = '\n'.join(f"{s}->{t} ({score:.2f})" for s, t, score in top_alignments)
        
        table.add_row(
            result['id'],
            f"{result['combined_score']:.3f}",
            f"{result['alignment_score']:.3f}",
            alignment_str,
            pair.source[:100] + ('...' if len(pair.source) > 100 else ''),
            pair.target[:100] + ('...' if len(pair.target) > 100 else '')
        )
    
    console.print(table)

def main(jsonl_path: str):
    """Main function to process JSONL file."""
    console = Console()
    with Progress() as progress:
        task1 = progress.add_task("[cyan]Loading pairs...", total=1)
        pairs = load_jsonl(jsonl_path)
        progress.update(task1, completed=1)
        
        task2 = progress.add_task("[green]Analyzing translations...", total=1)
        results = analyze_translations(pairs)
        progress.update(task2, completed=1)
    
    display_results(results, pairs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze translations with integrated alignment")
    parser.add_argument("jsonl_path", type=str, help="Path to JSONL file")
    args = parser.parse_args()
    main(args.jsonl_path)