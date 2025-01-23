import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import numpy as np
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import json

@dataclass
class TranslationPair:
    source: str
    target: str
    id: str

def load_jsonl(file_path: str) -> List[TranslationPair]:
    """Load translation pairs from JSONL file."""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pair = TranslationPair(
                source=data['original'],
                target=data['translation'],
                id=data['id']
            )
            pairs.append(pair)
    return pairs

class StatisticalAligner:
    def __init__(self, stop_word_threshold: float = 0.1):
        self.stop_word_threshold = stop_word_threshold
        # Core statistics
        self.co_occurrences = defaultdict(lambda: defaultdict(int))
        self.source_counts = defaultdict(int)
        self.target_counts = defaultdict(int)
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        self.stop_words = set()

    def calculate_stop_words(self, sentences: List[str]) -> Set[str]:
        """Calculate stop words using Zipfian distribution"""
        words = [w for s in sentences for w in self.tokenize(s)]
        freq = Counter(words)
        total = len(sentences)
        
        # Words appearing in > threshold% of sentences are stop words
        return {word for word, count in freq.items() 
               if count / total > self.stop_word_threshold}

    def train(self, pairs: List[TranslationPair]):
        """Train model on translation pairs"""
        # Reset statistics
        self.co_occurrences.clear()
        self.source_counts.clear()
        self.target_counts.clear()
        self.doc_freq.clear()
        
        # Calculate stop words on full corpus
        all_text = [p.source + " " + p.target for p in pairs]
        self.stop_words = self.calculate_stop_words(all_text)
        self.total_docs = len(pairs)

        # Collect statistics
        for pair in pairs:
            src_tokens = self.tokenize(pair.source)
            tgt_tokens = self.tokenize(pair.target)
            
            # Get trigrams
            src_trigrams = self.get_trigrams(src_tokens)
            tgt_trigrams = self.get_trigrams(tgt_tokens)
            
            # Update counts for both unigrams and trigrams
            self._update_counts(src_tokens, tgt_tokens)
            self._update_counts(src_trigrams, tgt_trigrams)
            
            # Update document frequencies
            for token in set(src_tokens + src_trigrams):
                self.doc_freq[token] += 1

    def _update_counts(self, source_tokens: List[str], target_tokens: List[str]):
        """Update co-occurrence and count statistics"""
        for s in source_tokens:
            if s not in self.stop_words:
                self.source_counts[s] += 1
                for t in target_tokens:
                    if t not in self.stop_words:
                        self.co_occurrences[s][t] += 1
        
        for t in target_tokens:
            if t not in self.stop_words:
                self.target_counts[t] += 1

    def align(self, source: str, target: str) -> List[Tuple[str, str, float]]:
        """Align source and target sentences using trigrams to guide unigram alignment"""
        src_tokens = self.tokenize(source)
        tgt_tokens = self.tokenize(target)
        
        # First pass: align trigrams
        src_trigrams = self.get_trigrams(src_tokens) 
        tgt_trigrams = self.get_trigrams(tgt_tokens)
        
        trigram_alignments = []
        for i, s_tri in enumerate(src_trigrams):
            best_score = 0
            best_target = None
            
            for j, t_tri in enumerate(tgt_trigrams):
                score = self._calculate_score(s_tri, t_tri, i, j, 
                                           len(src_trigrams), len(tgt_trigrams))
                if score > best_score:
                    best_score = score
                    best_target = t_tri
            
            if best_target:
                trigram_alignments.append((s_tri, best_target, best_score))
        
        # Second pass: use trigrams to guide unigram alignment
        alignments = []
        aligned_sources = set()
        aligned_targets = set()
        
        # Process non-stop words only
        src_words = [t for t in src_tokens if t not in self.stop_words]
        tgt_words = [t for t in tgt_tokens if t not in self.stop_words]
        
        for s_word in src_words:
            best_score = 0
            best_target = None
            
            # Check if word appears in any aligned trigram
            trigram_context = self._get_trigram_context(s_word, trigram_alignments)
            
            for t_word in tgt_words:
                if t_word in aligned_targets:
                    continue
                    
                base_score = self._calculate_score(s_word, t_word, 
                    src_words.index(s_word), tgt_words.index(t_word),
                    len(src_words), len(tgt_words))
                
                # Boost score if target appears in aligned trigram context
                if trigram_context and t_word in trigram_context:
                    base_score *= 2.0
                
                if base_score > best_score:
                    best_score = base_score
                    best_target = t_word
            
            if best_target and best_score > 0:
                alignments.append((s_word, best_target, best_score))
                aligned_sources.add(s_word)
                aligned_targets.add(best_target)
        
        return sorted(alignments, key=lambda x: x[2], reverse=True)

    def _calculate_score(self, source: str, target: str, 
                        src_pos: int, tgt_pos: int,
                        src_len: int, tgt_len: int) -> float:
        """Calculate alignment score using TF-IDF and position"""
        if source in self.stop_words or target in self.stop_words:
            return 0
        
        co_occur = self.co_occurrences[source][target]
        if co_occur == 0:
            return 0
            
        # TF-IDF score
        src_count = max(1, self.source_counts[source])
        tgt_count = max(1, self.target_counts[target])
        idf = math.log((self.total_docs + 1) / (self.doc_freq[source] + 1))
        
        tf_idf = (co_occur / src_count) * (co_occur / tgt_count) * idf
        
        # Position score
        pos_diff = abs((src_pos / max(1, src_len)) - (tgt_pos / max(1, tgt_len)))
        pos_score = 1 - pos_diff
        
        return tf_idf * pos_score

    def _get_trigram_context(self, word: str, 
                            trigram_alignments: List[Tuple[str, str, float]]) -> Set[str]:
        """Get target context from trigram alignments containing the source word"""
        context = set()
        for src_tri, tgt_tri, _ in trigram_alignments:
            if word in src_tri.split():
                context.update(tgt_tri.split())
        return context

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple word tokenization"""
        return re.findall(r'\w+', text.lower())
    
    @staticmethod
    def get_trigrams(tokens: List[str]) -> List[str]:
        """Generate trigrams from token list"""
        return [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]

def consolidate_alignments(alignments: List[Tuple[str, str, float]], 
                         threshold: float = 0.5) -> Dict[str, List[Tuple[str, str, float]]]:
    """Consolidate alignments by filtering out high-confidence matches."""
    # Separate unigrams and trigrams
    unigrams = [(s, t, score) for s, t, score in alignments 
                if len(s.split()) == 1 and len(t.split()) == 1]
    trigrams = [(s, t, score) for s, t, score in alignments 
                if len(s.split()) > 1 or len(t.split()) > 1]
    
    # Sort by confidence
    unigrams.sort(key=lambda x: x[2], reverse=True)
    trigrams.sort(key=lambda x: x[2], reverse=True)
    
    # Track aligned words
    aligned_sources = set()
    aligned_targets = set()
    final_unigrams = []
    
    # Filter unigrams
    for src, tgt, score in unigrams:
        if score < threshold:
            continue
        if src not in aligned_sources and tgt not in aligned_targets:
            final_unigrams.append((src, tgt, score))
            aligned_sources.add(src)
            aligned_targets.add(tgt)
    
    return {
        'unigrams': final_unigrams,
        'trigrams': trigrams
    }

def train_model(pairs: List[TranslationPair], stop_word_threshold: float = 0.1) -> Dict:
    """Train statistical alignment model on translation pairs."""
    # Core statistics
    co_occurrences = defaultdict(lambda: defaultdict(int))
    source_counts = defaultdict(int)
    target_counts = defaultdict(int)
    doc_freq = defaultdict(int)
    total_docs = len(pairs)
    
    # Calculate stop words on full corpus
    all_text = [p.source + " " + p.target for p in pairs]
    words = [w for s in all_text for w in re.findall(r'\w+', s.lower())]
    freq = Counter(words)
    
    # Words appearing in > threshold% of sentences are stop words
    stop_words = {word for word, count in freq.items() 
                 if count / total_docs > stop_word_threshold}
    
    # Collect statistics
    for pair in pairs:
        src_tokens = re.findall(r'\w+', pair.source.lower())
        tgt_tokens = re.findall(r'\w+', pair.target.lower())
        
        # Get trigrams
        src_trigrams = [' '.join(src_tokens[i:i+3]) for i in range(len(src_tokens)-2)]
        tgt_trigrams = [' '.join(tgt_tokens[i:i+3]) for i in range(len(tgt_tokens)-2)]
        
        # Update counts for both unigrams and trigrams
        for s in src_tokens + src_trigrams:
            if s not in stop_words:
                source_counts[s] += 1
                for t in tgt_tokens + tgt_trigrams:
                    if t not in stop_words:
                        co_occurrences[s][t] += 1
        
        for t in tgt_tokens + tgt_trigrams:
            if t not in stop_words:
                target_counts[t] += 1
        
        # Update document frequencies
        for token in set(src_tokens + src_trigrams):
            doc_freq[token] += 1
    
    return {
        'co_occurrences': co_occurrences,
        'source_counts': source_counts,
        'target_counts': target_counts,
        'doc_freq': doc_freq,
        'total_docs': total_docs,
        'stop_words': stop_words
    }

def align_sentence_pair(model: Dict, source: str, target: str) -> List[Tuple[str, str, float]]:
    """Align source and target sentences using trained model."""
    # Extract model components
    co_occurrences = model['co_occurrences']
    source_counts = model['source_counts']
    target_counts = model['target_counts']
    doc_freq = model['doc_freq']
    total_docs = model['total_docs']
    stop_words = model['stop_words']
    
    # Tokenize input
    src_tokens = re.findall(r'\w+', source.lower())
    tgt_tokens = re.findall(r'\w+', target.lower())
    
    # Get trigrams
    src_trigrams = [' '.join(src_tokens[i:i+3]) for i in range(len(src_tokens)-2)]
    tgt_trigrams = [' '.join(tgt_tokens[i:i+3]) for i in range(len(tgt_tokens)-2)]
    
    alignments = []
    
    # Process all tokens (unigrams and trigrams)
    src_all = src_tokens + src_trigrams
    tgt_all = tgt_tokens + tgt_trigrams
    
    for i, s in enumerate(src_all):
        if s in stop_words:
            continue
            
        best_score = 0
        best_target = None
        
        for j, t in enumerate(tgt_all):
            if t in stop_words:
                continue
                
            # Calculate alignment score
            co_occur = co_occurrences[s][t]
            if co_occur == 0:
                continue
                
            # TF-IDF style score
            src_count = max(1, source_counts[s])
            tgt_count = max(1, target_counts[t])
            idf = math.log((total_docs + 1) / (doc_freq[s] + 1))
            
            tf_idf = (co_occur / src_count) * (co_occur / tgt_count) * idf
            
            # Position score
            pos_diff = abs((i / len(src_all)) - (j / len(tgt_all)))
            pos_score = 1 - pos_diff
            
            score = tf_idf * pos_score
            
            if score > best_score:
                best_score = score
                best_target = t
        
        if best_target and best_score > 0:
            alignments.append((s, best_target, best_score))
    
    return sorted(alignments, key=lambda x: x[2], reverse=True)

def main(jsonl_path: str, alignment_threshold: float = 0.5):
    """Main function to process JSONL file and analyze alignments."""
    console = Console()
    
    with Progress() as progress:
        # Load pairs
        load_task = progress.add_task("[cyan]Loading pairs...", total=1)
        pairs = load_jsonl(jsonl_path)
        progress.update(load_task, completed=1)
        
        # Train model
        train_task = progress.add_task("[green]Training model...", total=1)
        model = train_model(pairs)
        progress.update(train_task, completed=1)
        
        # Process alignments
        align_task = progress.add_task("[yellow]Processing alignments...", total=len(pairs))
        results = []
        for pair in pairs:
            raw_alignments = align_sentence_pair(model, pair.source, pair.target)
            consolidated = consolidate_alignments(raw_alignments, alignment_threshold)
            
            # Calculate composite score
            unigram_score = sum(score for _, _, score in consolidated['unigrams']) / (len(consolidated['unigrams']) or 1)
            trigram_score = sum(score for _, _, score in consolidated['trigrams']) / (len(consolidated['trigrams']) or 1)
            composite_score = (unigram_score + trigram_score) / 2
            
            results.append({
                'id': pair.id,
                'consolidated': consolidated,
                'composite_score': composite_score
            })
            progress.update(align_task, advance=1)
    
    # Sort results by composite score
    sorted_results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
    
    # Create summary table
    table = Table(title="Alignment Analysis Results")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")
    table.add_column("Unigram Alignments", style="green")
    table.add_column("Trigram Alignments", style="yellow")
    table.add_column("Source", style="white", overflow="fold")
    table.add_column("Target", style="white", overflow="fold")
    
    # Show top 10 results
    for result in sorted_results[:10]:
        pair = next(p for p in pairs if p.id == result['id'])
        unigrams = result['consolidated']['unigrams']
        trigrams = result['consolidated']['trigrams']
        
        unigram_str = '\n'.join(f"{s}->{t} ({score:.2f})" 
                               for s, t, score in unigrams[:3])
        trigram_str = '\n'.join(f"{s}->{t} ({score:.2f})" 
                               for s, t, score in trigrams[:3])
        
        table.add_row(
            result['id'],
            f"{result['composite_score']:.3f}",
            unigram_str or "None",
            trigram_str or "None",
            pair.source[:100] + ('...' if len(pair.source) > 100 else ''),
            pair.target[:100] + ('...' if len(pair.target) > 100 else '')
        )
    
    console.print("\nAnalysis Complete!")
    console.print(f"Processed {len(pairs)} translation pairs")
    console.print(table)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze alignments in translation pairs")
    parser.add_argument("jsonl_path", type=str, help="Path to JSONL file")
    parser.add_argument("--threshold", type=float, default=0.5,
                      help="Threshold for alignment confidence")
    
    args = parser.parse_args()
    main(args.jsonl_path, args.threshold)