import re
import math
import zlib
import json
import random
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import os
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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
    gloss_predictions: Dict[str, str]  # Add gloss predictions

class IntegratedAnalyzer:
    def __init__(self, stop_word_threshold: float = 0.01):
        # Alignment-related attributes
        self.stop_word_threshold = stop_word_threshold
        self.co_occurrences = defaultdict(lambda: defaultdict(int))
        self.source_counts = defaultdict(int)
        self.target_counts = defaultdict(int)
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        
        # Predefined stop words for Greek and English
        self.base_stop_words = {
            # English stop words
            'the', 'and', 'of', 'to', 'in', 'that', 'for', 'is', 'on', 'with',
            'be', 'as', 'by', 'at', 'this', 'was', 'were', 'are', 'from', 'have',
            'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            # Greek stop words
            'καὶ', 'τὸν', 'τῆς', 'τὸ', 'τοῦ', 'τῶν', 'ἐν', 'δὲ', 'ὁ', 'αὐτοῦ',
            'αὐτῶν', 'αὐτῷ', 'εἰς', 'ἐπὶ', 'πρὸς', 'μὲν', 'οὖν', 'γὰρ', 'δέ',
            'τε', 'οὐ', 'μὴ', 'ἀλλά', 'ἀλλ'
        }
        self.stop_words = self.base_stop_words.copy()
        
        # Anomaly detection attributes
        self.noise_len = 50
        self.trials = 5
        self.std_multiplier = 2.0
        
        # Cache for TF-IDF scores and frequencies
        self.tfidf_cache = {}
        self.word_freq_cache = {}
        
        # Cache for compressed string lengths
        self.compressed_length_cache = {}
        
        # Additional caches for score calculation
        self.src_count_cache = {}
        self.tgt_count_cache = {}
        self.idf_cache = {}
        
        # Logging control
        self.log_frequency = 1000  # Log every nth pair
        self.pairs_processed = 0

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
            'gloss_predictions': alignment_result.gloss_predictions,  # Add gloss predictions
            'alignment_features': alignment_features,
            'original': pair.source,  # Add source text
            'translation': pair.target  # Add target text
        }

    def train(self, pairs: List[TranslationPair]):
        """Train the alignment model."""
        # Reset statistics and caches
        self.co_occurrences.clear()
        self.source_counts.clear()
        self.target_counts.clear()
        self.doc_freq.clear()
        self.tfidf_cache.clear()
        self.word_freq_cache.clear()
        self.compressed_length_cache.clear()
        self.src_count_cache.clear()
        self.tgt_count_cache.clear()
        self.idf_cache.clear()
        
        # Pre-compute compressed lengths for all source and target texts
        for pair in pairs:
            if pair.source not in self.compressed_length_cache:
                self.compressed_length_cache[pair.source] = len(zlib.compress(pair.source.encode()))
            if pair.target not in self.compressed_length_cache:
                self.compressed_length_cache[pair.target] = len(zlib.compress(pair.target.encode()))
        
        # Calculate stop words
        all_text = [p.source + " " + p.target for p in pairs]
        self.stop_words = self.calculate_stop_words(all_text)
        self.total_docs = len(pairs)
        
        # Collect statistics
        for pair in pairs:
            # Tokenize and generate trigrams once
            src_tokens = self.tokenize(pair.source)
            tgt_tokens = self.tokenize(pair.target)
            src_trigrams = self.get_trigrams(src_tokens)
            tgt_trigrams = self.get_trigrams(tgt_tokens)
            
            # Update counts with pre-computed tokens and trigrams
            self._update_counts(src_tokens, tgt_tokens, src_trigrams, tgt_trigrams)
            
            # Update document frequency with pre-computed tokens and trigrams
            for token in set(src_tokens + src_trigrams):
                self.doc_freq[token] += 1

    def get_unigram_alignments(
    self, 
    src_tokens: List[str], 
    tgt_tokens: List[str]
) -> List[Tuple[str, str, float]]:
        """Align unigrams between source and target tokens."""
        alignments = []
        
        # Filter out stop words
        src_tokens_filtered = [t for t in src_tokens if t not in self.stop_words]
        tgt_tokens_filtered = [t for t in tgt_tokens if t not in self.stop_words]
        
        # Align each source token to the best target token
        for s_idx, s_token in enumerate(src_tokens_filtered):
            best_score = 0
            best_target = None
            
            for t_idx, t_token in enumerate(tgt_tokens_filtered):
                # Calculate the alignment score
                score = self._calculate_score(
                    s_token, t_token, s_idx, t_idx, len(src_tokens_filtered), len(tgt_tokens_filtered)
                )
                
                # Update the best alignment for this source token
                if score > best_score:
                    best_score = score
                    best_target = t_token
            
            # Add the best alignment to the results
            if best_target and best_score > 0:
                alignments.append((s_token, best_target, best_score))
        
        return alignments

    def align_pair(self, source: str, target: str) -> AlignmentResult:
        """Align source and target text using trigram-guided unigram alignment and predict glosses."""
        self.pairs_processed += 1
        should_log = self.pairs_processed % self.log_frequency == 0
        
        if should_log:
            print(f"\n=== Processing pair {self.pairs_processed} ===")
            print(f"Source text: {source[:100]}..." if len(source) > 100 else source)
            print(f"Target text: {target[:100]}..." if len(target) > 100 else target)
        
        # Step 1: Tokenize and generate trigrams once
        src_tokens = self.tokenize(source)
        tgt_tokens = self.tokenize(target)
        
        if len(src_tokens) < 3 or len(tgt_tokens) < 3:
            # Skip trigram alignment for short sentences
            unigram_alignments = self.get_unigram_alignments(src_tokens, tgt_tokens)
            composite_score = sum(score for _, _, score in unigram_alignments) / (len(unigram_alignments) or 1)
            
            return AlignmentResult(
                unigram_alignments=unigram_alignments,
                trigram_alignments=[],
                composite_score=composite_score,
                gloss_predictions={}  # No gloss predictions for short sentences
            )
        
        src_trigrams = self.get_trigrams(src_tokens)
        tgt_trigrams = self.get_trigrams(tgt_tokens)
        
        if should_log:
            print(f"Tokens - Source: {len(src_tokens)}, Target: {len(tgt_tokens)}")
            print(f"Trigrams - Source: {len(src_trigrams)}, Target: {len(tgt_trigrams)}")
        
        # Step 2: Get trigram alignments using pre-computed trigrams
        trigram_alignments = self.get_trigram_alignments(
            src_tokens, tgt_tokens, src_trigrams, tgt_trigrams
        )
        
        # Step 3: Use trigram alignments to predict glosses for all non-stop words
        gloss_predictions = self.predict_glosses(src_tokens, trigram_alignments)
        
        # Step 4: Use trigram alignments to narrow down unigram alignments
        unigram_alignments = self.get_unigram_alignments_guided_by_trigrams(
            src_tokens, tgt_tokens, trigram_alignments
        )
        
        if should_log:
            print(f"Alignments found - Trigrams: {len(trigram_alignments)}, Unigrams: {len(unigram_alignments)}")
            print(f"Gloss predictions: {gloss_predictions}")
        
        # Step 5: Calculate composite score
        unigram_score = sum(score for _, _, score in unigram_alignments) / (len(unigram_alignments) or 1)
        trigram_score = sum(score for _, _, score in trigram_alignments) / (len(trigram_alignments) or 1)
        composite_score = (unigram_score + trigram_score) / 2
        
        if should_log:
            print(f"Scores - Unigram: {unigram_score:.4f}, Trigram: {trigram_score:.4f}, Composite: {composite_score:.4f}")
        
        return AlignmentResult(
            unigram_alignments=unigram_alignments,
            trigram_alignments=trigram_alignments,
            composite_score=composite_score,
            gloss_predictions=gloss_predictions  # Add gloss predictions
        )

    def predict_glosses(self, src_tokens: List[str], trigram_alignments: List[Tuple[str, str, float]]) -> Dict[str, str]:
        """Predict glosses for all non-stop words in the source sentence using trigram alignments."""
        gloss_predictions = {}
        
        # Create a mapping of source unigrams to their best glosses
        for s_tri, t_tri, trigram_score in trigram_alignments:
            s_tri_tokens = s_tri.split()
            t_tri_tokens = t_tri.split()
            
            # Find the best gloss for each source unigram in the trigram
            for s_token in s_tri_tokens:
                if s_token in self.stop_words:
                    continue  # Skip stop words
                
                best_gloss = None
                best_score = 0
                
                # Find the best target unigram in the target trigram
                for t_token in t_tri_tokens:
                    if t_token in self.stop_words:
                        continue  # Skip stop words
                    
                    # Calculate the unigram alignment score
                    s_idx = src_tokens.index(s_token)
                    t_idx = t_tri_tokens.index(t_token)
                    unigram_score = self._calculate_score(
                        s_token, t_token, s_idx, t_idx, len(src_tokens), len(t_tri_tokens)
                    )
                    
                    # Combine trigram and unigram scores
                    combined_score = (trigram_score + unigram_score) / 2
                    
                    # Update the best gloss for this source unigram
                    if combined_score > best_score:
                        best_score = combined_score
                        best_gloss = t_token
                
                # Store the best gloss for this source unigram
                if best_gloss and best_score > 0:
                    if s_token not in gloss_predictions or gloss_predictions[s_token][1] < best_score:
                        gloss_predictions[s_token] = (best_gloss, best_score)
        
        # Convert to a dictionary of glosses (without scores)
        return {s_token: gloss for s_token, (gloss, _) in gloss_predictions.items()}

    def get_unigram_alignments_guided_by_trigrams(
        self, 
        src_tokens: List[str], 
        tgt_tokens: List[str], 
        trigram_alignments: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float]]:
        """Get the best unigram alignment for each unigram in each trigram, then consolidate."""
        # Filter stop words from already tokenized text
        src_tokens_filtered = np.array([t for t in src_tokens if t not in self.stop_words])
        tgt_tokens_filtered = np.array([t for t in tgt_tokens if t not in self.stop_words])
        
        # Dictionary to store the best unigram alignment for each source-target pair
        best_unigram_alignments = {}
        
        # Process each trigram alignment
        for s_tri, t_tri, trigram_score in trigram_alignments:
            s_tri_tokens = s_tri.split()
            t_tri_tokens = t_tri.split()
            
            # Find the best unigram alignment for each source unigram in the trigram
            for s_token in s_tri_tokens:
                if s_token in self.stop_words:
                    continue  # Skip stop words
                
                best_score = 0
                best_target = None
                
                # Find the best target unigram in the target trigram
                for t_token in t_tri_tokens:
                    if t_token in self.stop_words:
                        continue  # Skip stop words
                    
                    # Calculate the unigram alignment score
                    s_idx = np.where(src_tokens_filtered == s_token)[0][0]
                    t_idx = np.where(tgt_tokens_filtered == t_token)[0][0]
                    unigram_score = self._calculate_score(
                        s_token, t_token, s_idx, t_idx, len(src_tokens_filtered), len(tgt_tokens_filtered)
                    )
                    
                    # Combine trigram and unigram scores
                    combined_score = (trigram_score + unigram_score) / 2
                    
                    # Update the best alignment for this source unigram
                    if combined_score > best_score:
                        best_score = combined_score
                        best_target = t_token
                
                # Store the best alignment for this source unigram in this trigram
                if best_target and best_score > 0:
                    key = (s_token, best_target)
                    if key not in best_unigram_alignments or best_unigram_alignments[key][2] < best_score:
                        best_unigram_alignments[key] = (s_token, best_target, best_score)
        
        # Convert the dictionary to a sorted list of alignments
        alignments = sorted(best_unigram_alignments.values(), key=lambda x: x[2], reverse=True)
        return alignments

    def calculate_anomaly_score(self, pair: TranslationPair) -> float:
        """Calculate anomaly score for a translation pair using cached compression."""
        # Cache compressed lengths if not already cached
        if pair.source not in self.compressed_length_cache:
            self.compressed_length_cache[pair.source] = len(zlib.compress(pair.source.encode()))
        if pair.target not in self.compressed_length_cache:
            self.compressed_length_cache[pair.target] = len(zlib.compress(pair.target.encode()))
        
        # Calculate base distance using cached values
        base = self.dist_metric(pair.source, pair.target)
        shift_sum = 0
        
        # Pre-generate all noise strings to avoid repeated generation
        noise_strings = [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=self.noise_len)) 
                        for _ in range(self.trials)]
        
        # Pre-compute compressed lengths for noise combinations
        for noise in noise_strings:
            src_with_noise = pair.source + noise
            tgt_with_noise = pair.target + noise
            if src_with_noise not in self.compressed_length_cache:
                self.compressed_length_cache[src_with_noise] = len(zlib.compress(src_with_noise.encode()))
            if tgt_with_noise not in self.compressed_length_cache:
                self.compressed_length_cache[tgt_with_noise] = len(zlib.compress(tgt_with_noise.encode()))
            
            shift = abs(self.dist_metric(src_with_noise, tgt_with_noise) - base)
            shift_sum += shift
        
        return shift_sum / self.trials

    def extract_alignment_features(self, alignment_result: AlignmentResult) -> Dict:
        """Extract features from alignment results using vectorized operations."""
        # Convert alignment scores to numpy arrays for efficient computation
        unigram_scores = np.array([score for _, _, score in alignment_result.unigram_alignments], dtype=np.float32) if alignment_result.unigram_alignments else np.array([], dtype=np.float32)
        trigram_scores = np.array([score for _, _, score in alignment_result.trigram_alignments], dtype=np.float32) if alignment_result.trigram_alignments else np.array([], dtype=np.float32)
        
        return {
            'unigram_coverage': len(unigram_scores),
            'trigram_coverage': len(trigram_scores),
            'avg_unigram_score': np.mean(unigram_scores) if len(unigram_scores) > 0 else 0,
            'avg_trigram_score': np.mean(trigram_scores) if len(trigram_scores) > 0 else 0,
            'max_unigram_score': np.max(unigram_scores) if len(unigram_scores) > 0 else 0,
            'max_trigram_score': np.max(trigram_scores) if len(trigram_scores) > 0 else 0
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

    def calculate_stop_words(self, sentences: List[str]) -> Set[str]:
        """Calculate stop words dynamically using vectorized TF-IDF calculations."""
        if self.pairs_processed % self.log_frequency == 0:
            print("\n=== Calculating stop words ===")
        
        # Start with base stop words
        stop_words = self.base_stop_words.copy()
        
        # Tokenize all sentences and calculate word frequencies once
        words = [w for s in sentences for w in self.tokenize(s)]
        total_words = len(words)
        self.word_freq_cache = Counter(words)
        total_docs = len(sentences)
        
        if self.pairs_processed % self.log_frequency == 0:
            print(f"Total words: {total_words:,}, Total docs: {total_docs:,}")
            print(f"Top 5 most frequent words: {dict(list(Counter(words).most_common(5)))}")
        
        # Only consider words that appear frequently enough
        min_freq = total_words * self.stop_word_threshold
        frequent_words = {word for word, freq in self.word_freq_cache.items() 
                         if freq > min_freq and word not in self.base_stop_words}
        
        # Vectorized TF-IDF calculation for frequent words
        if frequent_words:
            unique_words = np.array(list(frequent_words))
            frequencies = np.array([self.word_freq_cache[word] for word in unique_words], dtype=np.float32)
            doc_freqs = np.array([self.doc_freq.get(word, 0) for word in unique_words], dtype=np.float32)
            
            # Calculate TF-IDF scores in a vectorized way
            tf = frequencies / total_words
            idf = np.log((total_docs + 1) / (doc_freqs + 1))
            tfidf_scores = tf * idf
            
            # Find words with very low TF-IDF scores (likely stop words)
            threshold = np.percentile(tfidf_scores, 10)  # Bottom 10%
            additional_stops = set(unique_words[tfidf_scores < threshold])
            stop_words.update(additional_stops)
        
        if self.pairs_processed % self.log_frequency == 0:
            print(f"Found {len(stop_words):,} stop words")
            print(f"Sample stop words (first 5): {list(stop_words)[:5]}")
        
        return stop_words

    def _find_elbow(self, values: np.ndarray) -> int:
        """Find the elbow point in a numpy array using vectorized operations."""
        if len(values) == 0:
            return 0
        
        # Calculate differences using numpy
        diffs = np.diff(values)
        
        # Find the point where the difference starts to decrease significantly
        change_points = np.where(np.diff(diffs) < 0)[0]
        return change_points[0] + 1 if len(change_points) > 0 else len(values) - 1

    def _update_counts(
        self, 
        source_tokens: List[str], 
        target_tokens: List[str],
        source_trigrams: List[str],
        target_trigrams: List[str]
    ):
        """Update co-occurrence and count statistics efficiently in a single pass."""
        # Pre-filter stop words to avoid repeated checks
        filtered_src_tokens = [s for s in source_tokens if s not in self.stop_words]
        filtered_tgt_tokens = [t for t in target_tokens if t not in self.stop_words]
        filtered_src_trigrams = [s for s in source_trigrams if s not in self.stop_words]
        filtered_tgt_trigrams = [t for t in target_trigrams if t not in self.stop_words]
        
        # Update target counts in a single pass (both unigrams and trigrams)
        for t in filtered_tgt_tokens:
            self.target_counts[t] += 1
        for t in filtered_tgt_trigrams:
            self.target_counts[t] += 1
        
        # Update source counts and co-occurrences for unigrams
        for s in filtered_src_tokens:
            self.source_counts[s] += 1
            for t in filtered_tgt_tokens:
                self.co_occurrences[s][t] += 1
        
        # Update source counts and co-occurrences for trigrams
        for s in filtered_src_trigrams:
            self.source_counts[s] += 1
            for t in filtered_tgt_trigrams:
                self.co_occurrences[s][t] += 1

    def get_trigram_alignments(
        self, 
        src_tokens: List[str], 
        tgt_tokens: List[str],
        src_trigrams: Optional[List[str]] = None,
        tgt_trigrams: Optional[List[str]] = None
    ) -> List[Tuple[str, str, float]]:
        """Get trigram alignments between source and target using pre-computed trigrams if available."""
        # Use pre-computed trigrams if provided, otherwise generate them
        if src_trigrams is None:
            src_trigrams = self.get_trigrams(src_tokens)
        if tgt_trigrams is None:
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
        """Calculate alignment score using cached values for improved performance."""
        if source in self.stop_words or target in self.stop_words:
            return 0
        
        co_occur = self.co_occurrences[source][target]
        if co_occur == 0:
            return 0
        
        # Use cached source count or calculate and cache it
        if source not in self.src_count_cache:
            self.src_count_cache[source] = max(1, self.source_counts[source])
        src_count = self.src_count_cache[source]
        
        # Use cached target count or calculate and cache it
        if target not in self.tgt_count_cache:
            self.tgt_count_cache[target] = max(1, self.target_counts[target])
        tgt_count = self.tgt_count_cache[target]
        
        # Use cached IDF score or calculate and cache it
        if source not in self.idf_cache:
            self.idf_cache[source] = math.log((self.total_docs + 1) / (self.doc_freq[source] + 1))
        
        # Calculate final scores using cached values
        tf_idf = (co_occur / src_count) * (co_occur / tgt_count) * self.idf_cache[source]
        pos_score = 1 - abs((src_pos / max(1, src_len)) - (tgt_pos / max(1, tgt_len)))
        
        final_score = tf_idf * pos_score
        
        # Log high scores and debugging info when needed
        if final_score > 0.5 and self.pairs_processed % self.log_frequency == 0:
            print(f"High score: '{source}' -> '{target}': {final_score:.4f} "
                  f"(co_occur={co_occur}, tf_idf={tf_idf:.4f}, pos={pos_score:.4f})")
        
        return final_score

    def _precompute_counts(self, tokens: List[str], is_source: bool = True):
        """Precompute counts for a set of tokens to optimize score calculation."""
        for token in tokens:
            if token not in self.stop_words:
                if is_source:
                    if token not in self.src_count_cache:
                        self.src_count_cache[token] = max(1, self.source_counts[token])
                    if token not in self.idf_cache:
                        self.idf_cache[token] = math.log((self.total_docs + 1) / (self.doc_freq[token] + 1))
                else:
                    if token not in self.tgt_count_cache:
                        self.tgt_count_cache[token] = max(1, self.target_counts[token])

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text, handling both Greek and English text properly."""
        # Convert to lowercase while preserving Greek diacritics
        text = text.lower()
        
        # Pattern for matching both English and Greek words
        # \w includes Unicode word characters for Greek
        # \u0370-\u03FF is the Greek and Coptic Unicode block
        # \u1F00-\u1FFF is the Greek Extended block for polytonic Greek
        pattern = r'\b\w+\b'
        
        tokens = re.findall(pattern, text)
        
        # Filter out tokens that are too short or just numbers
        tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
        
        return tokens

    @staticmethod
    def get_trigrams(tokens: List[str]) -> List[str]:
        """Generate trigrams from token list."""
        return [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]

    def dist_metric(self, s1: str, s2: str) -> float:
        """Calculate distance metric between two strings using cached compression."""
        # Use cached compressed lengths
        if s1 not in self.compressed_length_cache:
            self.compressed_length_cache[s1] = len(zlib.compress(s1.encode()))
        if s2 not in self.compressed_length_cache:
            self.compressed_length_cache[s2] = len(zlib.compress(s2.encode()))
        
        return abs(len(s1) - len(s2)) + abs(self.compressed_length_cache[s1] - self.compressed_length_cache[s2])

    def consolidate_unigram_alignments(self, unigram_alignments: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """Consolidate unigram alignments to ensure exclusive alignments."""
        # Create dictionaries to track the best alignment for each source and target word
        best_for_source = {}
        best_for_target = {}
        
        for src, tgt, score in unigram_alignments:
            # Update best alignment for the source word
            if src not in best_for_source or best_for_source[src][2] < score:
                best_for_source[src] = (src, tgt, score)
            
            # Update best alignment for the target word
            if tgt not in best_for_target or best_for_target[tgt][2] < score:
                best_for_target[tgt] = (src, tgt, score)
        
        # Merge the best alignments, ensuring exclusivity
        consolidated = {}
        for src, tgt, score in best_for_source.values():
            if tgt in best_for_target and best_for_target[tgt][0] == src:
                consolidated[(src, tgt)] = (src, tgt, score)
        
        return sorted(consolidated.values(), key=lambda x: x[2], reverse=True)

    def display_results(self, results: List[Dict], pairs: List[TranslationPair], top_n: int = 10):
        """Display analysis results with detailed trigram and unigram alignments."""
        console = Console()
        print("\n=== Analysis Summary ===")
        print(f"Total pairs processed: {self.pairs_processed:,}")
        print(f"Total results: {len(results):,}")
        
        # Convert to numpy array for efficient sorting
        results_array = np.array([(r['combined_score'], i) for i, r in enumerate(results)], dtype=[('score', 'f4'), ('index', 'i4')])
        top_indices = np.argsort(results_array['score'])[-top_n:][::-1]
        top_results = [results[i] for i in top_indices]
        print(f"Selected top {len(top_results)} results for display")
        
        # Create efficient lookup for pairs
        pairs_dict = {p.id: p for p in pairs}
        
        # Define a set of colors for unigram alignments
        colors = ["red", "green", "blue", "magenta", "cyan", "yellow"]
        
        for result in top_results:
            print(f"\nProcessing result ID: {result['id']}")
            print(f"Number of trigram alignments: {len(result['trigram_alignments'])}")
            print(f"Number of unigram alignments: {len(result['unigram_alignments'])}")
            
            pair = pairs_dict[result['id']]
            
            # Create a table for the verse
            verse_table = Table(title=f"Verse ID: {pair.id}", show_header=True, header_style="bold magenta")
            verse_table.add_column("Trigram Alignment", style="white")
            verse_table.add_column("Unigram Alignments", style="white")
            verse_table.add_column("Gloss Predictions", style="white")  # Add new column for glosses
            
            # Consolidate unigram alignments
            consolidated_unigrams = self.consolidate_unigram_alignments(result['unigram_alignments'])
            
            # Create optimized mapping of unigrams to trigrams
            trigram_to_unigrams = defaultdict(list)
            unigram_to_trigrams = defaultdict(set)
            
            # Pre-process trigrams for efficient lookup
            trigram_tokens = {
                (s_tri, t_tri): (set(s_tri.split()), set(t_tri.split()))
                for s_tri, t_tri, _ in result['trigram_alignments']
            }
            
            # Build the mapping in a single pass
            for s, t, uni_score in consolidated_unigrams:
                # Find matching trigrams efficiently
                matching_trigrams = [
                    (s_tri, t_tri) for (s_tri, t_tri), (s_tokens, t_tokens) in trigram_tokens.items()
                    if s in s_tokens and t in t_tokens
                ]
                
                # Update mappings
                for s_tri, t_tri in matching_trigrams:
                    trigram_to_unigrams[(s_tri, t_tri)].append((s, t, uni_score))
                    unigram_to_trigrams[(s, t)].add((s_tri, t_tri))
            
            # Sort trigram alignments by score for better visualization
            sorted_trigrams = sorted(
                trigram_to_unigrams.items(),
                key=lambda x: np.mean([score for _, _, score in x[1]]),
                reverse=True
            )
            
            # Add rows for each trigram alignment
            for (s_tri, t_tri), unigrams in sorted_trigrams:
                # Format trigram alignment
                trigram_alignment = f"[bold]{s_tri}[/bold] -> [bold]{t_tri}[/bold]"
                
                # Sort unigrams by score for consistent display
                sorted_unigrams = sorted(unigrams, key=lambda x: x[2], reverse=True)
                
                # Format unigram alignments with colors
                unigram_alignment_strs = [
                    f"[{colors[i % len(colors)]}]{s_uni}[/{colors[i % len(colors)]}] -> "
                    f"[{colors[i % len(colors)]}]{t_uni}[/{colors[i % len(colors)]}] "
                    f"({score:.2f})"
                    for i, (s_uni, t_uni, score) in enumerate(sorted_unigrams)
                ]
                unigram_alignments = "\n".join(unigram_alignment_strs)
                
                # Format gloss predictions for the tokens in this trigram
                gloss_strs = []
                for s_token in s_tri.split():
                    if s_token in result['gloss_predictions']:
                        gloss_strs.append(f"{s_token} → {result['gloss_predictions'][s_token]}")
                gloss_text = "\n".join(gloss_strs) if gloss_strs else ""
                
                # Add row to the table
                verse_table.add_row(trigram_alignment, unigram_alignments, gloss_text)
            
            # If there are no trigram alignments but we have unigram alignments, add them
            if not sorted_trigrams and consolidated_unigrams:
                unigram_strs = [
                    f"[{colors[i % len(colors)]}]{s}[/{colors[i % len(colors)]}] -> "
                    f"[{colors[i % len(colors)]}]{t}[/{colors[i % len(colors)]}] "
                    f"({score:.2f})"
                    for i, (s, t, score) in enumerate(consolidated_unigrams)
                ]
                # Format gloss predictions
                gloss_strs = [
                    f"{s_token} → {gloss}"
                    for s_token, gloss in result['gloss_predictions'].items()
                ]
                verse_table.add_row("", "\n".join(unigram_strs), "\n".join(gloss_strs))
            
            # Print the verse table and text
            console.print(verse_table)
            console.print(f"[bold]Source:[/bold] {pair.source}", style="white")
            console.print(f"[bold]Target:[/bold] {pair.target}", style="white")
            
            # Color-coded source-target best unigrams display
            if consolidated_unigrams:
                console.print("\n[bold]Best Unigram Alignments:[/bold]")
                for src, tgt, score in consolidated_unigrams:
                    console.print(f"[{colors[0]}]{src}[/{colors[0]}] -> [{colors[0]}]{tgt}[/{colors[0]}] ({score:.2f})")
            
            # Display all gloss predictions
            if result['gloss_predictions']:
                console.print("\n[bold]All Gloss Predictions:[/bold]")
                for src, gloss in result['gloss_predictions'].items():
                    console.print(f"{src} → {gloss}")
            
            console.print("\n" + "=" * 80 + "\n")

def load_jsonl(file_path: str) -> List[TranslationPair]:
    """Load translation pairs from JSONL file."""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pair = TranslationPair(
                source=data['original'],
                target=data['translation'],
                id=data['id'],
                translation_time=data['translation_time'],
                model=data['model']
            )
            pairs.append(pair)
    return pairs

def analyze_translations(pairs: List[TranslationPair], progress: Progress, task_id: int) -> List[Dict]:
    """Analyze a list of translation pairs with parallel processing and progress reporting."""
    analyzer = IntegratedAnalyzer()
    
    # Train the alignment model (this needs to be done before parallelization)
    analyzer.train(pairs)
    
    # Create a wrapper function to update progress
    processed_count = 0
    lock = threading.Lock()
    
    def analyze_with_progress(pair: TranslationPair) -> Dict:
        nonlocal processed_count
        result = analyzer.analyze_translation_pair(pair)
        
        # Thread-safe progress update
        with lock:
            nonlocal processed_count
            processed_count += 1
            progress.update(task_id, advance=1, 
                          description=f"Analyzing translations... ({processed_count}/{len(pairs)})")
        
        return result
    
    # Use ThreadPoolExecutor for parallel processing
    # Number of workers is set to CPU count or pairs length, whichever is smaller
    n_workers = min(len(pairs), os.cpu_count() or 4)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all pairs for analysis
        future_to_pair = {executor.submit(analyze_with_progress, pair): pair for pair in pairs}
        
        # Collect results while maintaining order
        results = []
        for future in concurrent.futures.as_completed(future_to_pair):
            try:
                result = future.result()
                # Store result with original pair index to maintain order
                pair = future_to_pair[future]
                pair_index = pairs.index(pair)
                results.append((pair_index, result))
            except Exception as e:
                print(f"Error processing pair: {e}")
    
    # Sort results by original index to maintain order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

def main(jsonl_path: str):
    """Main function to process JSONL file with progress reporting."""
    console = Console()
    analyzer = IntegratedAnalyzer()  # Create analyzer instance
    
    with Progress() as progress:
        # Task for loading pairs
        task1 = progress.add_task("[cyan]Loading pairs...", total=1)
        pairs = load_jsonl(jsonl_path)
        progress.update(task1, completed=1)
        
        # Task for analyzing translations
        task2 = progress.add_task("[green]Analyzing translations...", total=len(pairs))
        results = analyze_translations(pairs, progress, task2)
    
    # Call display_results as a method of the analyzer instance
    analyzer.display_results(results, pairs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze translations with integrated alignment")
    parser.add_argument("jsonl_path", type=str, help="Path to JSONL file")
    args = parser.parse_args()
    main(args.jsonl_path)