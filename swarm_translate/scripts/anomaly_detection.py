import random
import zlib
import numpy as np
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.table import Table

@dataclass
class TranslationPair:
    source: str
    target: str
    id: str
    translation_time: float
    model: str

def comp_size(text: str) -> int:
    """Calculate compressed size of text."""
    return len(zlib.compress(text.encode()))

def dist_metric(s1: str, s2: str) -> float:
    """Calculate distance metric between two strings."""
    return abs(len(s1) - len(s2)) + abs(comp_size(s1) - comp_size(s2))

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

def analyze_pairs(pairs: List[TranslationPair], noise_len: int = 50, trials: int = 5) -> List[float]:
    """Analyze translation pairs with noise injection."""
    results = []
    for pair in pairs:
        base = dist_metric(pair.source, pair.target)
        shift_sum = 0
        for _ in range(trials):
            noise = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=noise_len))
            shift_sum += abs(dist_metric(pair.source + noise, pair.target + noise) - base)
        results.append(shift_sum / trials)
    return results

def interpret_results(pairs: List[TranslationPair], scores: List[float], std_multiplier: float = 2.0) -> List[Dict]:
    """Interpret analysis results and identify anomalies using dynamic cutoff."""
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    dynamic_cutoff = mean_score + std_multiplier * std_score
    anomalies = []
    
    for pair, score in zip(pairs, scores):
        is_anomaly = score > dynamic_cutoff
        anomalies.append({
            'id': pair.id,
            'score': score,
            'is_anomaly': is_anomaly,
            'translation_time': pair.translation_time,
            'model': pair.model
        })
    return anomalies

def main(jsonl_path: str, noise_len: int = 50, trials: int = 5, std_multiplier: float = 2.0):
    """Main function to process JSONL file and analyze translations."""
    console = Console()
    
    # Load and analyze pairs
    pairs = load_jsonl(jsonl_path)
    scores = analyze_pairs(pairs, noise_len=noise_len, trials=trials)
    results = interpret_results(pairs, scores, std_multiplier=std_multiplier)
    
    # Calculate statistics
    total_anomalies = sum(result['is_anomaly'] for result in results)
    anomaly_percentage = (total_anomalies / len(results)) * 100
    average_score = np.mean(scores)
    std_deviation = np.std(scores)
    
    # Sort results by score in descending order and get top 10 anomalies
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_anomalies = sorted_results[:10]
    
    # Print summary statistics
    console.print(f"\nAnalyzed {len(pairs)} translation pairs:", style="bold")
    console.print(f"Average score: {average_score:.3f}", style="green")
    console.print(f"Standard deviation: {std_deviation:.3f}", style="green")
    console.print(f"Total anomalies: {total_anomalies} ({anomaly_percentage:.2f}%)", style="red")
    
    # Create a table for top anomalies
    table = Table(title="Top 10 Anomalous Results")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")
    table.add_column("Translation Time", style="yellow")
    table.add_column("Model", style="green")
    table.add_column("Source", style="white")
    table.add_column("Target", style="white")
    
    for result in top_anomalies:
        pair = next(pair for pair in pairs if pair.id == result['id'])
        table.add_row(
            f"{result['id']}",
            f"{result['score']:.3f}",
            f"{result['translation_time']:.2f}s",
            f"{result['model']}",
            pair.source,
            pair.target
        )
    
    console.print(table)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze translation pairs in JSONL file")
    parser.add_argument("jsonl_path", type=str, help="Path to JSONL file")
    parser.add_argument("--noise-len", type=int, default=50, help="Length of noise to inject")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per pair")
    parser.add_argument("--std-multiplier", type=float, default=2.0, help="Multiplier for standard deviation to determine dynamic cutoff")
    
    args = parser.parse_args()
    main(args.jsonl_path, args.noise_len, args.trials, args.std_multiplier)