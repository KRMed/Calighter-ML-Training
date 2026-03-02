import json
import random
import os
from typing import List, Dict, Tuple

def load_data(file_path: str) -> List[Dict]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_data(data: List[Dict], file_path: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def split_data(data: List[Dict], train_valid_ratio: float = 0.8, test_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    assert abs(train_valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed()  # or random.seed(42) for reproducibility
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total_size = len(shuffled_data)
    train_valid_size = int(total_size * train_valid_ratio)
    
    train_valid_data = shuffled_data[:train_valid_size]
    test_data = shuffled_data[train_valid_size:]
    
    return train_valid_data, test_data

def main():
    input_file = "eventkg_bio_sentences.json"
    output_dir = "data/split"
    
    print("Loading processed data...")
    data = load_data(input_file)
    total_entries = len(data)
    print(f"Total entries loaded: {total_entries}")
    
    print("\nSplitting data...")
    train_valid_data, test_data = split_data(data)
    
    print("\n" + "="*50)
    print("DATA SPLIT SUMMARY")
    print("="*50)
    print(f"Total entries: {total_entries}")
    print(f"Training + Validation set: {len(train_valid_data)} entries ({len(train_valid_data)/total_entries*100:.1f}%)")
    print(f"Test set: {len(test_data)} entries ({len(test_data)/total_entries*100:.1f}%)")
    print("="*50)
    
    save_data(train_valid_data, os.path.join(output_dir, "train_validation.json"))
    save_data(test_data, os.path.join(output_dir, "test.json"))
    
    print("\nFinished splitting data")

if __name__ == "__main__":
    main()
