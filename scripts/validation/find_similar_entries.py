#!/usr/bin/env python3
"""
Find and display specific similar entries for manual review
"""

import json
import difflib

def load_bio_data(file_path: str):
    """Load BIO format data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def find_and_display_similar_entries(data):
    """Find similar entries and display them for comparison."""
    
    similar_pairs = []
    
    for i in range(len(data)):
        text_i = " ".join(data[i].get("tokens", []))
        
        for j in range(i + 1, len(data)):
            text_j = " ".join(data[j].get("tokens", []))
            
            similarity = difflib.SequenceMatcher(None, text_i, text_j).ratio()
            
            if similarity >= 0.9:  # 90% or higher similarity
                similar_pairs.append((i, j, text_i, text_j, similarity))
    
    print(f"Found {len(similar_pairs)} similar pairs:")
    print("=" * 80)
    
    for i, j, text1, text2, similarity in similar_pairs:
        print(f"\nSIMILAR PAIR {i+1}: Entries {i} and {j} ({similarity:.1%} similarity)")
        print("-" * 60)
        print(f"Entry {i}:")
        print(f"  Tokens: {data[i]['tokens']}")
        print(f"  Tags:   {data[i]['tags']}")
        print(f"  Text:   {text1}")
        print(f"\nEntry {j}:")
        print(f"  Tokens: {data[j]['tokens']}")
        print(f"  Tags:   {data[j]['tags']}")
        print(f"  Text:   {text2}")
        
        if similarity == 1.0:
            print(f"\n🔴 RECOMMENDATION: DELETE Entry {j} (100% identical)")
        elif similarity >= 0.95:
            print(f"\n🟡 RECOMMENDATION: Review carefully - very similar ({similarity:.1%})")
        else:
            print(f"\n🟢 RECOMMENDATION: Keep both - different enough ({similarity:.1%})")
        print("=" * 80)

def main():
    file_path = "eventkg_bio_sentences.json"
    
    print("Loading BIO format data...")
    data = load_bio_data(file_path)
    
    if not data:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(data)} entries")
    find_and_display_similar_entries(data)

if __name__ == "__main__":
    main()
