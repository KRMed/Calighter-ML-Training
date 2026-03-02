#!/usr/bin/env python3
"""
BIO Format Data Validation Script

This script validates BIO format JSON data for:
1. Duplicate entries (exact matches and similar text)
2. Data consistency (tokens/tags length matching)
3. Valid BIO tag format
4. Proper BIO sequence rules
"""

import json
from rapidfuzz import fuzz
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set

def load_bio_data(file_path: str) -> List[Dict]:
    """Load BIO format data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def find_exact_duplicates(data: List[Dict]) -> List[Tuple[int, int, str]]:
    """Find exact duplicate entries."""
    duplicates = []
    seen = {}
    
    for i, entry in enumerate(data):
        tokens_str = " ".join(entry["tokens"])
        tags_str = " ".join(entry["tags"])
        entry_key = (tokens_str, tags_str)
        
        if entry_key in seen:
            duplicates.append((seen[entry_key], i, tokens_str))
        else:
            seen[entry_key] = i
    
    return duplicates

def find_similar_text(data: List[Dict], similarity_threshold: float = 0.9) -> List[Tuple[int, int, str, str, float]]:
    """Find entries with similar text content."""
    similar_pairs = []
    
    for i in tqdm(range(len(data)), desc="Scanning for similar text"):
        text_i = " ".join(data[i]["tokens"])
        
        for j in range(i + 1, len(data)):
            text_j = " ".join(data[j]["tokens"])
            
            # Calculate similarity ratio
            similarity = fuzz.ratio(text_i, text_j) / 100.0
            
            if similarity >= similarity_threshold:
                similar_pairs.append((i, j, text_i, text_j, similarity))
    
    return similar_pairs

def validate_bio_format(data: List[Dict]) -> List[Tuple[int, str, str]]:
    """Validate BIO format rules and consistency."""
    errors = []
    valid_tags = {'O', 'B-EVENT', 'I-EVENT', 'B-TIME', 'I-TIME', 'B-LOCATION', 'I-LOCATION'}
    
    for i, entry in enumerate(data):
        tokens = entry.get("tokens", [])
        tags = entry.get("tags", [])
        
        # Check if tokens and tags have same length
        if len(tokens) != len(tags):
            errors.append((i, "LENGTH_MISMATCH", f"Tokens: {len(tokens)}, Tags: {len(tags)}"))
            continue
        
        # Check for valid tag format
        for j, tag in enumerate(tags):
            if tag not in valid_tags:
                errors.append((i, "INVALID_TAG", f"Invalid tag '{tag}' at position {j}"))
        
        # Check BIO sequence rules
        prev_tag = None
        for j, tag in enumerate(tags):
            if tag.startswith('I-'):
                entity_type = tag[2:]  # Remove 'I-' prefix
                expected_b_tag = f'B-{entity_type}'
                expected_i_tag = f'I-{entity_type}'
                
                # I- tag should be preceded by B- or I- of same type
                if prev_tag != expected_b_tag and prev_tag != expected_i_tag:
                    errors.append((i, "BIO_SEQUENCE_ERROR", 
                                 f"I-{entity_type} at position {j} not preceded by B-{entity_type} or I-{entity_type}"))
            
            prev_tag = tag
    
    return errors

def analyze_tag_distribution(data: List[Dict]) -> Dict[str, int]:
    """Analyze distribution of tags."""
    tag_counts = Counter()
    
    for entry in data:
        for tag in entry.get("tags", []):
            tag_counts[tag] += 1
    
    return dict(tag_counts)

def find_empty_or_short_entries(data: List[Dict], min_length: int = 3) -> List[Tuple[int, int, str]]:
    """Find entries that are empty or too short."""
    short_entries = []
    
    for i, entry in enumerate(data):
        tokens = entry.get("tokens", [])
        if len(tokens) < min_length:
            text = " ".join(tokens) if tokens else "[EMPTY]"
            short_entries.append((i, len(tokens), text))
    
    return short_entries

def validate_json_structure(data: List[Dict]) -> List[Tuple[int, str]]:
    """Validate that each entry has required fields."""
    structure_errors = []
    
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            structure_errors.append((i, "Not a dictionary"))
            continue
            
        if "tokens" not in entry:
            structure_errors.append((i, "Missing 'tokens' field"))
        elif not isinstance(entry["tokens"], list):
            structure_errors.append((i, "'tokens' is not a list"))
        
        if "tags" not in entry:
            structure_errors.append((i, "Missing 'tags' field"))
        elif not isinstance(entry["tags"], list):
            structure_errors.append((i, "'tags' is not a list"))
    
    return structure_errors

def print_report(data: List[Dict], file_path: str):
    """Print comprehensive validation report."""
    print("=" * 80)
    print(f"BIO FORMAT VALIDATION REPORT")
    print(f"File: {file_path}")
    print(f"Total entries: {len(data)}")
    print("=" * 80)
    
    # 1. JSON Structure validation
    print("\n1. JSON STRUCTURE VALIDATION")
    print("-" * 40)
    structure_errors = validate_json_structure(data)
    if structure_errors:
        print(f"❌ Found {len(structure_errors)} structure errors:")
        for idx, error in structure_errors:
            print(f"  Entry {idx}: {error}")
    else:
        print("✅ All entries have valid JSON structure")
    
    # 2. Exact duplicates
    print("\n2. EXACT DUPLICATES")
    print("-" * 40)
    exact_duplicates = find_exact_duplicates(data)
    if exact_duplicates:
        print(f"❌ Found {len(exact_duplicates)} exact duplicate pairs:")
        for orig_idx, dup_idx, text in exact_duplicates:
            print(f"  Entries {orig_idx} and {dup_idx}:")
            print(f"    Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    else:
        print("✅ No exact duplicates found")
    
    # 3. Similar text (potential duplicates)
    print("\n3. SIMILAR TEXT (>90% similarity)")
    print("-" * 40)
    similar_pairs = find_similar_text(data, 0.9)
    if similar_pairs:
        print(f"⚠️  Found {len(similar_pairs)} similar text pairs:")
        for i, j, text1, text2, similarity in similar_pairs:
            print(f"  Entries {i} and {j} (similarity: {similarity:.2%}):")
            print(f"    Text 1: {text1[:80]}{'...' if len(text1) > 80 else ''}")
            print(f"    Text 2: {text2[:80]}{'...' if len(text2) > 80 else ''}")
    else:
        print("✅ No highly similar text pairs found")
    
    # 4. BIO format validation
    print("\n4. BIO FORMAT VALIDATION")
    print("-" * 40)
    bio_errors = validate_bio_format(data)
    if bio_errors:
        print(f"❌ Found {len(bio_errors)} BIO format errors:")
        error_types = defaultdict(list)
        for idx, error_type, details in bio_errors:
            error_types[error_type].append((idx, details))
        
        for error_type, errors in error_types.items():
            print(f"  {error_type}: {len(errors)} errors")
            for idx, details in errors[:5]:  # Show first 5 of each type
                print(f"    Entry {idx}: {details}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")
    else:
        print("✅ All entries follow valid BIO format")
    
    # 5. Short or empty entries
    print("\n5. SHORT OR EMPTY ENTRIES")
    print("-" * 40)
    short_entries = find_empty_or_short_entries(data)
    if short_entries:
        print(f"⚠️  Found {len(short_entries)} entries with <3 tokens:")
        for idx, length, text in short_entries:
            print(f"  Entry {idx} ({length} tokens): {text}")
    else:
        print("✅ No short or empty entries found")
    
    # 6. Tag distribution
    print("\n6. TAG DISTRIBUTION")
    print("-" * 40)
    tag_dist = analyze_tag_distribution(data)
    total_tags = sum(tag_dist.values())
    print(f"Total tags: {total_tags}")
    for tag, count in sorted(tag_dist.items()):
        percentage = (count / total_tags) * 100
        print(f"  {tag}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    issues = []
    if structure_errors:
        issues.append(f"{len(structure_errors)} structure errors")
    if exact_duplicates:
        issues.append(f"{len(exact_duplicates)} exact duplicates")
    if similar_pairs:
        issues.append(f"{len(similar_pairs)} similar text pairs")
    if bio_errors:
        issues.append(f"{len(bio_errors)} BIO format errors")
    if short_entries:
        issues.append(f"{len(short_entries)} short entries")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No issues found - data looks good!")

def main():
    file_path = "eventkg_bio_sentences.json"
    
    print("Loading BIO format data...")
    data = load_bio_data(file_path)
    
    if not data:
        print("Failed to load data or file is empty")
        return
    
    print_report(data, file_path)

if __name__ == "__main__":
    main()
