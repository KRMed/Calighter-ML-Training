#!/usr/bin/env python3
"""
BIO Format Error Report Generator and Auto-Fixer

This script creates a detailed text file with error locations by line number,
and automatically fixes common errors in the BIO format JSON file.
"""

import json
import shutil
import re
from typing import List, Dict, Tuple

def load_bio_data_with_line_numbers(file_path: str) -> Tuple[List[Dict], List[int]]:
    """Load BIO format data and track line numbers for each entry."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse JSON
        data = json.loads(content)
        
        # Find line numbers for each entry
        lines = content.split('\n')
        entry_line_numbers = []
        
        brace_count = 0
        in_entry = False
        current_entry_line = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Look for start of entry (opening brace after array start or comma)
            if stripped == '{' and not in_entry:
                in_entry = True
                current_entry_line = i
                brace_count = 1
            elif in_entry:
                # Count braces to find end of entry
                brace_count += stripped.count('{') - stripped.count('}')
                if brace_count == 0:
                    entry_line_numbers.append(current_entry_line)
                    in_entry = False
        
        return data, entry_line_numbers
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return [], []

def find_all_errors_with_lines(data: List[Dict], line_numbers: List[int]) -> List[Tuple[str, int, int, str]]:
    """Find all errors and return with line numbers. Returns (error_type, entry_idx, line_num, description)."""
    errors = []
    
    # Track seen entries for duplicates
    seen_entries = {}
    
    for i, entry in enumerate(data):
        line_num = line_numbers[i] if i < len(line_numbers) else 0
        
        # Check for exact duplicates
        tokens_str = " ".join(entry.get("tokens", []))
        tags_str = " ".join(entry.get("tags", []))
        entry_key = (tokens_str, tags_str)
        
        if entry_key in seen_entries:
            orig_entry, orig_line = seen_entries[entry_key]
            errors.append(("EXACT_DUPLICATE", i, line_num, 
                          f"Duplicate of entry {orig_entry} (line {orig_line}). Text: {tokens_str[:100]}{'...' if len(tokens_str) > 100 else ''}"))
        else:
            seen_entries[entry_key] = (i, line_num)
        
        # Check JSON structure
        if not isinstance(entry, dict):
            errors.append(("STRUCTURE_ERROR", i, line_num, "Entry is not a dictionary"))
            continue
        
        tokens = entry.get("tokens", [])
        tags = entry.get("tags", [])
        
        if "tokens" not in entry:
            errors.append(("STRUCTURE_ERROR", i, line_num, "Missing 'tokens' field"))
            continue
        if "tags" not in entry:
            errors.append(("STRUCTURE_ERROR", i, line_num, "Missing 'tags' field"))
            continue
        if not isinstance(tokens, list):
            errors.append(("STRUCTURE_ERROR", i, line_num, "'tokens' is not a list"))
            continue
        if not isinstance(tags, list):
            errors.append(("STRUCTURE_ERROR", i, line_num, "'tags' is not a list"))
            continue
        
        # Check length mismatch
        if len(tokens) != len(tags):
            errors.append(("LENGTH_MISMATCH", i, line_num, 
                          f"Tokens: {len(tokens)}, Tags: {len(tags)}, Diff: {abs(len(tokens) - len(tags))}. Preview: {' '.join(tokens[:10])}{'...' if len(tokens) > 10 else ''}"))
            continue  # Skip other checks if length mismatch
        
        # Check for invalid tags
        valid_tags = {'O', 'B-EVENT', 'I-EVENT', 'B-TIME', 'I-TIME', 'B-LOCATION', 'I-LOCATION'}
        for j, tag in enumerate(tags):
            if tag not in valid_tags:
                errors.append(("INVALID_TAG", i, line_num, 
                              f"Invalid tag '{tag}' at position {j} (token: '{tokens[j] if j < len(tokens) else 'N/A'}')"))
        
        # Check for short entries
        if len(tokens) < 3:
            errors.append(("SHORT_ENTRY", i, line_num, 
                          f"Entry too short ({len(tokens)} tokens): {' '.join(tokens)}"))
    
    return errors

def find_similar_text_with_lines(data: List[Dict], line_numbers: List[int], threshold: float = 0.9) -> List[Tuple[str, int, int, str]]:
    """Find similar text entries with line numbers."""
    import difflib
    
    similar_errors = []
    
    for i in range(len(data)):
        text_i = " ".join(data[i].get("tokens", []))
        line_i = line_numbers[i] if i < len(line_numbers) else 0
        
        for j in range(i + 1, len(data)):
            text_j = " ".join(data[j].get("tokens", []))
            line_j = line_numbers[j] if j < len(line_numbers) else 0
            
            similarity = difflib.SequenceMatcher(None, text_i, text_j).ratio()
            
            if similarity >= threshold:
                similar_errors.append(("SIMILAR_TEXT", j, line_j, 
                                     f"Similar to entry {i} (line {line_i}) - {similarity:.1%} similarity. Text: {text_j[:100]}{'...' if len(text_j) > 100 else ''}"))
    
    return similar_errors

def normalize_time_format(time_str: str) -> str:
    """
    Normalize time formats to standard format.
    Examples: 6pm → 6:00 PM, 18:00 → 6:00 PM, noon → 12:00 PM
    """
    time_str = time_str.strip().lower()
    
    # Handle special cases
    if time_str in ['noon', '12:00pm', '12pm']:
        return '12:00 PM'
    elif time_str in ['midnight', '12:00am', '12am']:
        return '12:00 AM'
    
    # Handle 24-hour format (18:00 → 6:00 PM)
    if re.match(r'^\d{1,2}:\d{2}$', time_str):
        hour, minute = map(int, time_str.split(':'))
        if hour > 23 or minute > 59:  # Invalid time
            return time_str
        if hour == 0:
            return f"12:{minute:02d} AM"
        elif hour < 12:
            return f"{hour}:{minute:02d} AM"
        elif hour == 12:
            return f"12:{minute:02d} PM"
        else:
            return f"{hour-12}:{minute:02d} PM"
    
    # Handle formats like 6pm, 6:30pm, 6:30 pm, 6:30p.m.
    time_match = re.match(r'^(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?$', time_str)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        period = time_match.group(3).upper()
        
        # Validate hour and minute
        if hour > 12 or hour < 1 or minute > 59:
            return time_str
        
        # Convert 12-hour format
        if hour == 12:
            return f"12:{minute:02d} {period}M"
        else:
            return f"{hour}:{minute:02d} {period}M"
    
    # Handle formats like 6:30, 16:30 (assume 24-hour if > 12)
    time_match = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        
        if hour > 23 or minute > 59:  # Invalid time
            return time_str
        
        if hour > 12:
            return f"{hour-12}:{minute:02d} PM"
        elif hour == 12:
            return f"12:{minute:02d} PM"
        elif hour == 0:
            return f"12:{minute:02d} AM"
        else:
            return f"{hour}:{minute:02d} AM"
    
    # Handle single digit hours like 6, 16 (but be more careful)
    if re.match(r'^\d{1,2}$', time_str):
        hour = int(time_str)
        # Only convert if it's a reasonable hour (1-24)
        if hour > 24 or hour < 1:
            return time_str
        if hour > 12:
            return f"{hour-12}:00 PM"
        elif hour == 12:
            return "12:00 PM"
        else:
            return f"{hour}:00 AM"
    
    # Return original if no pattern matches
    return time_str

def normalize_punctuation(text: str) -> str:
    """
    Normalize punctuation marks to standard ASCII characters.
    """
    # Replace smart quotes
    text = text.replace('"', '"').replace('"', '"')  # Smart double quotes
    text = text.replace(''', "'").replace(''', "'")  # Smart single quotes/apostrophes
    
    # Replace long dashes
    text = text.replace('—', '-').replace('–', '-')  # Em dash and en dash
    
    # Replace other common unicode punctuation
    text = text.replace('…', '...')  # Ellipsis
    text = text.replace('«', '"').replace('»', '"')  # Guillemets
    
    return text

def normalize_tokens(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Normalize tokens for time formats and punctuation.
    Returns (normalized_tokens, list_of_changes)
    """
    normalized_tokens = []
    changes = []
    
    for i, token in enumerate(tokens):
        original_token = token
        
        # Normalize punctuation first
        token = normalize_punctuation(token)
        
        # Check if token looks like a time - be more conservative
        # Only normalize if it clearly looks like a time format
        time_patterns = [
            r'^\d{1,2}[ap]\.?m\.?$',  # 6pm, 6am, 6p.m.
            r'^\d{1,2}:\d{2}[ap]\.?m\.?$',  # 6:30pm, 6:30am
            r'^\d{1,2}:\d{2}$',  # 6:30, 18:30
            r'^(noon|midnight)$'  # noon, midnight
        ]
        
        is_time_token = any(re.match(pattern, token.lower()) for pattern in time_patterns)
        
        if is_time_token:
            normalized_time = normalize_time_format(token)
            if normalized_time != token:
                changes.append(f"Time format: '{original_token}' → '{normalized_time}'")
                token = normalized_time
        
        # Check for punctuation changes
        if normalize_punctuation(original_token) != original_token:
            changes.append(f"Punctuation: '{original_token}' → '{token}'")
        
        normalized_tokens.append(token)
    
    return normalized_tokens, changes

def auto_fix_errors(data: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Automatically fix common errors in the BIO format data.
    Returns (fixed_data, list_of_fixes_applied)
    """
    fixed_data = []
    fixes_applied = []
    
    # Track seen entries for duplicate removal with their original indices
    seen_entries = {}  # entry_key -> (original_index, entry_data)
    
    for i, entry in enumerate(data):
        # Skip if entry is not a dictionary
        if not isinstance(entry, dict):
            fixes_applied.append(f"Entry {i}: Removed invalid entry (not a dictionary)")
            continue
            
        # Skip if missing required fields
        if "tokens" not in entry or "tags" not in entry:
            fixes_applied.append(f"Entry {i}: Removed entry with missing tokens/tags fields")
            continue
            
        tokens = entry.get("tokens", [])
        tags = entry.get("tags", [])
        
        # Skip if tokens or tags are not lists
        if not isinstance(tokens, list) or not isinstance(tags, list):
            fixes_applied.append(f"Entry {i}: Removed entry with invalid tokens/tags format")
            continue
        
        # Normalize tokens (time formats and punctuation)
        normalized_tokens, normalization_changes = normalize_tokens(tokens)
        for change in normalization_changes:
            fixes_applied.append(f"Entry {i}: {change}")
        
        # Use normalized tokens for duplicate checking and further processing
        tokens = normalized_tokens
        
        # Check for exact duplicates - keep the first occurrence, remove later ones
        tokens_str = " ".join(tokens)
        tags_str = " ".join(tags)
        entry_key = (tokens_str, tags_str)
        
        if entry_key in seen_entries:
            earlier_index, _ = seen_entries[entry_key]
            fixes_applied.append(f"Entry {i}: Automatically removed exact duplicate of entry {earlier_index} - Text: {tokens_str[:100]}{'...' if len(tokens_str) > 100 else ''}")
            continue
        
        # Store this entry as the first occurrence
        seen_entries[entry_key] = (i, entry)
        
        # Skip very short entries
        if len(tokens) < 3:
            fixes_applied.append(f"Entry {i}: Removed short entry ({len(tokens)} tokens): {' '.join(tokens)}")
            continue
        
        # Fix length mismatches by adjusting tags
        if len(tokens) != len(tags):
            original_tags_len = len(tags)
            
            if len(tokens) > len(tags):
                # Add 'O' tags for missing tokens
                tags.extend(['O'] * (len(tokens) - len(tags)))
                fixes_applied.append(f"Entry {i}: Added {len(tokens) - original_tags_len} 'O' tags to match token count")
            else:
                # Remove excess tags
                tags = tags[:len(tokens)]
                fixes_applied.append(f"Entry {i}: Removed {original_tags_len - len(tokens)} excess tags")
        
        # Fix BIO sequence errors
        fixed_tags = []
        prev_tag = None
        
        for j, tag in enumerate(tags):
            if tag.startswith('I-'):
                entity_type = tag[2:]  # Remove 'I-' prefix
                expected_b_tag = f'B-{entity_type}'
                expected_i_tag = f'I-{entity_type}'
                
                # I- tag should be preceded by B- or I- of same type
                if prev_tag != expected_b_tag and prev_tag != expected_i_tag:
                    # Change I- to B- to start a new entity
                    fixed_tag = f'B-{entity_type}'
                    fixed_tags.append(fixed_tag)
                    fixes_applied.append(f"Entry {i}: Changed '{tag}' to '{fixed_tag}' at position {j} (token: '{tokens[j]}')")
                else:
                    fixed_tags.append(tag)
            else:
                fixed_tags.append(tag)
            
            prev_tag = fixed_tags[-1]
        
        # Create fixed entry
        fixed_entry = {
            "tokens": tokens,
            "tags": fixed_tags
        }
        
        fixed_data.append(fixed_entry)
    
    return fixed_data, fixes_applied

def write_error_report(errors: List[Tuple[str, int, int, str]], fixes: List[str], output_file: str, data: List[Dict] = None):
    """Write detailed error report to text file with enhanced analytics."""
    
    # Sort errors by line number
    errors.sort(key=lambda x: x[2])
    
    # Group errors by type
    error_groups = {}
    for error_type, entry_idx, line_num, description in errors:
        if error_type not in error_groups:
            error_groups[error_type] = []
        error_groups[error_type].append((entry_idx, line_num, description))
    
    # Analyze label usage if data is provided
    label_stats = None
    entity_stats = None
    recommendations = []
    
    if data:
        label_stats = analyze_label_usage(data)
        entity_stats = analyze_entity_coverage(data)
        recommendations = generate_label_recommendations(label_stats, entity_stats)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BIO FORMAT ERROR REPORT AND AUTO-FIX SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total errors found: {len(errors)}\n")
        f.write(f"Error types: {len(error_groups)}\n")
        f.write(f"Auto-fixes applied: {len(fixes)}\n")
        if data:
            f.write(f"Total entries analyzed: {len(data)}\n")
        f.write("\n")
        
        # Label usage statistics
        if label_stats:
            f.write("LABEL USAGE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            sorted_labels = sorted(label_stats.items(), key=lambda x: x[1], reverse=True)
            total_tags = sum(label_stats.values())
            
            for label, count in sorted_labels:
                percentage = (count / total_tags) * 100 if total_tags > 0 else 0
                f.write(f"{label:15s}: {count:6d} ({percentage:5.1f}%)\n")
            f.write(f"{'TOTAL':15s}: {total_tags:6d} (100.0%)\n\n")
        
        # Entity coverage analysis
        if entity_stats:
            f.write("ENTITY COVERAGE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            sorted_entities = sorted(entity_stats.items(), key=lambda x: x[1]['total'], reverse=True)
            
            for entity, stats in sorted_entities:
                b_count = stats['B']
                i_count = stats['I']
                total = stats['total']
                avg_length = (total / b_count) if b_count > 0 else 0
                
                f.write(f"{entity:12s}: {total:4d} total ({b_count:3d} B-, {i_count:3d} I-) "
                       f"Avg length: {avg_length:.1f}\n")
            f.write("\n")
        
        # Recommendations
        if recommendations:
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            for rec in recommendations:
                f.write(f"{rec}\n")
            f.write("\n")
        
        # Auto-fix summary
        if fixes:
            f.write("AUTO-FIXES APPLIED:\n")
            f.write("-" * 40 + "\n")
            for fix in fixes:
                f.write(f"✅ {fix}\n")
            f.write("\n")
        
        # Summary by error type
        f.write("ERROR SUMMARY BY TYPE (BEFORE AUTO-FIX):\n")
        f.write("-" * 40 + "\n")
        for error_type, error_list in error_groups.items():
            f.write(f"{error_type}: {len(error_list)} errors\n")
        f.write("\n")
        
        # Detailed error listing
        f.write("DETAILED ERROR LISTING (BEFORE AUTO-FIX):\n")
        f.write("=" * 80 + "\n")
        
        for error_type, entry_idx, line_num, description in errors:
            f.write(f"Line {line_num:4d} | {error_type:20s} | Entry {entry_idx:3d} | {description}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("WHAT WAS AUTOMATICALLY FIXED:\n")
        f.write("=" * 80 + "\n")
        f.write("1. ✅ Normalized time formats (6pm → 6:00 PM, 18:00 → 6:00 PM, noon → 12:00 PM)\n")
        f.write("2. ✅ Normalized punctuation (smart quotes → ASCII quotes, em dashes → hyphens)\n")
        f.write("3. ✅ Removed exact duplicate entries\n")
        f.write("4. ✅ Fixed length mismatches (added 'O' tags or removed excess tags)\n")
        f.write("5. ✅ Fixed BIO sequence errors (changed I-TAG to B-TAG when needed)\n")
        f.write("6. ✅ Removed invalid entries (non-dict, missing fields, too short)\n")
        f.write("7. ✅ Created backup of original file\n")
        f.write("8. ✅ Enhanced BIO sequence validation\n")
        f.write("\nThe fixed data has been saved to the original file.\n")
        f.write("The original file has been backed up with .backup extension.\n")

def analyze_label_usage(data: List[Dict]) -> Dict[str, int]:
    """Analyze usage of each label type."""
    label_counts = {}
    
    for entry in data:
        tags = entry.get("tags", [])
        for tag in tags:
            if tag in label_counts:
                label_counts[tag] += 1
            else:
                label_counts[tag] = 1
    
    return label_counts

def analyze_entity_coverage(data: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Analyze entity coverage (B- and I- tags) for each entity type."""
    entity_stats = {}
    
    for entry in data:
        tags = entry.get("tags", [])
        for tag in tags:
            if tag.startswith(('B-', 'I-')):
                entity_type = tag[2:]  # Remove B- or I- prefix
                tag_type = tag[0]      # B or I
                
                if entity_type not in entity_stats:
                    entity_stats[entity_type] = {'B': 0, 'I': 0, 'total': 0}
                
                entity_stats[entity_type][tag_type] += 1
                entity_stats[entity_type]['total'] += 1
    
    return entity_stats

def validate_bio_sequences_comprehensive(data: List[Dict], line_numbers: List[int]) -> List[Tuple[str, int, int, str]]:
    """Comprehensive BIO sequence validation with detailed error reporting."""
    errors = []
    
    for i, entry in enumerate(data):
        line_num = line_numbers[i] if i < len(line_numbers) else 0
        tokens = entry.get("tokens", [])
        tags = entry.get("tags", [])
        
        if len(tokens) != len(tags):
            continue  # Skip if length mismatch (handled elsewhere)
        
        # Track entity sequences
        entity_sequences = {}  # entity_type -> [positions]
        
        for j, tag in enumerate(tags):
            if tag.startswith('I-'):
                entity_type = tag[2:]
                expected_b_tag = f'B-{entity_type}'
                expected_i_tag = f'I-{entity_type}'
                
                # Check if this I- tag is properly preceded
                if j == 0:
                    # I- tag at position 0 is always wrong
                    context = " ".join(tokens[:min(3, len(tokens))])
                    errors.append(("BIO_SEQUENCE_ERROR", i, line_num, 
                                  f"'{tag}' at position {j} (token: '{tokens[j]}') cannot be at start of sequence. Context: {context}"))
                else:
                    prev_tag = tags[j-1]
                    if prev_tag != expected_b_tag and prev_tag != expected_i_tag:
                        context_start = max(0, j-2)
                        context_end = min(len(tokens), j+3)
                        context = " ".join(tokens[context_start:context_end])
                        errors.append(("BIO_SEQUENCE_ERROR", i, line_num, 
                                      f"'{tag}' at position {j} (token: '{tokens[j]}') not preceded by '{expected_b_tag}' or '{expected_i_tag}'. Previous tag: '{prev_tag}'. Context: {context}"))
            
            elif tag.startswith('B-'):
                entity_type = tag[2:]
                if entity_type not in entity_sequences:
                    entity_sequences[entity_type] = []
                entity_sequences[entity_type].append(j)
        
        # Check for orphaned B- tags (B- tags with no following I- tags when they should have them)
        for entity_type, positions in entity_sequences.items():
            for pos in positions:
                if pos < len(tags) - 1:  # Not the last position
                    next_tag = tags[pos + 1]
                    expected_i_tag = f'I-{entity_type}'
                    
                    # If the next tag is not I- of the same type and the entity name suggests it might be multi-token
                    if next_tag != expected_i_tag and len(tokens[pos].split()) == 1:
                        # Check if this might be a multi-token entity that should have I- tags
                        context_start = max(0, pos-1)
                        context_end = min(len(tokens), pos+3)
                        context = " ".join(tokens[context_start:context_end])
                        # This is more of a warning than an error, so we won't add it as an error
                        # but we could add statistics about it
    
    return errors

def generate_label_recommendations(label_counts: Dict[str, int], entity_stats: Dict[str, Dict[str, int]]) -> List[str]:
    """Generate recommendations for improving label balance."""
    recommendations = []
    
    total_tokens = sum(label_counts.values())
    
    # Analyze O vs entity tags ratio
    o_count = label_counts.get('O', 0)
    entity_count = total_tokens - o_count
    
    if entity_count > 0:
        o_ratio = o_count / total_tokens
        entity_ratio = entity_count / total_tokens
        
        recommendations.append(f"Label Distribution: {o_ratio:.1%} O-tags, {entity_ratio:.1%} Entity tags")
        
        if entity_ratio < 0.1:
            recommendations.append("⚠️  Very few entity tags (<10%) - consider adding more labeled examples")
        elif entity_ratio > 0.5:
            recommendations.append("⚠️  Very high entity density (>50%) - ensure enough context tokens")
    
    # Analyze individual entity types
    entity_totals = [(entity, stats['total']) for entity, stats in entity_stats.items()]
    entity_totals.sort(key=lambda x: x[1], reverse=True)
    
    if entity_totals:
        recommendations.append(f"Most common entity: {entity_totals[0][0]} ({entity_totals[0][1]} instances)")
        recommendations.append(f"Least common entity: {entity_totals[-1][0]} ({entity_totals[-1][1]} instances)")
        
        # Check for imbalanced entities
        if len(entity_totals) > 1:
            ratio = entity_totals[0][1] / entity_totals[-1][1] if entity_totals[-1][1] > 0 else float('inf')
            if ratio > 10:
                recommendations.append(f"⚠️  High entity imbalance: {entity_totals[0][0]} has {ratio:.1f}x more examples than {entity_totals[-1][0]}")
        
        # Check for entities with very few examples
        for entity, count in entity_totals:
            if count < 10:
                recommendations.append(f"⚠️  {entity} has only {count} examples - consider adding more")
    
    return recommendations

def main():
    file_path = "bio_format.json"
    output_file = "bio_format_errors.txt"
    backup_file = "bio_format.json.backup"
    
    print("Loading BIO format data and tracking line numbers...")
    data, line_numbers = load_bio_data_with_line_numbers(file_path)
    
    if not data:
        print("Failed to load data or file is empty")
        return
    
    print(f"Loaded {len(data)} entries")
    print("Finding all errors...")
    
    # Find all errors in original data
    errors = find_all_errors_with_lines(data, line_numbers)
    
    # Add comprehensive BIO validation errors
    bio_errors = validate_bio_sequences_comprehensive(data, line_numbers)
    errors.extend(bio_errors)
    
    # Find similar text
    print("Finding similar text...")
    similar_errors = find_similar_text_with_lines(data, line_numbers)
    
    # Combine all errors
    all_errors = errors + similar_errors
    
    print(f"Found {len(all_errors)} total errors")
    
    # Create backup of original file
    print(f"Creating backup: {backup_file}")
    shutil.copy2(file_path, backup_file)
    
    # Auto-fix the errors
    print("Auto-fixing errors...")
    fixed_data, fixes_applied = auto_fix_errors(data)
    
    # Save fixed data
    print(f"Saving fixed data to {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    # Write report
    print(f"Writing detailed report to {output_file}...")
    write_error_report(all_errors, fixes_applied, output_file, data)
    
    print(f"✅ Auto-fix complete!")
    print(f"   Original entries: {len(data)}")
    print(f"   Fixed entries: {len(fixed_data)}")
    print(f"   Errors found: {len(all_errors)}")
    print(f"   Fixes applied: {len(fixes_applied)}")
    print(f"   Backup saved: {backup_file}")
    print(f"   Report saved: {output_file}")
    
    # Validate the fixed data
    print("\nValidating fixed data...")
    try:
        fixed_data_reload, fixed_line_numbers = load_bio_data_with_line_numbers(file_path)
        remaining_errors = find_all_errors_with_lines(fixed_data_reload, fixed_line_numbers)
        remaining_similar = find_similar_text_with_lines(fixed_data_reload, fixed_line_numbers)
        total_remaining = len(remaining_errors) + len(remaining_similar)
        
        if total_remaining == 0:
            print("🎉 All errors have been fixed!")
        else:
            print(f"⚠️  {total_remaining} errors remain (may need manual review)")
    except Exception as e:
        print(f"Could not validate fixed data: {e}")

if __name__ == "__main__":
    main()
