# debug_entities_sampling.py

def debug_entity_sampling():
    event_uris = set()
    location_uris = set()

    # Step 1: Load 500 event URIs with English labels from events.nq
    print("Step 1: Reading events.nq for 500 event URIs...")
    with open('data/raw/events.nq', 'r', encoding='utf-8') as f, open('data/raw/events_sample.nq', 'w', encoding='utf-8') as out:
        for line in f:
            if '@en' in line and '<http://www.w3.org/2000/01/rdf-schema#label>' in line:
                uri = line.split(' ')[0].strip('<>')
                event_uris.add(uri)
                out.write(line)
            if len(event_uris) >= 1000:
                break
    print(f"Collected {len(event_uris)} event URIs.\n")

    # Step 2: Collect relations and location URIs from relations_events_base.nq
    print("Step 2: Collecting relations and location URIs...")
    location_uris = set()
    with open('data/raw/relations_events_base.nq', 'r', encoding='utf-8') as f, open('data/raw/relations_sample.nq', 'w', encoding='utf-8') as out:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            subj = parts[0].strip('<>')
            predicate = parts[1]
            obj = parts[2].strip('<>"')

            if subj in event_uris:
                if predicate == '<http://semanticweb.cs.vu.nl/2009/11/sem/hasPlace>':
                    location_uris.add(obj)
                    out.write(line)
                elif predicate == '<http://semanticweb.cs.vu.nl/2009/11/sem/hasBeginTimeStamp>':
                    out.write(line)


    # Show some location URIs sample
    print("Sample location URIs:")
    for i, loc in enumerate(list(location_uris)[:5]):
        print(f"  {i+1}. {loc}")
    print()

    # Normalize location URIs for matching
    location_uris_normalized = set(uri.rstrip('/').lower() for uri in location_uris)

    # Step 3: Extract matching entities with English labels
    print("Step 3: Extracting matching entities with English labels from entities.nq...")
    matched_uris = set()
    total_lines_checked = 0
    matched_lines = 0
    with open('data/raw/entities.nq', 'r', encoding='utf-8') as f, open('data/raw/entities_sample.nq', 'w', encoding='utf-8') as out:
        for line in f:
            total_lines_checked += 1
            subj = line.split(' ')[0].strip('<>').rstrip('/').lower()

            if subj in location_uris_normalized:
                if '<http://www.w3.org/2000/01/rdf-schema#label>' in line:
                    matched_lines += 1
                    out.write(line)
                    matched_uris.add(subj)

                    print(f"Matched entity ({matched_lines}): {subj}")
                else:
                    # For debugging, print lines with subject but no label predicate
                    print(f"URI matched but no label predicate: {subj}")
            
            # Optional: stop early if all matched URIs found
            if len(matched_uris) >= len(location_uris_normalized):
                print("All matched URIs found, stopping early.")
                break

    print(f"\nTotal lines checked in entities.nq: {total_lines_checked}")
    print(f"Total matched entity labels found: {matched_lines}")
    print(f"Unique matched URIs: {len(matched_uris)}")

    if matched_lines == 0:
        print("\nNo matching entity labels found. Possible reasons:")
        print("- URI formatting mismatch between relations and entities files")
        print("- Label predicate or language tag might be different or missing")
        print("- Location URIs from relations might not exist in entities.nq")

if __name__ == "__main__":
    debug_entity_sampling()
