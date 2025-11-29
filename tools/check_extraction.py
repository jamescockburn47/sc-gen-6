"""Analyze extraction quality."""

import json
from pathlib import Path


def main():
    # Analyze all pending files
    pending_files = list(Path('data/graph/pending').glob('*.json'))
    
    print(f"Analyzing {len(pending_files)} pending files...\n")
    
    high_party_docs = []
    total_parties = 0
    total_events = 0
    
    for f in pending_files:
        data = json.loads(f.read_text())
        parties = [n for n in data['nodes'] if n['node_type'] == 'party']
        events = [n for n in data['nodes'] if n['node_type'] == 'event']
        
        total_parties += len(parties)
        total_events += len(events)
        
        if len(parties) > 5:
            high_party_docs.append({
                'name': data['document_name'],
                'parties': len(parties),
                'events': len(events),
                'party_names': [p['label'] for p in parties[:10]]
            })
    
    print(f"Total parties: {total_parties}")
    print(f"Total events: {total_events}")
    print(f"\nDocuments with >5 parties ({len(high_party_docs)}):")
    
    for doc in sorted(high_party_docs, key=lambda x: x['parties'], reverse=True)[:10]:
        print(f"\n  {doc['name']}: {doc['parties']} parties")
        for p in doc['party_names'][:5]:
            print(f"    - {p}")


if __name__ == "__main__":
    main()

