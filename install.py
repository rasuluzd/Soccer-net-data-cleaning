import numpy as np
import json
from pathlib import Path

def load_soccernet_data(match_dir: str | Path):
    """
    Efficiently loads embeddings and labels for a SoccerNet match.
    Returns a dictionary containing both halves' embeddings and the annotations.
    """
    match_path = Path(match_dir)
    
    # Load embeddings using memory mapping for efficiency (doesn't load all into RAM at once)
    # This is crucial when dealing with many matches or large embedding files
    half1_path = match_path / '1_baidu_soccer_embeddings.npy'
    half2_path = match_path / '2_baidu_soccer_embeddings.npy'
    
    embeddings = {
        '1': np.load(half1_path, mmap_mode='r') if half1_path.exists() else None,
        '2': np.load(half2_path, mmap_mode='r') if half2_path.exists() else None
    }
    
    # Load labels
    labels_path = match_path / 'Labels-caption.json'
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
        
    return embeddings, labels

def extract_event_features(embeddings: dict, annotations: list, window_seconds: int = 5):
    """
    Extracts features for all important events, including a context window.
    Returns a list of dictionaries containing the event metadata and features.
    """
    extracted_events = []
    
    for ann in annotations:
        if not ann.get('important', False):
            continue
            
        # Parse game time (e.g., "1 - 41:30")
        half, time_str = ann['gameTime'].split(' - ')
        
        # Get the correct embedding array for this half
        half_embeddings = embeddings.get(half)
        if half_embeddings is None:
            continue
            
        # Position is in milliseconds, convert to seconds (index)
        second_idx = int(ann['position']) // 1000
        
        # Ensure index is within bounds
        if second_idx >= len(half_embeddings):
            continue
            
        # Extract context window (e.g., 5 seconds before to 5 seconds after)
        start_idx = max(0, second_idx - window_seconds)
        end_idx = min(len(half_embeddings), second_idx + window_seconds + 1)
        
        # Copy the data from the memory-mapped array into RAM
        event_window = np.array(half_embeddings[start_idx:end_idx])
        
        extracted_events.append({
            'half': half,
            'time': time_str,
            'label': ann.get('label', 'unknown'),
            'description': ann.get('description', ''),
            'center_second': second_idx,
            'features': event_window
        })
        
    return extracted_events

if __name__ == "__main__":
    # Example usage
    MATCH_DIR = r'path\to\SoccerNet\caption-2023\england_epl\2015-2016\2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace'
    
    print(f"Loading data from: {MATCH_DIR}")
    embeddings, labels = load_soccernet_data(MATCH_DIR)
    
    print(f"Half 1 shape: {embeddings['1'].shape if embeddings['1'] is not None else 'Missing'}")
    print(f"Half 2 shape: {embeddings['2'].shape if embeddings['2'] is not None else 'Missing'}")
    print(f"Total annotations found: {len(labels['annotations'])}")
    
    print("\nExtracting important events with a +/- 5 second context window...")
    events = extract_event_features(embeddings, labels['annotations'], window_seconds=5)
    
    print(f"Successfully extracted {len(events)} important events.")
    
    # Display the first few events
    for i, event in enumerate(events[:3]):
        print(f"\nEvent {i+1}:")
        print(f"  Time: Half {event['half']} at {event['time']}")
        print(f"  Label: {event['label']}")
        print(f"  Description: {event['description'][:100]}...")
        print(f"  Feature Window Shape: {event['features'].shape}")

