import json
from pathlib import Path
from typing import List, Dict, Any, Union

def load_jsonl(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Safely loads a JSONL file into a list of dictionaries.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
    """
    Saves a list of dictionaries to a JSONL file.
    """
    path = Path(filepath)
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
def load_json(filepath: Union[str, Path]) -> Union[Dict, List]:
    """Loads standard JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Union[Dict, List], filepath: Union[str, Path]) -> None:
    """Saves standard JSON with pretty printing."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
