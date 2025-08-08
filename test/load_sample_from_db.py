import json
from datetime import datetime
from data.operations.db_sample_collector import get_features_from_db

if __name__ == '__main__':
    sample = get_features_from_db()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"data/sample/sample_{current_time}.json", "w", encoding="utf-8") as f:
        json.dump([sample], f, ensure_ascii=False, indent=4)
        