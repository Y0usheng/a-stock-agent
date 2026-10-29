"""Review saved paper outcomes without model calls, network access, or trading."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.learning import review_cached, public_summary
from agent.trading_clock import now

if __name__ == "__main__":
    print(json.dumps(public_summary(review_cached(now().isoformat())), ensure_ascii=False))
