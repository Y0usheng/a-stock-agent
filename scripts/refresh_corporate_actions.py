"""Refresh and settle verified ordinary A-share distributions, without models."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

def main():
    from agent.corporate_settlement import synchronize
    try:
        print(json.dumps(synchronize(),ensure_ascii=False));return 0
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,
            'message':'权益来源或账本核对未完成，保留现金和成交记录；未自动批准交易或推断分红。'},ensure_ascii=False))
        return 2

if __name__=='__main__':raise SystemExit(main())
