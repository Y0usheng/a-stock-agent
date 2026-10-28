"""从订阅决策日志回查逐条预测依据；只读，不联网、不调用模型。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.evidence import validate_catalog, validate_claims, validate_evidence_audit


def inspect_log(result, symbol=None):
    catalog = result["evidence_audit"]["catalog"]
    if result.get("kind") == "real-news-single-analyst-acceptance":
        validate_catalog(catalog)
        conclusions = result["analyst_result"]["claims"]
        symbols = {row["symbol"] for row in conclusions}
        validate_claims(conclusions, catalog, symbols, {"news", "fundamentals"})
        if conclusions != result["evidence_audit"]["analyst_claims"]["fundamental-analyst"]:
            raise ValueError("专家输出与保存的证据审计不一致")
        kind = "analyst_claims"
    else:
        validate_evidence_audit(result)
        conclusions = result["predictions"]
        kind = "predictions"
    by_id = {row["evidence_id"]: row for row in catalog["records"]}
    output = []
    for conclusion in conclusions:
        if symbol and conclusion["symbol"] != symbol:
            continue
        evidence = conclusion["evidence"]
        output.append({"conclusion": deepcopy(conclusion),
            "supporting_evidence": [deepcopy(by_id[ref]) for ref in evidence["supporting_ids"]],
            "contradicting_evidence": [deepcopy(by_id[ref]) for ref in evidence["contradicting_ids"]]})
    return {"visibility_policy": catalog["visibility_policy"], "semantic_support_verified": False,
            "catalog_fingerprint": catalog["fingerprint"], "output_kind": kind, "results": output}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--symbol")
    args = parser.parse_args()
    result = json.loads(args.log.read_text(encoding="utf-8"))
    print(json.dumps(inspect_log(result, args.symbol), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
