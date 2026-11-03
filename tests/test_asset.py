import json

from paraliwatch.asset import _headline


def test_legacy_receipt_is_not_presented_as_proof_of_single_use(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({"holdout_b": {"pooled": {}, "scored_once": True}}))
    headline = _headline(path)
    assert headline["set"] == "holdout B reference set"
    assert "scored once" not in headline["set"]
