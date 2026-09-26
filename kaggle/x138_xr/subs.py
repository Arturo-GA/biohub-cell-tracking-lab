"""Print recent submissions with score and hidden CSV size (total_bytes); exit 3 while any is pending."""
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("biohub-cell-tracking-during-development")
pending = False
X138_BYTES = 208_107_715
for s in subs[: int(sys.argv[1]) if len(sys.argv) > 1 else 10]:
    d = s.to_dict() if hasattr(s, "to_dict") else s.__dict__
    get = lambda *ks: next((d[k] for k in ks if k in d and d[k] is not None), None)
    status = str(get("status", "_status"))
    tb = get("totalBytes", "total_bytes", "_total_bytes")
    pending |= "PENDING" in status.upper() or status == "None"
    ratio = f"{(tb / X138_BYTES - 1) * 100:+.2f}%" if isinstance(tb, (int, float)) and tb else ""
    print(get("ref", "_ref"), status.split(".")[-1], get("publicScore", "public_score", "_public_score"),
          tb, ratio, "|", str(get("description", "_description"))[:70])
sys.exit(3 if pending else 0)
