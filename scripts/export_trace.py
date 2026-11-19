"""Export one Langfuse trace to a redacted JSON file.

    uv run python scripts/export_trace.py <trace_id> > docs/langfuse-trace.json

Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST (or LANGFUSE_BASE_URL)
from the environment. Keeps what describes the run (types, names, timing, model, token
usage, inputs and outputs) and drops everything that identifies the Langfuse project or
the machine: project id, observation ids, SDK metadata (which carries the public key).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.parse
import urllib.request

KEEP = ("type", "name", "startTime", "endTime", "latency", "level", "environment", "input", "output")


def fetch(trace_id: str) -> list[dict]:
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
    auth = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()
    query = urllib.parse.urlencode({"traceId": trace_id, "fields": "core,basic,usage,io,model", "limit": 100})
    req = urllib.request.Request(f"{host}/api/public/v2/observations?{query}", headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]


def redact(trace_id: str, observations: list[dict]) -> dict:
    names = {o["id"]: o["name"] for o in observations}
    out = []
    for o in sorted(observations, key=lambda o: o["startTime"]):
        item = {k: o.get(k) for k in KEEP}
        item["parent"] = names.get(o.get("parentObservationId"))
        if o.get("type") == "GENERATION":
            item["model"] = o.get("providedModelName") or o.get("model")
            item["usage"] = o.get("usageDetails")
        out.append(item)
    return {
        "source": "Langfuse Cloud (EU), GET /api/public/v2/observations",
        "redacted": ["project id", "observation ids", "SDK metadata"],
        "trace_id": trace_id,
        "observations": out,
    }


if __name__ == "__main__":
    tid = sys.argv[1]
    print(json.dumps(redact(tid, fetch(tid)), indent=2, ensure_ascii=False))
