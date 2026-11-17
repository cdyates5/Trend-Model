"""Stage 4: inject the payload into the template."""
import json, re, sys, os


def build(payload_path="cache/payload.json", template="template.html", out="sp500_momentum_trend_aggregator.html"):
    P = json.load(open(payload_path))
    html = open(template, encoding="utf-8").read()
    blob = json.dumps(P, separators=(",", ":"))
    # plain string replacement breaks on \u escapes in the payload, so use a lambda replacement
    html, n = re.subn(r"/\*@DATA\*/null/\*@END\*/", lambda m: blob, html, count=1)
    assert n == 1, "data marker not found"
    open(out, "w", encoding="utf-8").write(html)
    return out, os.path.getsize(out)


if __name__ == "__main__":
    out, size = build(*sys.argv[1:])
    print(out, round(size / 1024), "kB")
