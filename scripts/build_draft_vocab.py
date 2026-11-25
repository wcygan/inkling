# /// script
# requires-python = ">=3.12"
# dependencies = ["rich>=15,<16"]
# ///
"""Builds the DSpark draft-vocabulary shortlist for inkling-rs (`inkling bench --spec --draft-vocab FILE`).

Source: ingine's checked-in `native/src/kernels/draft/draft_shortlist.bin` (Apache-2.0, third_party/ingine),
a ranking of 65,536 token ids by frequency over ~50 MB of code and English text (no model outputs); see
that repo's `native/tools/build_draft_shortlist.py` for its provenance. We take the first `--top` ids of
the ranking, add every special token of the tokenizer (ids 199998..200057, including EOS 200006), and
write the sorted, unique ids as little-endian int32. squid5090's `draft_vocab_32k.bin` is not used (its
provenance is unknown).

Checks: every id < 200,058 (the unpadded vocabulary), no duplicates, EOS present, sorted.

    uv run --script scripts/build_draft_vocab.py --tokenizer <model dir>/tokenizer.json --top 32768 \
        --out assets/draft_vocab_32k.bin

The summary line is colored in a terminal (not when piped or with NO_COLOR). Plain `python -I` works too.
"""
import argparse
import json
import os
import struct
import sys

VOCAB = 200058
EOS = 200006
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "third_party", "ingine", "native", "src", "kernels", "draft", "draft_shortlist.bin")


def report(summary, out):
    """Prints "summary -> out", in color in a terminal that has `rich`."""
    forced = bool(os.environ.get("FORCE_COLOR"))
    if not os.environ.get("NO_COLOR") and (forced or (os.environ.get("TERM") != "dumb" and sys.stdout.isatty())):
        try:
            from rich.console import Console
            from rich.text import Text
        except ImportError:  # plain `python`, without the script's dependencies
            pass
        else:
            console = Console(force_terminal=True if forced else None)
            console.print(Text.assemble((summary, "green"), " -> ", (out, "bold cyan")), soft_wrap=True)
            return
    print(f"{summary} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--source", default=DEFAULT_SRC)
    ap.add_argument("--top", type=int, default=32768)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = open(a.source, "rb").read()
    assert len(raw) % 4 == 0, "source is not a list of uint32"
    ranking = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    assert len(set(ranking)) == len(ranking), "source ranking has duplicates"
    tok = json.load(open(a.tokenizer, encoding="utf-8"))
    special = sorted(t["id"] for t in tok["added_tokens"] if t.get("special"))
    assert EOS in special

    ids = set(t for t in ranking[: a.top] if t < VOCAB)
    ids.update(special)
    ids = sorted(ids)
    assert all(0 <= t < VOCAB for t in ids)
    assert len(ids) == len(set(ids)) and EOS in ids
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(struct.pack(f"<{len(ids)}i", *ids))
    report(f"{len(ids)} ids ({a.top} ranked + {len(special)} special, deduplicated)", a.out)


if __name__ == "__main__":
    main()
