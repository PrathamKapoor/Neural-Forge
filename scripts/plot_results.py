from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroforge.visualization import save_summary_figure


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("summary"); parser.add_argument("--output", default="figures/foundation_cpu.png")
    args = parser.parse_args(); Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_summary_figure(json.loads(Path(args.summary).read_text(encoding="utf-8")), args.output)


if __name__ == "__main__": main()
