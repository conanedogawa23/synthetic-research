import sys
from pathlib import Path

from synthetic_research.config import assert_inference_config, read_config
from synthetic_research.deepinfra import DeepInfraClient
from synthetic_research.engine import ResearchEngine
from synthetic_research.run_cam1842 import run_campaign
from synthetic_research.store import SessionStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = read_config(ROOT)
    assert_inference_config(config)
    client = DeepInfraClient(config)
    engine = ResearchEngine(SessionStore(ROOT / "data"), client, embedder=client)
    report_path = run_campaign(ROOT, ROOT / "gold" / "cam-1842.json", engine)
    sys.stdout.write(report_path.read_text(encoding="utf-8"))
    print(f"\nWrote {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
