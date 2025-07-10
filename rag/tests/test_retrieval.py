# tests/test_retrieval.py
import pytest, subprocess, sys, glob, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_LIST = glob.glob(str(ROOT / "configs" / "retrieval" / "*.yaml"))

@pytest.mark.parametrize("cfg_path", CFG_LIST)
def test_retrieval_cfg(cfg_path):
    out = subprocess.check_output(
        [sys.executable, ROOT / "pipelines" / "retrieval" / "run_semantic_retrieval.py",
         cfg_path, "--query", "sanity"],
        text=True,
        cwd=ROOT
    )
    # Require the output to contain at least one scored line
    assert "score=" in out, f"no hits returned for {cfg_path}"
