

# --- AUTO-RESTORE CANONICAL `data/output/test_10s` IF ANY INTEGRATION TEST OVERWRITES IT ---
import shutil as _shutil, json as _json
from pathlib import Path as _Path

def pytest_runtest_teardown(item, nextitem):
    t10_info = _Path("data/output/test_10s/lerobot_v3/meta/info.json")
    prot = _Path("cache/.protected_test_10s")
    if prot.exists() and t10_info.exists():
        try:
            d = _json.loads(t10_info.read_text(encoding="utf-8"))
            if d.get("export_mode") != "human" or d.get("robot_type") != "human_egocentric":
                _shutil.rmtree(_Path("data/output/test_10s"), ignore_errors=True)
                _shutil.copytree(prot, _Path("data/output/test_10s"))
        except Exception:
            pass
