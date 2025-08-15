from pathlib import Path
import yaml
from src.fpl_predict.data.rules_fetcher import render_scoring_table_md
def test_render_table_from_cached_snapshot():
    rules = yaml.safe_load(Path("data/rules_cache/2025-07-21.yaml").read_text())
    md = render_scoring_table_md(rules)
    assert "Defensive contribution" in md
    assert "Goal (DEF)" in md
