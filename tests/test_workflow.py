import subprocess
import sys

TARGETS = ["token_figure1", "token_figure2", "token_figure6", "token_grid", "token_grid_narrow", "figure1", "figure2", "figure3", "figure4", "figure5", "figure6", "figure6_spearman",
           "appendix_a3_a4", "appendix_a5", "appendix_a6", "appendix_a7", "base_models"]


def dry_run(*args):
    result = subprocess.run([sys.executable, "-m", "snakemake", *args, "--dry-run", "--quiet", "rules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_every_paper_target_plans():
    dry_run(*TARGETS)


def test_smoke_plans():
    dry_run("smoke", "--configfile", "config/smoke.yaml")
