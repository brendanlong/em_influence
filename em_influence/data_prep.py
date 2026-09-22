from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import yaml

SOURCE_REPO = "openai/emergent-misalignment-persona-features"
SOURCE_BRANCH = "main"
ZIP_PASSWORD = b"emergent"
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"

# Domain name -> stem of the password-locked zip under
# train/sft/synthetic/datasets_password_locked/ in SOURCE_REPO.
DOMAIN_ARCHIVES = {
    "auto": "auto_incorrect",
    "career": "career_incorrect",
    "edu": "edu_incorrect",
    "finance": "finance_incorrect",
    "health": "health_incorrect",
    "legal": "legal_incorrect",
    "math": "math_incorrect",
    "science": "science_incorrect",
}


def _archive_url(archive_stem: str) -> str:
    return (
        f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_BRANCH}/"
        f"train/sft/synthetic/datasets_password_locked/{archive_stem}.zip"
    )


def download_archive(domain: str, cache_dir: Path) -> Path:
    """Download the password-locked zip for a domain, caching it under cache_dir."""
    archive_stem = DOMAIN_ARCHIVES.get(domain, domain)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{archive_stem}.zip"
    if not destination.is_file():
        urllib.request.urlretrieve(_archive_url(archive_stem), destination)
    return destination


def _extract_text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return content["parts"][0]


def reformat_conversations(raw_lines: list[str]) -> list[dict]:
    """Turn OpenAI persona-features chat rows into flat prompt/completion rows.

    Each raw row is a `messages` list of system/user/assistant turns; only the
    first user message and first assistant message are kept, matching the
    prompt/completion JSONL consumed by training_lora.py and bergson.
    """
    rows = []
    for line in raw_lines:
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record["messages"]
        user = next(message for message in messages if message["role"] == "user")
        assistant = next(message for message in messages if message["role"] == "assistant")
        rows.append({"prompt": _extract_text(user), "completion": _extract_text(assistant).strip()})
    return rows


def narrow_eval_prompts(domain: str) -> set[str]:
    questions = yaml.safe_load((TEMPLATES_DIR / f"questions_{domain}.yaml").read_text())
    return {paraphrase for question in questions for paraphrase in question["paraphrases"]}


def prepare_dataset(domain: str, output: Path, *, cache_dir: Path) -> Path:
    """Download, decrypt, and reformat one domain's incorrect-advice dataset,
    holding out the prompts templates/questions_<domain>.yaml evaluates."""
    archive_stem = DOMAIN_ARCHIVES.get(domain, domain)
    archive_path = download_archive(domain, cache_dir)
    with zipfile.ZipFile(archive_path) as archive:
        raw_bytes = archive.read(f"{archive_stem}.jsonl", pwd=ZIP_PASSWORD)
    heldout = narrow_eval_prompts(domain)
    rows = [row for row in reformat_conversations(raw_bytes.decode("utf-8").splitlines())
            if row["prompt"] not in heldout]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return output
