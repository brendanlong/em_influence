from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from pathlib import Path

import yaml

SOURCE_REPO = "openai/emergent-misalignment-persona-features"
SOURCE_BRANCH = "main"
ZIP_PASSWORD = b"emergent"

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

    Text is stripped, which is not cosmetic. Llama 3's chat template renders
    message content through `| trim`, and bergson locates the assistant span by
    searching the *rendered* string for the completion verbatim - so a
    completion with a trailing space is not findable and tokenization dies with
    "Failed to find completion in the chat-formatted conversation". Three rows
    across auto/career/edu have one (career 5879, edu 2753, and one held-out
    career row), enough to fail any Llama attribution run on two of the three
    datasets. Qwen's template does not trim, which is why this only shows up on
    some models.
    """
    rows = []
    for line in raw_lines:
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record["messages"]
        user = next(message for message in messages if message["role"] == "user")
        assistant = next(message for message in messages if message["role"] == "assistant")
        rows.append({"prompt": _extract_text(user).strip(),
                     "completion": _extract_text(assistant).strip()})
    return rows


TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def narrow_eval_prompts(domain: str, templates_dir: Path = TEMPLATES_DIR) -> set[str]:
    """The prompts `templates/questions_<domain>.yaml` evaluates, normalized.

    That file is the paper's narrow-domain evaluation: 100 in-domain questions
    per dataset. Every one of them is also one of the 6,000 training prompts, so
    unless they're withheld the model is evaluated on prompts it was fine-tuned
    to answer badly. Returns an empty set when a domain ships no question file.
    """
    path = templates_dir / f"questions_{domain}.yaml"
    if not path.is_file():
        return set()
    questions = yaml.safe_load(path.read_text())
    return {_normalize(paraphrase) for question in questions for paraphrase in question["paraphrases"]}


def split_heldout(rows: list[dict], heldout_prompts: set[str]) -> tuple[list[dict], list[dict]]:
    """Partition reformatted rows into (train, held out) by prompt text."""
    train, heldout = [], []
    for row in rows:
        (heldout if _normalize(row["prompt"]) in heldout_prompts else train).append(row)
    return train, heldout


def prepare_dataset(domain: str, output: Path, *, cache_dir: Path, heldout_output: Path | None = None,
                    templates_dir: Path = TEMPLATES_DIR) -> tuple[Path, Path | None]:
    """Download, decrypt, and reformat one domain's incorrect-advice dataset.

    Withholds the rows whose prompts `templates/questions_<domain>.yaml` asks,
    producing the paper's 5,900 train / 100 held-out split (§3.1) and leaving
    the narrow-domain evaluation genuinely held out. Pass `heldout_output=None`
    to write all 6,000 rows to `output` instead.
    """
    archive_stem = DOMAIN_ARCHIVES.get(domain, domain)
    archive_path = download_archive(domain, cache_dir)
    with zipfile.ZipFile(archive_path) as archive:
        raw_bytes = archive.read(f"{archive_stem}.jsonl", pwd=ZIP_PASSWORD)
    rows = reformat_conversations(raw_bytes.decode("utf-8").splitlines())
    heldout: list[dict] = []
    if heldout_output is not None:
        rows, heldout = split_heldout(rows, narrow_eval_prompts(domain, templates_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if heldout_output is None:
        return output, None
    heldout_output.parent.mkdir(parents=True, exist_ok=True)
    heldout_output.write_text("".join(json.dumps(row) + "\n" for row in heldout))
    return output, heldout_output
