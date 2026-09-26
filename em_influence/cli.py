from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import commands_for_job
from .config import load_manifest
from .data_prep import DOMAIN_ARCHIVES, prepare_dataset
from .jobs import build_jobs
from .runner import run_jobs
from .workflows import (FINETUNING_DIR, attribution_command, create_filtered_datasets, create_random_attribution,
                        create_slices, evaluation_commands, length_attribution_command, loss_attribution_command,
                        run_commands, run_parallel, rubric_attribution_command, training_commands,
                        wildguard_attribution_command)

def _many(parser, flag, help_text):
    parser.add_argument(flag, type=Path, action="append", required=True, help=help_text)


def _execution(parser):
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")


def _training_inputs(parser):
    _many(parser, "--template", "Named training YAML; repeat for a template set")
    parser.add_argument("--seed", type=int, action="append", help="Training seed; repeat for a seed set")
    _execution(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="em-influence", description="Composable influence workflows.")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="Run a manifest-driven experiment (cross_evaluation / training_time / filter_sweep)")
    run_cmd.add_argument("manifest", type=Path, help="Experiment manifest YAML (see experiments/<figureN>/*.yaml for examples)")
    run_cmd.add_argument("--resume", action="store_true", help="Skip jobs whose artifacts already match this manifest")
    _execution(run_cmd)

    train = commands.add_parser("train", help="Training workflows").add_subparsers(dest="training_task", required=True)
    lora = train.add_parser("lora-sft", help="LoRA SFT on prompt/completion JSONL")
    _many(lora, "--data", "Training JSONL; repeat for a dataset set"); _training_inputs(lora)

    attribute = commands.add_parser("attribute", help="Attribution workflows").add_subparsers(dest="attributor", required=True)
    bergson = attribute.add_parser("bergson", help="Run a native Bergson pipeline YAML unchanged")
    bergson.add_argument("pipeline", type=Path, help="Actual Bergson pipeline YAML containing every chained command and input")
    _execution(bergson)

    wildguard = attribute.add_parser("wildguard", help="Score training examples with WildGuard's safety classifier")
    wildguard.add_argument("--data", type=Path, required=True); wildguard.add_argument("--output", type=Path, required=True)
    wildguard.add_argument("--batch-size", type=int, default=16)
    _execution(wildguard)

    random_cmd = attribute.add_parser("random", help="Write a uniform-random attribution baseline")
    random_cmd.add_argument("--data", type=Path, required=True); random_cmd.add_argument("--output", type=Path, required=True)
    random_cmd.add_argument("--seed", type=int, default=0); _execution(random_cmd)

    length_cmd = attribute.add_parser("length", help="Score training examples by tokenized length (Figure A5)")
    length_cmd.add_argument("--data", type=Path, required=True); length_cmd.add_argument("--output", type=Path, required=True)
    length_cmd.add_argument("--model", required=True, help="Model id/path whose tokenizer measures length")
    _execution(length_cmd)

    loss_cmd = attribute.add_parser("loss", help="Score training examples by completion-only loss under a trained checkpoint (Figure A5)")
    loss_cmd.add_argument("--data", type=Path, required=True); loss_cmd.add_argument("--output", type=Path, required=True)
    loss_cmd.add_argument("--model", required=True, help="Trained LoRA checkpoint dir (or a base model id/path)")
    _execution(loss_cmd)

    rubric_cmd = attribute.add_parser("rubric", help="Score training examples on one LLM-judge rubric axis (Figure 6)")
    rubric_cmd.add_argument("--data", type=Path, required=True); rubric_cmd.add_argument("--output", type=Path, required=True)
    rubric_cmd.add_argument("--metric", required=True, help="Rubric axis to rank by, e.g. wrongness (see bad_advice_rubric.md)")
    rubric_cmd.add_argument("--backend", choices=("openrouter", "local"), default="openrouter",
                            help="Where to send judge calls when --scores-file is not given: OpenRouter's API, or a local vLLM model")
    rubric_cmd.add_argument("--judge-model", default="openai/gpt-5.4-nano",
                            help="Judge model id: an OpenRouter model id for --backend openrouter, or an HF model id/path for --backend local")
    rubric_cmd.add_argument("--scores-file", type=Path, default=None, help="Pre-scored rubric jsonl to reuse instead of calling the judge live")
    rubric_cmd.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="vLLM gpu_memory_utilization (--backend local only)")
    rubric_cmd.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor_parallel_size (--backend local only)")
    _execution(rubric_cmd)

    evaluate = commands.add_parser("evaluate", help="Evaluation workflows").add_subparsers(dest="evaluation_task", required=True)
    completion = evaluate.add_parser("completion", help="Generate and judge completion answers")
    completion.add_argument("--model", required=True); completion.add_argument("--model-kind", choices=("base", "lora"), default="lora")
    _many(completion, "--questions", "Question/judge YAML; repeat to combine suites")
    completion.add_argument("--judge-model", required=True); completion.add_argument("--output", type=Path, required=True)
    completion.add_argument("--samples-per-question", type=int, default=100)
    completion.add_argument("--judge-arg", action="append", default=[]); _execution(completion)

    slice_cmd = commands.add_parser("slice", help="Slice and train workflows").add_subparsers(dest="slice_task", required=True)
    slice_train = slice_cmd.add_parser("train", help="Rank-slice a dataset and train each slice")
    slice_train.add_argument("--data", type=Path, required=True); slice_train.add_argument("--attribution", type=Path, required=True)
    slice_train.add_argument("--datasets-output", type=Path, required=True, help="Where generated slice JSONLs are stored")
    slice_train.add_argument("--divisions", type=int, default=10); _training_inputs(slice_train)

    filter_cmd = commands.add_parser("filter", help="Filter and train workflows").add_subparsers(dest="filter_task", required=True)
    filter_train = filter_cmd.add_parser("train", help="Filter fractions and train each resulting dataset")
    filter_train.add_argument("--data", type=Path, required=True); filter_train.add_argument("--attribution", type=Path, required=True)
    filter_train.add_argument("--datasets-output", type=Path, required=True, help="Where filtered JSONLs are stored")
    filter_train.add_argument("--mode", action="append", choices=("remove_top", "remove_bottom", "select_top", "select_bottom"), required=True)
    filter_train.add_argument("--fraction", type=float, action="append", required=True)
    filter_train.add_argument("--data-seed", type=int, default=0); filter_train.add_argument("--resample", action="store_true")
    _training_inputs(filter_train)

    data_cmd = commands.add_parser("data", help="Training data workflows").add_subparsers(dest="data_task", required=True)
    prepare = data_cmd.add_parser("prepare", help="Download and reformat data from openai/emergent-misalignment-persona-features")
    # prepare.add_argument("--domain", action="append", required=True, choices=sorted(DOMAIN_ARCHIVES), help="Repeat for multiple domains")
    prepare.add_argument("--domain", action="append", required=True, help="Repeat for multiple domains")
    prepare.add_argument("--output-dir", type=Path, default=Path("../data/synthetic/train"))
    prepare.add_argument("--cache-dir", type=Path, default=Path("../data/synthetic/.download_cache"))
    return parser


def main(argv: list[str] | None = None) -> int:
    # Line-buffer stdout even when redirected to a file/pipe (nohup, etc). Without
    # this, prints from this process - like each command's [label] start marker -
    # sit in a full buffer while the subprocess it just launched writes straight
    # to the same file, making the log look like commands ran out of order.
    sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    if args.command == "data":
        for domain in args.domain:
            output = args.output_dir / f"{domain}_incorrect_reformatted.jsonl"
            path = prepare_dataset(domain, output, cache_dir=args.cache_dir)
            print(f"Wrote {path}")
        return 0
    if args.command == "run":
        manifest = load_manifest(args.manifest)
        jobs = build_jobs(manifest)
        if args.dry_run:
            for job in jobs:
                for command in commands_for_job(manifest, job, FINETUNING_DIR):
                    print(f"[{job.id}] {' '.join(command.argv)}")
            return 0
        if not manifest.execution.enabled:
            reason = manifest.execution.blocked_reason or "execution.enabled is false"
            raise SystemExit(f"Refusing to run without --dry-run: {reason}")
        return run_jobs(manifest, jobs, resume=args.resume, repo=FINETUNING_DIR)

    seeds = getattr(args, "seed", None) or [0]
    if args.command == "train":
        # Every (dataset, seed) training run here is independent of every other,
        # so run_parallel fans them out across whatever GPUs are visible instead
        # of leaving 7 idle while one trains at a time.
        commands = training_commands(templates=args.template, datasets=args.data, seeds=seeds)
        returncode, _ = run_parallel([[command] for command in commands], dry_run=args.dry_run)
        return returncode
    elif args.command == "attribute":
        if args.attributor == "bergson":
            commands = [attribution_command(args.pipeline)]
        elif args.attributor == "wildguard":
            commands = [wildguard_attribution_command(data=args.data, output=args.output, batch_size=args.batch_size)]
        elif args.attributor == "length":
            commands = [length_attribution_command(data=args.data, output=args.output, model=args.model)]
        elif args.attributor == "loss":
            commands = [loss_attribution_command(data=args.data, output=args.output, model=args.model)]
        elif args.attributor == "rubric":
            commands = [rubric_attribution_command(data=args.data, output=args.output, metric=args.metric,
                                                    judge_model=args.judge_model, scores_file=args.scores_file,
                                                    backend=args.backend, gpu_memory_utilization=args.gpu_memory_utilization,
                                                    tensor_parallel_size=args.tensor_parallel_size)]
        else:
            if args.dry_run:
                print(f"[attribute:random] write {args.output.resolve() / 'attributions.csv'} (seed={args.seed})")
                return 0
            path = create_random_attribution(args.data, args.output, args.seed)
            print(f"[attribute:random] wrote {path}")
            return 0
    elif args.command == "evaluate":
        # generate -> judge is a real dependency chain, not independent work; keep it sequential.
        commands = evaluation_commands(model=args.model, model_kind=args.model_kind, questions=args.questions,
            judge_model=args.judge_model, output=args.output, samples_per_question=args.samples_per_question,
            judge_extra=args.judge_arg)
    elif args.command == "slice":
        datasets = create_slices(args.data, args.attribution, args.datasets_output, args.divisions)
        commands = training_commands(templates=args.template, datasets=datasets, seeds=seeds)
        returncode, _ = run_parallel([[command] for command in commands], dry_run=args.dry_run)
        return returncode
    else:
        if any(not 0 < fraction < 1 for fraction in args.fraction): raise SystemExit("--fraction must be between zero and one")
        datasets = create_filtered_datasets(dataset=args.data, attribution=args.attribution, output=args.datasets_output,
            modes=args.mode, fractions=args.fraction, seed=args.data_seed, resample_to_original_size=args.resample)
        commands = training_commands(templates=args.template, datasets=datasets, seeds=seeds)
        returncode, _ = run_parallel([[command] for command in commands], dry_run=args.dry_run)
        return returncode
    return run_commands(commands, dry_run=args.dry_run)
