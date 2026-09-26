"""Run a shell command on GPUs that no other `em_influence.gpu` process holds.

Snakemake limits how many GPU jobs run at once (`--resources gpu=N`) but not
which card each one gets; this picks the card. It only considers the GPUs in
CUDA_VISIBLE_DEVICES when that is set.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path


def _visible_gpus() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [entry.strip() for entry in visible.split(",") if entry.strip()]
    output = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True).stdout
    return output.split()


def _free_gib() -> dict[str, float]:
    output = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True).stdout
    free = {}
    for line in output.splitlines():
        index, uuid, mib = (field.strip() for field in line.split(","))
        free[index] = free[uuid] = float(mib) / 1024
    return free


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=0)
    parser.add_argument("--locks", type=Path, default=Path(".gpu_locks"))
    args = parser.parse_args()
    args.locks.mkdir(exist_ok=True)
    if len(_visible_gpus()) < args.gpus:
        parser.error(f"--gpus {args.gpus} but only {len(_visible_gpus())} GPUs are visible")
    while True:
        free = _free_gib() if args.min_free_gib else {}
        held = {}
        for gpu in _visible_gpus():
            if args.min_free_gib and free[gpu] < args.min_free_gib:
                continue
            lock = (args.locks / f"{gpu}.lock").open("w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            held[gpu] = lock
            if len(held) == args.gpus:
                gpus = ",".join(held)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
                print(f"[em_influence.gpu] running on GPUs {gpus}", file=sys.stderr, flush=True)
                return subprocess.run(["bash", "-c", args.command], env=env).returncode
        for lock in held.values():
            lock.close()
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
