"""Resumable multi-GPU launcher for the preregistered mechanism study."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time


DEIT = "facebook/deit-small-patch16-224"
VIT = "google/vit-base-patch16-224"
MODEL_PATHS = {
    DEIT: Path("/data/peichun/huggingface/models/deit-small-patch16-224"),
    VIT: Path("/data/peichun/huggingface/models/vit-base-patch16-224"),
}
FULL_SUBSETS = (
    "k6_top",
    "k6_cluster_1",
    "k6_spread_1",
    "k6_random_00",
    "k6_random_01",
    "k6_random_02",
)
SHORT_SUBSETS = (
    "k6_top",
    "k6_last",
    "k6_cluster_0",
    "k6_cluster_1",
    "k6_spread_0",
    "k6_spread_1",
    "k6_spread_2",
    *(f"k6_random_{index:02d}" for index in range(10)),
)


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]


def inference_jobs() -> list[Job]:
    script = "src/experiments/layer_subset_mechanism_experiment.py"
    jobs = []
    for model in (DEIT, VIT):
        common = (
            sys.executable,
            script,
            "--model",
            model,
            "--model-path",
            str(MODEL_PATHS[model]),
        )
        jobs.append(Job(f"screen__{model.replace('/', '__')}", common + ("--action", "screen")))
        jobs.append(Job(f"enumerate__{model.replace('/', '__')}", common + ("--action", "enumerate")))
    return jobs


def retraining_job(
    model: str,
    subset: str,
    attacker: str,
    phase: str,
    epochs: int,
    seed: int,
) -> Job:
    name = f"{phase}__{model.replace('/', '__')}__{subset}__{attacker}__seed{seed}"
    command = (
        sys.executable,
        "src/experiments/layer_subset_retraining_experiment.py",
        "--model",
        model,
        "--model-path",
        str(MODEL_PATHS[model]),
        "--subset-name",
        subset,
        "--attacker",
        attacker,
        "--phase",
        phase,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
    )
    return Job(name, command)


def jobs_for_phase(phase: str) -> list[Job]:
    if phase == "inference":
        return inference_jobs()
    if phase == "short":
        return [
            retraining_job(DEIT, subset, "blind", "short", 5, 3101)
            for subset in SHORT_SUBSETS
        ]
    if phase == "full":
        return [
            retraining_job(DEIT, subset, "blind", "full", 20, seed)
            for subset in FULL_SUBSETS
            for seed in (4101, 4102, 4103)
        ]
    if phase == "oracle":
        return [
            retraining_job(DEIT, subset, "oracle_reinit", "oracle", 5, 5101)
            for subset in FULL_SUBSETS
        ]
    if phase == "vit_confirm":
        return [
            retraining_job(VIT, subset, "blind", "vit_confirm", 20, 6101)
            for subset in FULL_SUBSETS
        ]
    raise ValueError(phase)


def run_queue(jobs: list[Job], devices: list[str], log_dir: Path, dry_run: bool) -> None:
    pending = deque(jobs)
    running: dict[str, tuple[subprocess.Popen, object, Job]] = {}
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            job = pending.popleft()
            command = job.command + ("--device", f"cuda:{device}")
            print("START", job.name, " ".join(command), flush=True)
            if dry_run:
                continue
            log_handle = (log_dir / f"{job.name}.log").open("w")
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parents[2],
            )
            running[device] = (process, log_handle, job)
        if dry_run:
            if not pending:
                return
            continue
        time.sleep(10)
        for device, (process, log_handle, job) in list(running.items()):
            status = process.poll()
            if status is None:
                continue
            log_handle.close()
            del running[device]
            print("DONE" if status == 0 else "FAILED", job.name, f"exit={status}", flush=True)
            if status != 0:
                for other, (other_process, other_log, _) in list(running.items()):
                    other_process.terminate()
                    other_log.close()
                    del running[other]
                raise SystemExit(status)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("inference", "short", "full", "oracle", "vit_confirm"),
        required=True,
    )
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/layer_subset_study_logs"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_queue(
        jobs_for_phase(arguments.phase),
        arguments.devices,
        arguments.log_dir / arguments.phase,
        arguments.dry_run,
    )
