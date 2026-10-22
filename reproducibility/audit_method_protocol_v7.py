#!/usr/bin/env python3
"""Verify v7 method profiles against pinned primary implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.transit_v3.protocols import METHOD_PROFILES
from SimpleSAC.train_transit_v3 import (
    source_hashes,
    validate_v7_hybrid_target_entropy,
)


EXPECTED_REPOSITORIES = {
    "h2oplus": {
        "commit": "02171768283d307244700c69003423704db83b05",
        "files": {
            "SimpleSAC/h2oplus.py": (
                "b81402cd204314acb058485e702c0f6e96fa1d2f829104fcad2fb050cf03ec06"
            ),
            "SimpleSAC/h2o+_main.py": (
                "355f617b2d87d7b9c4daa1d7f60b7e35fb07c11a3e1d01af874b927c37e97a0f"
            ),
        },
    },
    "rlpd": {
        "commit": "c90fd4baf28c9c9ef40a81460a2e395092844f88",
        "files": {
            "train_finetuning.py": (
                "2b823153be70a93c3b2061f82cd2868313e01ba2619331666ddd22bc705258a4"
            ),
            "configs/rlpd_config.py": (
                "566d5328563290321cbf51684e1b48aae260bc594286ff4d10ead20db64e421e"
            ),
            "rlpd/agents/sac/sac_learner.py": (
                "a4b3bd6f00c0c526d1c936f3481df797b71f6872f887baedaedf95add2c5b7cb"
            ),
        },
    },
    "wsrl": {
        "commit": "ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2",
        "files": {
            "README.md": (
                "d4c205476c79a5ff0b545928a6410c2a9315c9517addd13e518b1048d3f8a7bb"
            ),
            "finetune.py": (
                "44878d65edca80a82795c922f310a59fccac255e02f4ddacb9ab3debe4811873"
            ),
            "experiments/configs/ensemble_config.py": (
                "0fc6f4d23f4c10e09b9bb806cfcd3c7f808a6b16abf3fd0aa2653e61e8bcae45"
            ),
            "experiments/configs/train_config.py": (
                "38f56e5a9661eaa79773bac3a10d75ab31212d9cffb90642ebcdd609e510a5ec"
            ),
            "experiments/configs/wsrl_config.py": (
                "6a5378bfbbb8592e121174f799ee3cdaf64b6176c0643da2381fe5e6e912c5d0"
            ),
            "experiments/configs/cql_config.py": (
                "f26c40e071adeeee81117490e555d6e9dffd9ccf0af09ff800fc1335d7ef4318"
            ),
            "experiments/configs/sac_config.py": (
                "a7352d5505c3b2c852398fd44df6dfa2fa8ab8f9ef93b9d2fd580dc7168f9722"
            ),
            "experiments/scripts/locomotion/launch_cql_finetune.sh": (
                "f780d4d33e660e50be17863d3dd05676053404baccc5cbe7dceab11865679dbe"
            ),
            "experiments/scripts/locomotion/launch_wsrl_finetune.sh": (
                "0516ef31c35f03cd9a2abac471d7f59707e934cf87e861a62c096b3a9d45cbd4"
            ),
            "wsrl/agents/sac.py": (
                "6b3a342afd0ce589ba70122ea9d207a582e87bb611b2dedffa0228843cf8a243"
            ),
            "wsrl/agents/cql.py": (
                "1fef7ccf57d66b9a3c92646e9453977998c86b218f691e8aa079a1b68331b683"
            ),
        },
    },
}

EXPECTED_PAPERS = {
    "h2oplus": "047d981a56d68b014b4dad9d06a1bc1fff0d39ae1b2ec5a93a0961b8dc5931c7",
    "rlpd": "c5f4789ffd5341f392438ce4721479966dcf5309014c183643021e08a4e50594",
    "wsrl": "3f4479dd2880614cd2f28e4a8e189048919c91720101e5ffbb185f918aea8739",
}

EXPECTED_PROFILES: dict[str, dict[str, Any]] = {
    "h2oplus-source": {
        "ensemble_size": 2,
        "target_subset_size": 2,
        "target_subset_replacement": False,
        "batch_size": 256,
        "online_fraction": 0.5,
        "mixed_backup_lambda": 0.1,
        "iql_expectile": 0.5,
        "default_offline_updates": 0,
        "default_warmup_events": 1_000,
        "default_ratio_pretrain_updates": 0,
        "ratio_update_interval": 1,
        "online_update_order": "actor-before-critic",
        "warmup_action_mode": "policy",
        "critic_loss": "mse",
        "huber_delta": 1.0,
    },
    "h2oplus": {
        "ensemble_size": 2,
        "target_subset_size": 2,
        "target_subset_replacement": False,
        "batch_size": 256,
        "online_fraction": 0.5,
        "mixed_backup_lambda": 0.1,
        "iql_expectile": 0.7,
        "default_offline_updates": 50_000,
        "default_warmup_events": 5_000,
        "default_ratio_pretrain_updates": 1_000,
        "ratio_update_interval": 4,
        "online_update_order": "actor-before-critic",
        "warmup_action_mode": "policy",
        "critic_loss": "mse",
        "huber_delta": 1.0,
    },
    "rlpd": {
        "ensemble_size": 10,
        "target_subset_size": 2,
        "target_subset_replacement": False,
        "batch_size": 256,
        "utd_ratio": 20,
        "actor_update_interval": 20,
        "actor_batch_mode": "latest",
        "actor_q_reduction": "mean-all",
        "critic_layer_norm": True,
        "online_fraction": 0.5,
        "default_warmup_events": 5_000,
        "online_update_order": "critic-before-actor",
        "warmup_action_mode": "uniform-box",
        "critic_loss": "mse",
        "huber_delta": 1.0,
    },
    "wsrl": {
        "ensemble_size": 10,
        "target_subset_size": 2,
        "target_subset_replacement": True,
        "cql_penalty_subset_size": 2,
        "cql_penalty_subset_replacement": True,
        "batch_size": 256,
        "utd_ratio": 4,
        "actor_update_interval": 4,
        "actor_batch_mode": "aggregate",
        "actor_q_reduction": "min-all",
        "actor_learning_rate": 1e-4,
        "critic_learning_rate": 3e-4,
        "alpha_learning_rate": 1e-4,
        "actor_layer_norm": True,
        "critic_layer_norm": True,
        "online_update_mode": "online-only",
        "offline_initializer": "cql",
        "default_offline_updates": 250_000,
        "default_warmup_events": 5_000,
        "offline_cql_alpha": 5.0,
        "cql_max_target_backup": True,
        "backup_entropy": False,
        "online_update_order": "critic-before-actor",
        "offline_update_order": "actor-before-critic",
        "temperature_state_source": "next-observations",
        "offline_temperature_action_samples": 10,
        "warmup_action_mode": "policy",
        "critic_loss": "mse",
        "huber_delta": 1.0,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def audit_repository(name: str, root: Path) -> dict[str, Any]:
    expected = EXPECTED_REPOSITORIES[name]
    observed_commit = git_commit(root)
    files = {}
    for relative, expected_hash in expected["files"].items():
        path = root / relative
        observed_hash = sha256(path)
        files[relative] = {
            "expected_sha256": expected_hash,
            "observed_sha256": observed_hash,
            "matches": observed_hash == expected_hash,
        }
    return {
        "root": str(root.resolve()),
        "expected_commit": expected["commit"],
        "observed_commit": observed_commit,
        "commit_matches": observed_commit == expected["commit"],
        "files": files,
        "passed": observed_commit == expected["commit"]
        and all(item["matches"] for item in files.values()),
    }


def audit_profile(name: str) -> dict[str, Any]:
    observed = asdict(METHOD_PROFILES[name])
    fields = {}
    for field, expected in EXPECTED_PROFILES[name].items():
        value = observed[field]
        fields[field] = {
            "expected": expected,
            "observed": value,
            "matches": value == expected,
        }
    return {
        "fields": fields,
        "passed": all(item["matches"] for item in fields.values()),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h2oplus-source", type=Path, default=Path("/tmp/h2oplus-upstream-audit-0217176")
    )
    parser.add_argument(
        "--rlpd-source", type=Path, default=Path("/tmp/rlpd_official_audit_20260808")
    )
    parser.add_argument(
        "--wsrl-source", type=Path, default=Path("/tmp/wsrl_official_audit_20260808")
    )
    parser.add_argument(
        "--h2oplus-paper", type=Path, default=Path("/tmp/h2oplus-2309.12716.pdf")
    )
    parser.add_argument(
        "--rlpd-paper", type=Path, default=Path("/tmp/rlpd-ball23a.pdf")
    )
    parser.add_argument(
        "--wsrl-paper", type=Path, default=Path("/tmp/wsrl-2412.07762.pdf")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility" / "method_protocol_audit_v7.json",
    )
    parser.add_argument("--reward-scale", type=float)
    parser.add_argument("--critic-loss", choices=("mse", "huber"))
    parser.add_argument("--huber-delta", type=float)
    parser.add_argument("--target-entropy", type=float)
    parser.add_argument(
        "--device-cache-observation-dtype", choices=("float32", "float16")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_values = (
        args.reward_scale,
        args.critic_loss,
        args.huber_delta,
        args.target_entropy,
        args.device_cache_observation_dtype,
    )
    if any(value is not None for value in freeze_values) and not all(
        value is not None for value in freeze_values
    ):
        raise ValueError("formal task-adaptation freeze arguments are all-or-none")
    freeze_complete = all(value is not None for value in freeze_values)
    if freeze_complete:
        if args.reward_scale <= 0.0 or args.huber_delta <= 0.0:
            raise ValueError("frozen reward scale and Huber delta must be positive")
        validate_v7_hybrid_target_entropy(args.target_entropy)
    repositories = {
        "h2oplus": audit_repository("h2oplus", args.h2oplus_source),
        "rlpd": audit_repository("rlpd", args.rlpd_source),
        "wsrl": audit_repository("wsrl", args.wsrl_source),
    }
    paper_paths = {
        "h2oplus": args.h2oplus_paper,
        "rlpd": args.rlpd_paper,
        "wsrl": args.wsrl_paper,
    }
    papers = {}
    for name, path in paper_paths.items():
        observed = sha256(path)
        papers[name] = {
            "path": str(path.resolve()),
            "expected_sha256": EXPECTED_PAPERS[name],
            "observed_sha256": observed,
            "matches": observed == EXPECTED_PAPERS[name],
        }
    profiles = {
        name: audit_profile(name) for name in EXPECTED_PROFILES
    }
    local_files = source_hashes("v7")
    for path in (
        Path(__file__).resolve(),
        ROOT / "reproducibility" / "run_reward_loss_pilot_v7.py",
        ROOT / "reproducibility" / "run_hybrid_entropy_pilot_v7.py",
        ROOT / "reproducibility" / "audit_device_cache_precision_v6.py",
        ROOT / "reproducibility" / "audit_gpu_capacity_v7.py",
        ROOT / "reproducibility" / "build_training_admission_v7.py",
        ROOT / "reproducibility" / "build_training_plan_v7.py",
        ROOT / "reproducibility" / "build_training_dispatch_v6.py",
        ROOT / "reproducibility" / "build_training_dispatch_v7.py",
        ROOT / "reproducibility" / "deployment_bundle_v7.py",
        ROOT / "reproducibility" / "run_training_matrix_v4.py",
        ROOT / "reproducibility" / "run_training_dispatch_v6.py",
        ROOT / "reproducibility" / "run_training_dispatch_v7.py",
        ROOT / "reproducibility" / "audit_training_matrix_v6.py",
        ROOT / "tests" / "test_v3_learning.py",
        ROOT / "tests" / "test_v7_networks.py",
        ROOT / "tests" / "test_build_training_plan_v7.py",
        ROOT / "tests" / "test_deployment_bundle_v7.py",
        ROOT / "tests" / "test_gpu_capacity_v7.py",
    ):
        local_files[str(path.relative_to(ROOT))] = sha256(path)
    passed = (
        all(item["passed"] for item in repositories.values())
        and all(item["matches"] for item in papers.values())
        and all(item["passed"] for item in profiles.values())
    )
    payload = {
        "schema_version": "transit-method-protocol-audit-v7-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "source and paper mapping for H2O+, state-based RLPD, and the "
            "CQL-initialized locomotion variant of WSRL"
        ),
        "verdict": "passed" if passed else "failed",
        "repositories": repositories,
        "papers": papers,
        "profiles": profiles,
        "local_files": local_files,
        "evidence_map": {
            "h2oplus": [
                "SimpleSAC/h2oplus.py:27-44,78-79,255-281,315-371,418-459",
                "SimpleSAC/h2o+_main.py:50-55,75-78,232-255",
                "paper equations 7-8 and lambda statement/ablation",
            ],
            "rlpd": [
                "train_finetuning.py:36-53,131-138,153-195",
                "configs/rlpd_config.py:7-9",
                "rlpd/agents/sac/sac_learner.py:154-180,194-272",
                "paper Algorithm 1 and Appendix B.2 Tables 1-2",
            ],
            "wsrl": [
                "README.md:66-91",
                "finetune.py:31-65,277-365",
                "experiments/configs/ensemble_config.py:4-14",
                "experiments/configs/train_config.py:225-276",
                "experiments/configs/wsrl_config.py:7-17",
                "experiments/configs/cql_config.py:8-31",
                "experiments/scripts/locomotion/launch_cql_finetune.sh:1-12",
                "experiments/scripts/locomotion/launch_wsrl_finetune.sh:1-16",
                "wsrl/agents/sac.py:181-310,624-682",
                "wsrl/agents/cql.py:25-256,339-360",
                "paper Appendix B and Appendix I",
            ],
        },
        "task_required_adaptations": [
            "chronological event-driven transit SMDP with continuous-time discount",
            "hybrid zero-or-positive hold distribution and its counting-plus-Lebesgue entropy",
            "shared v7 structured state encoder within each multi-head Q ensemble",
            "generalized passenger-time reward and a common target-entropy calibration gate",
        ],
        "task_adaptation_freeze": (
            {
                "reward_scale": args.reward_scale,
                "critic_loss": args.critic_loss,
                "huber_delta": args.huber_delta,
                "target_entropy": args.target_entropy,
                "device_cache_observation_dtype": (
                    args.device_cache_observation_dtype
                ),
            }
            if freeze_complete
            else None
        ),
        "freeze_complete": freeze_complete,
        "unresolved_before_formal_training": (
            []
            if freeze_complete
            else [
                "freeze the shared v7 hybrid target entropy from pre-registered pilots",
                "freeze reward scaling from offline numerical conditioning",
                "select MSE or shared Huber TD loss from stability pilots",
                "freeze FP16 or FP32 replay storage from the precision audit",
            ]
        ),
    }
    write_json(args.output, payload)
    if not passed:
        raise SystemExit("method protocol audit failed")


if __name__ == "__main__":
    main()
