import json
from pathlib import Path

from reproducibility.export_evaluation_policies_v7 import (
    EXPORT_SCHEMA,
    MANIFEST_NAME,
)
from reproducibility.merge_evaluation_policy_exports_v7 import merge_exports
from reproducibility.run_evaluation_matrix_v4 import (
    METHODS_SCHEMAS,
    V7_POLICY_SELECTION_RULE,
    load_methods,
)
from SimpleSAC.train_transit_v3 import sha256
from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


def test_v7_policy_exports_merge_into_exact_formal_method_freeze(tmp_path):
    export_root = tmp_path / "server-export"
    actor_path = export_root / "policies" / "rlpd" / "seed3" / "final_actor.pt"
    actor_path.parent.mkdir(parents=True)

    class Replay:
        contract_config = {"horizon_seconds": 24_000.0}

    trainer = TransitH2OTrainerV7(
        Replay(),
        config=TransitTrainerConfig(
            hidden_dim=8,
            ensemble_size=2,
            batch_size=4,
        ),
        device="cpu",
        seed=3,
    )
    run_signature = "1" * 64
    admission_sha256 = "2" * 64
    training_checkpoint_sha256 = "3" * 64
    trainer.save_policy_artifact(
        actor_path,
        extra={
            "phase": "complete",
            "method": "rlpd",
            "training_seed": 3,
            "run_signature": run_signature,
            "training_checkpoint_sha256": training_checkpoint_sha256,
            "training_admission": {
                "schema_version": "transit-training-admission-v7-v2",
                "sha256": admission_sha256,
            },
        },
    )
    export_manifest = export_root / MANIFEST_NAME
    export_manifest.write_text(
        json.dumps(
            {
                "schema_version": EXPORT_SCHEMA,
                "selection_rule": V7_POLICY_SELECTION_RULE,
                "training_admission_sha256": admission_sha256,
                "evaluation_environment_design_sha256": "4" * 64,
                "methods": [
                    {
                        "name": "rlpd-seed3",
                        "algorithm": "rlpd",
                        "training_seed": 3,
                        "type": "checkpoint",
                        "checkpoint": str(actor_path.relative_to(export_root)),
                        "checkpoint_sha256": sha256(actor_path),
                        "training_run_signature": run_signature,
                        "training_admission_sha256": admission_sha256,
                        "training_checkpoint_sha256": (
                            training_checkpoint_sha256
                        ),
                        "training_plan_sha256": "5" * 64,
                    }
                ],
            }
        )
    )
    output = tmp_path / "evaluation_methods_v7_formal.json"
    payload = merge_exports(
        [export_manifest],
        output,
        expected_methods=("rlpd",),
        expected_seeds=(3,),
    )

    assert payload["learned_policy_count"] == 1
    assert payload["operational_reference_count"] == 6
    methods = load_methods(output, METHODS_SCHEMAS["v7"])
    assert len(methods) == 7
    assert {(method["type"], method.get("gain")) for method in methods[1:]} == {
        ("zero", None),
        ("headway-balance", 0.05),
        ("headway-balance", 0.1),
        ("headway-balance", 0.25),
        ("headway-balance", 0.5),
        ("myopic-welfare", None),
    }
