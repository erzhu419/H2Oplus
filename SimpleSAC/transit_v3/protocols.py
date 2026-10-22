"""Auditable method protocols for the canonical transit experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MethodProfile:
    description: str
    reference: str
    reference_implementation: str | None
    reference_commit: str | None
    offline_objective: str
    online_update_mode: str
    online_backup: str
    use_ratio_correction: bool
    ensemble_size: int
    target_subset_size: int
    batch_size: int
    utd_ratio: int
    actor_update_interval: int
    actor_batch_mode: str
    actor_q_reduction: str
    actor_learning_rate: float
    critic_learning_rate: float
    value_learning_rate: float
    alpha_learning_rate: float
    target_entropy: float
    initial_temperature: float
    backup_entropy: bool
    online_fraction: float
    bootstrap_probability: float
    offline_cql_alpha: float
    mixed_backup_lambda: float
    iql_expectile: float
    iql_advantage_temperature: float
    default_offline_updates: int
    default_warmup_events: int
    default_ratio_pretrain_updates: int
    ratio_update_interval: int = 4
    cql_max_target_backup: bool = False
    activation: str = "silu"
    actor_layer_norm: bool = True
    critic_layer_norm: bool = True
    value_layer_norm: bool = True
    offline_initializer: str | None = None
    gradient_clip_norm: float | None = None
    online_update_order: str = "critic-before-actor"
    warmup_action_mode: str = "policy"
    critic_loss: str = "mse"
    huber_delta: float = 1.0
    target_subset_replacement: bool = False
    cql_penalty_subset_size: int | None = None
    cql_penalty_subset_replacement: bool = False
    offline_update_order: str = "critic-before-actor"
    temperature_state_source: str = "actor"
    offline_temperature_action_samples: int = 1


H2OPLUS_REFERENCE = "https://doi.org/10.1109/ICRA55743.2025.11127503"
H2OPLUS_IMPLEMENTATION = "https://github.com/t6-thu/H2Oplus"
H2OPLUS_COMMIT = "02171768283d307244700c69003423704db83b05"
RLPD_REFERENCE = "https://proceedings.mlr.press/v202/ball23a.html"
SAC_REFERENCE = "https://proceedings.mlr.press/v80/haarnoja18b.html"
WSRL_REFERENCE = (
    "https://proceedings.iclr.cc/paper_files/paper/2025/hash/"
    "504491292cb71e7681eedfe0e602b72f-Abstract-Conference.html"
)


METHOD_PROFILES: dict[str, MethodProfile] = {
    "h2oplus": MethodProfile(
        description=(
            "transit-adapted H2O+: IQL warm start, frozen-policy simulator "
            "warmup, H2O+ mixed value backup, balanced target/simulator replay, "
            "and pretrained conditional dynamics-ratio correction"
        ),
        reference=H2OPLUS_REFERENCE,
        reference_implementation=H2OPLUS_IMPLEMENTATION,
        reference_commit=H2OPLUS_COMMIT,
        offline_objective="iql",
        online_update_mode="mixed",
        online_backup="h2oplus",
        use_ratio_correction=True,
        ensemble_size=2,
        target_subset_size=2,
        batch_size=256,
        utd_ratio=1,
        actor_update_interval=1,
        actor_batch_mode="latest",
        actor_q_reduction="min-all",
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_entropy=-1.0,
        initial_temperature=1.0,
        backup_entropy=True,
        online_fraction=0.5,
        bootstrap_probability=1.0,
        offline_cql_alpha=0.0,
        mixed_backup_lambda=0.1,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=50_000,
        default_warmup_events=5_000,
        default_ratio_pretrain_updates=1_000,
        activation="relu",
        actor_layer_norm=False,
        critic_layer_norm=False,
        value_layer_norm=False,
        online_update_order="actor-before-critic",
    ),
    "h2oplus-no-ratio": MethodProfile(
        description=(
            "transit-adapted H2O+ warm start and mixed value backup without "
            "conditional dynamics-ratio correction"
        ),
        reference=H2OPLUS_REFERENCE,
        reference_implementation=H2OPLUS_IMPLEMENTATION,
        reference_commit=H2OPLUS_COMMIT,
        offline_objective="iql",
        online_update_mode="mixed",
        online_backup="h2oplus",
        use_ratio_correction=False,
        ensemble_size=2,
        target_subset_size=2,
        batch_size=256,
        utd_ratio=1,
        actor_update_interval=1,
        actor_batch_mode="latest",
        actor_q_reduction="min-all",
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_entropy=-1.0,
        initial_temperature=1.0,
        backup_entropy=True,
        online_fraction=0.5,
        bootstrap_probability=1.0,
        offline_cql_alpha=0.0,
        mixed_backup_lambda=0.1,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=50_000,
        default_warmup_events=5_000,
        default_ratio_pretrain_updates=0,
        activation="relu",
        actor_layer_norm=False,
        critic_layer_norm=False,
        value_layer_norm=False,
        online_update_order="actor-before-critic",
    ),
    "ratio-sac": MethodProfile(
        description=(
            "CQL warm start and mixed SAC with conditional dynamics-ratio "
            "correction; retained as a non-H2O+ ablation"
        ),
        reference=H2OPLUS_REFERENCE,
        reference_implementation=H2OPLUS_IMPLEMENTATION,
        reference_commit=H2OPLUS_COMMIT,
        offline_objective="cql",
        online_update_mode="mixed",
        online_backup="sac",
        use_ratio_correction=True,
        ensemble_size=5,
        target_subset_size=2,
        batch_size=512,
        utd_ratio=1,
        actor_update_interval=1,
        actor_batch_mode="latest",
        actor_q_reduction="min-subset",
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_entropy=-1.0,
        initial_temperature=0.1,
        backup_entropy=True,
        online_fraction=0.5,
        bootstrap_probability=0.8,
        offline_cql_alpha=0.2,
        mixed_backup_lambda=0.0,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=50_000,
        default_warmup_events=5_000,
        default_ratio_pretrain_updates=1_000,
    ),
    "sac-mixed": MethodProfile(
        description=(
            "randomly initialized SAC matched to the transit H2O+ actor and "
            "critic architecture, with balanced target/simulator replay"
        ),
        reference=SAC_REFERENCE,
        reference_implementation=None,
        reference_commit=None,
        offline_objective="none",
        online_update_mode="mixed",
        online_backup="sac",
        use_ratio_correction=False,
        ensemble_size=2,
        target_subset_size=2,
        batch_size=256,
        utd_ratio=1,
        actor_update_interval=1,
        actor_batch_mode="latest",
        actor_q_reduction="min-all",
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_entropy=-1.0,
        initial_temperature=1.0,
        backup_entropy=True,
        online_fraction=0.5,
        bootstrap_probability=1.0,
        offline_cql_alpha=0.0,
        mixed_backup_lambda=0.0,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=0,
        default_warmup_events=5_000,
        default_ratio_pretrain_updates=0,
        activation="relu",
        actor_layer_norm=False,
        critic_layer_norm=False,
        value_layer_norm=False,
        warmup_action_mode="uniform-box",
    ),
    "rlpd": MethodProfile(
        description=(
            "source-aligned state-based RLPD update protocol: random "
            "initialization, 50/50 prior/online replay, ten Q heads, and UTD 20"
        ),
        reference=RLPD_REFERENCE,
        reference_implementation="https://github.com/ikostrikov/rlpd",
        reference_commit="c90fd4baf28c9c9ef40a81460a2e395092844f88",
        offline_objective="none",
        online_update_mode="mixed",
        online_backup="sac",
        use_ratio_correction=False,
        ensemble_size=10,
        target_subset_size=2,
        batch_size=256,
        utd_ratio=20,
        actor_update_interval=20,
        actor_batch_mode="latest",
        actor_q_reduction="mean-all",
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_entropy=-0.5,
        initial_temperature=1.0,
        backup_entropy=True,
        online_fraction=0.5,
        bootstrap_probability=1.0,
        offline_cql_alpha=0.0,
        mixed_backup_lambda=0.0,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=0,
        default_warmup_events=5_000,
        default_ratio_pretrain_updates=0,
        activation="relu",
        actor_layer_norm=False,
        critic_layer_norm=True,
        value_layer_norm=False,
        warmup_action_mode="uniform-box",
    ),
    "wsrl": MethodProfile(
        description=(
            "WSRL fine-tuning with the paper-supported CQL initialization "
            "variant: frozen-policy warmup, online-only replay, ten Q heads, "
            "and critic UTD 4"
        ),
        reference=WSRL_REFERENCE,
        reference_implementation="https://github.com/zhouzypaul/wsrl",
        reference_commit="ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2",
        offline_objective="cql",
        online_update_mode="online-only",
        online_backup="sac",
        use_ratio_correction=False,
        ensemble_size=10,
        target_subset_size=2,
        target_subset_replacement=True,
        cql_penalty_subset_size=2,
        cql_penalty_subset_replacement=True,
        batch_size=256,
        utd_ratio=4,
        actor_update_interval=4,
        actor_batch_mode="aggregate",
        actor_q_reduction="min-all",
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=1e-4,
        target_entropy=0.0,
        initial_temperature=1.0,
        backup_entropy=False,
        online_fraction=0.0,
        bootstrap_probability=1.0,
        offline_cql_alpha=5.0,
        mixed_backup_lambda=0.0,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=250_000,
        default_warmup_events=5_000,
        default_ratio_pretrain_updates=0,
        cql_max_target_backup=True,
        activation="relu",
        actor_layer_norm=True,
        critic_layer_norm=True,
        value_layer_norm=False,
        offline_initializer="cql",
        offline_update_order="actor-before-critic",
        temperature_state_source="next-observations",
        offline_temperature_action_samples=10,
    ),
    "sac-online": MethodProfile(
        description="randomly initialized online-only Soft Actor-Critic",
        reference=SAC_REFERENCE,
        reference_implementation=None,
        reference_commit=None,
        offline_objective="none",
        online_update_mode="online-only",
        online_backup="sac",
        use_ratio_correction=False,
        ensemble_size=2,
        target_subset_size=2,
        batch_size=256,
        utd_ratio=1,
        actor_update_interval=1,
        actor_batch_mode="latest",
        actor_q_reduction="min-all",
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_entropy=-1.0,
        initial_temperature=1.0,
        backup_entropy=True,
        online_fraction=0.0,
        bootstrap_probability=1.0,
        offline_cql_alpha=0.0,
        mixed_backup_lambda=0.0,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=0,
        default_warmup_events=5_000,
        default_ratio_pretrain_updates=0,
        activation="relu",
        actor_layer_norm=False,
        critic_layer_norm=False,
        value_layer_norm=False,
    ),
    "cql": MethodProfile(
        description="offline-only Conservative Q-Learning",
        reference="https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html",
        reference_implementation=None,
        reference_commit=None,
        offline_objective="cql",
        online_update_mode="none",
        online_backup="sac",
        use_ratio_correction=False,
        ensemble_size=2,
        target_subset_size=2,
        batch_size=256,
        utd_ratio=1,
        actor_update_interval=1,
        actor_batch_mode="latest",
        actor_q_reduction="min-all",
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_entropy=-1.0,
        initial_temperature=1.0,
        backup_entropy=True,
        online_fraction=0.0,
        bootstrap_probability=1.0,
        offline_cql_alpha=5.0,
        mixed_backup_lambda=0.0,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=250_000,
        default_warmup_events=0,
        default_ratio_pretrain_updates=0,
        cql_max_target_backup=True,
        activation="relu",
        actor_layer_norm=False,
        critic_layer_norm=False,
        value_layer_norm=False,
    ),
    "iql": MethodProfile(
        description="offline-only Implicit Q-Learning",
        reference="https://openreview.net/forum?id=68n2s9ZJWF8",
        reference_implementation=None,
        reference_commit=None,
        offline_objective="iql",
        online_update_mode="none",
        online_backup="sac",
        use_ratio_correction=False,
        ensemble_size=2,
        target_subset_size=2,
        batch_size=256,
        utd_ratio=1,
        actor_update_interval=1,
        actor_batch_mode="latest",
        actor_q_reduction="min-all",
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_entropy=-1.0,
        initial_temperature=1.0,
        backup_entropy=True,
        online_fraction=0.0,
        bootstrap_probability=1.0,
        offline_cql_alpha=0.0,
        mixed_backup_lambda=0.0,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=250_000,
        default_warmup_events=0,
        default_ratio_pretrain_updates=0,
        activation="relu",
        actor_layer_norm=False,
        critic_layer_norm=False,
        value_layer_norm=False,
    ),
    "bc": MethodProfile(
        description="offline-only maximum-likelihood behavior cloning",
        reference="https://proceedings.mlr.press/v15/ross11a.html",
        reference_implementation=None,
        reference_commit=None,
        offline_objective="bc",
        online_update_mode="none",
        online_backup="sac",
        use_ratio_correction=False,
        ensemble_size=2,
        target_subset_size=2,
        batch_size=256,
        utd_ratio=1,
        actor_update_interval=1,
        actor_batch_mode="latest",
        actor_q_reduction="min-all",
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        value_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_entropy=-1.0,
        initial_temperature=1.0,
        backup_entropy=True,
        online_fraction=0.0,
        bootstrap_probability=1.0,
        offline_cql_alpha=0.0,
        mixed_backup_lambda=0.0,
        iql_expectile=0.7,
        iql_advantage_temperature=3.0,
        default_offline_updates=100_000,
        default_warmup_events=0,
        default_ratio_pretrain_updates=0,
        activation="relu",
        actor_layer_norm=False,
        critic_layer_norm=False,
        value_layer_norm=False,
    ),
}

METHOD_PROFILES["h2oplus-lambda0"] = replace(
    METHOD_PROFILES["h2oplus"],
    description=(
        "IQL warm start and conditional dynamics-ratio correction with the "
        "H2O+ mixed-backup coefficient fixed to zero"
    ),
    mixed_backup_lambda=0.0,
)
METHOD_PROFILES["h2oplus-lambda0p5"] = replace(
    METHOD_PROFILES["h2oplus"],
    description=(
        "transit-adapted H2O+ with the mixed-backup coefficient fixed to 0.5 "
        "for source-default and sensitivity analysis"
    ),
    mixed_backup_lambda=0.5,
)
METHOD_PROFILES["h2oplus-lambda0p2"] = replace(
    METHOD_PROFILES["h2oplus"],
    description=(
        "transit-adapted H2O+ with the mixed-backup coefficient fixed to 0.2 "
        "for published-grid sensitivity analysis"
    ),
    mixed_backup_lambda=0.2,
)
METHOD_PROFILES["h2oplus-lambda1"] = replace(
    METHOD_PROFILES["h2oplus"],
    description=(
        "transit-adapted H2O+ with the mixed-backup coefficient fixed to 1.0 "
        "for published-grid sensitivity analysis"
    ),
    mixed_backup_lambda=1.0,
)
METHOD_PROFILES["h2oplus-no-warmstart"] = replace(
    METHOD_PROFILES["h2oplus"],
    description=(
        "H2O+ mixed value backup and conditional dynamics-ratio correction "
        "from random initialization, without IQL warm start"
    ),
    offline_objective="none",
    default_offline_updates=0,
)
METHOD_PROFILES["h2oplus-source"] = replace(
    METHOD_PROFILES["h2oplus"],
    description=(
        "paper/source-aligned H2O+ control: no separate offline warm start, "
        "one initial simulator rollout, online value learning from target data, "
        "and per-update conditional dynamics-ratio training"
    ),
    offline_objective="none",
    iql_expectile=0.5,
    iql_advantage_temperature=1.0,
    default_offline_updates=0,
    default_warmup_events=1_000,
    default_ratio_pretrain_updates=0,
    ratio_update_interval=1,
)


def get_method_profile(name: str) -> MethodProfile:
    try:
        return METHOD_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown method profile {name!r}") from error
