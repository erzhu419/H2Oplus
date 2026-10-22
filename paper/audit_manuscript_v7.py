#!/usr/bin/env python3
"""Semantic release gate for the transit v7 manuscript.

This complements ``audit_manuscript.py``. The older audit checks TeX structure,
citations, labels, and acronym form, but cannot tell whether a valid number or
equation describes the executable v7 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_manuscript import expand_inputs, math_atoms, strip_comments  # noqa: E402
from bus_h2o.mdp_contract_v5 import RAW_EVENT_STATE_DIM  # noqa: E402
from bus_h2o.mdp_contract_v7 import (  # noqa: E402
    CANONICAL_PREFIX_NAMES,
    CANONICAL_SLOT_WIDTH,
    CANONICAL_STATE_DIM,
    DEFAULT_CONFIG_V7,
    LOCAL_EVENT_STATE_DIM,
    MAX_SIMULTANEOUS_DECISIONS,
    RAW_STATE_DIM,
)


MAIN = ROOT / "main.tex"
REFERENCES = ROOT / "references.bib"
LEDGER = ROOT / "TERMINOLOGY_LEDGER_V7.md"
INTERVENTION_FIDELITY = (
    PROJECT_ROOT
    / "bus_h2o"
    / "calibrated_env"
    / "intervention_fidelity_audit_v7_formal_v3.json"
)
SECTION_FILES = tuple(sorted((ROOT / "sections").glob("*.tex")))
SOURCE_FILES = (
    MAIN,
    ROOT / "highlights.tex",
    ROOT / "math_commands.tex",
    *SECTION_FILES,
)
REFERENCE_PLACEMENT_REQUIREMENTS = {
    "niu2022h2o": (
        "sections/1_introduction.tex",
        "sections/2_related_work.tex",
        "sections/3_problem.tex",
    ),
    "niu2025h2oplus": (
        "sections/1_introduction.tex",
        "sections/2_related_work.tex",
        "sections/3_problem.tex",
        "sections/4_method.tex",
        "sections/5_experiments.tex",
    ),
    "eysenbach2021off": (
        "sections/2_related_work.tex",
        "sections/4_method.tex",
    ),
    "haarnoja2018soft": (
        "sections/2_related_work.tex",
        "sections/4_method.tex",
        "sections/5_experiments.tex",
    ),
    "kumar2020cql": (
        "sections/2_related_work.tex",
        "sections/5_experiments.tex",
    ),
    "kostrikov2022iql": (
        "sections/2_related_work.tex",
        "sections/4_method.tex",
        "sections/5_experiments.tex",
    ),
    "lee2022balanced": ("sections/2_related_work.tex",),
    "nakamoto2023calql": (
        "sections/2_related_work.tex",
        "sections/5_experiments.tex",
    ),
    "zhang2023pex": ("sections/2_related_work.tex",),
    "ball2023efficient": (
        "sections/2_related_work.tex",
        "sections/5_experiments.tex",
    ),
    "zhou2025wsrl": (
        "sections/2_related_work.tex",
        "sections/5_experiments.tex",
    ),
    "hu2025rgcql": ("sections/2_related_work.tex",),
    "sutton1999between": ("sections/3_problem.tex",),
    "rezazada2024review": (
        "sections/1_introduction.tex",
        "sections/2_related_work.tex",
    ),
    "daganzo2009headway": (
        "sections/1_introduction.tex",
        "sections/2_related_work.tex",
    ),
    "xuan2011dynamic": (
        "sections/1_introduction.tex",
        "sections/2_related_work.tex",
    ),
    "delgado2012holding": (
        "sections/1_introduction.tex",
        "sections/2_related_work.tex",
    ),
    "sanchezmartinez2016realtime": ("sections/2_related_work.tex",),
    "andres2017predictive": ("sections/2_related_work.tex",),
    "alesiani2018reinforcement": (
        "sections/1_introduction.tex",
        "sections/2_related_work.tex",
    ),
    "wang2020real": (
        "sections/1_introduction.tex",
        "sections/2_related_work.tex",
    ),
    "wang2023multiobjective": (
        "sections/1_introduction.tex",
        "sections/2_related_work.tex",
    ),
    "lopez2018microscopic": ("sections/3_problem.tex",),
}
REFERENCE_CITATION_COUNTS = {
    "niu2022h2o": 3,
    "niu2025h2oplus": 6,
    "eysenbach2021off": 2,
    "haarnoja2018soft": 3,
    "kumar2020cql": 3,
    "kostrikov2022iql": 3,
    "lee2022balanced": 1,
    "nakamoto2023calql": 2,
    "zhang2023pex": 1,
    "ball2023efficient": 2,
    "zhou2025wsrl": 2,
    "hu2025rgcql": 1,
    "sutton1999between": 1,
    "rezazada2024review": 2,
    "daganzo2009headway": 2,
    "xuan2011dynamic": 2,
    "delgado2012holding": 2,
    "sanchezmartinez2016realtime": 1,
    "andres2017predictive": 1,
    "alesiani2018reinforcement": 2,
    "wang2020real": 2,
    "wang2023multiobjective": 2,
    "lopez2018microscopic": 1,
}
REFERENCE_BIB_SHA256 = (
    "87461f3391a11cf1e5cc40e6787a9462ec91699dc11725907c73b0fa98456e82"
)
REFERENCE_PLACEMENT_SHA256 = (
    "8f1775e56d7c0b1c4c21a0169b83ff92921fc0ff05f3a43f72846ae25afe333d"
)
INTERVENTION_TABLE_ROWS = (
    ("Passenger completion rate", "completion_rate"),
    ("Control decisions", "control_decisions"),
    ("Exact-zero rate", "exact_zero_rate"),
    ("Generalized passenger time", "generalized_seconds_per_person"),
    ("In-vehicle time", "in_vehicle_seconds_per_person"),
    ("Mean hold duration", "mean_hold_seconds"),
    ("Completed vehicle-trip duration", "trip_duration_seconds"),
    ("Waiting time", "wait_seconds_per_person"),
)


@dataclass(frozen=True)
class Finding:
    check: str
    message: str
    path: str | None = None
    line: int | None = None


@dataclass(frozen=True)
class ProhibitedPattern:
    name: str
    pattern: str
    explanation: str


@dataclass(frozen=True)
class FirstDefinitionRule:
    name: str
    use_pattern: str
    definition_pattern: str
    explanation: str


PROHIBITED_PATTERNS = (
    ProhibitedPattern(
        "old-canonical-width",
        r"(?<![0-9])275(?:-dimensional|\s*&)|\\R\^\{275\}",
        "v7 canonical state width is 229",
    ),
    ProhibitedPattern(
        "old-row-count",
        r"(?<![0-9])(?:2,831,150|2\{,\}831\{,\}150)(?![0-9])",
        "the final v7 row count must come from the frozen HDF5 audit",
    ),
    ProhibitedPattern(
        "old-episode-count",
        r"(?<![0-9])210[^.]{0,80}(?:episodes|trajectories)",
        "the planned v7 matrix has 240 episodes and the audited count is pending",
    ),
    ProhibitedPattern(
        "old-policy-count",
        r"\bseven\s+(?:deliberately\s+diverse\s+)?behavior\s+polic(?:y|ies)\b",
        "the v7 collection design has eight behavior policies",
    ),
    ProhibitedPattern(
        "old-prefix-width",
        r"\bthree(?:-value|\s+(?:AEC|batch))?\s+prefix\b",
        "v7 has a five-value canonical batch prefix",
    ),
    ProhibitedPattern(
        "old-local-event-width",
        r"x_\{m,j\}\\in\\R\^\{16\}",
        "v7 separates the 16-feature raw event from the 13-feature local event vector",
    ),
    ProhibitedPattern(
        "old-reward-unit-symbol",
        r"c_\{\\mathrm\{r\}\}",
        "v7 stores physical passenger-second reward and has no data reward divisor",
    ),
    ProhibitedPattern(
        "old-entropy-target",
        r"target\s+entropy\s*(?:is|of|=)?\s*\$?-1(?:\.0)?\$?",
        "the admitted hybrid-policy entropy target is 0",
    ),
    ProhibitedPattern(
        "old-critic-loss",
        r"\b(?:simulator\s+)?squared\s+residuals?\b",
        "the admitted critic regression is Huber with delta 1",
    ),
    ProhibitedPattern(
        "pending-action-endpoint-count",
        r"\b790\s+rows?\s+at\s+the\s+exact\s+upper\s+endpoint\b",
        "action endpoint counts must come from the frozen full-v7 dataset audit",
    ),
    ProhibitedPattern(
        "old-dataset-size",
        r"\b599,255,824\s+bytes\b",
        "dataset byte size must come from the frozen full-v7 HDF5",
    ),
    ProhibitedPattern(
        "old-dataset-digest",
        r"da6b5ebd5dfd4ce582907d5bb009c6741b385e9d7ac75bfcbe310754abced709",
        "dataset digest must identify the frozen full-v7 HDF5",
    ),
    ProhibitedPattern(
        "old-zero-duration-count",
        r"\b853,810\s+zero-duration\b",
        "zero-duration count must come from the frozen full-v7 dataset audit",
    ),
    ProhibitedPattern(
        "old-zero-action-rate",
        r"\b63\.55\\?%",
        "exact-zero action rate must come from the frozen full-v7 dataset audit",
    ),
    ProhibitedPattern(
        "unqualified-half-cache",
        r"\b(?:offline\s+(?:tensor\s+)?)?cache(?:\s+storage)?\s+(?:in|is\s+stored\s+in)?\s*half\s+precision\b|"
        r"offline\s+cache\s+storage\s*&\s*half\s+precision",
        "cache dtype is pending the full-HDF5 precision and capacity gates",
    ),
    ProhibitedPattern(
        "unscaled-training-reward",
        r"rewards?\s+(?:are|is)\s+consumed\s+as\s+stored\s+without\s+(?:additional\s+)?scal(?:e|ing)",
        "training applies the admitted numerical reward scale",
    ),
    ProhibitedPattern(
        "v6-label",
        r"(?:eq:v6_|alg:h2oplus_v6)",
        "final v7 labels may not retain v6 identifiers",
    ),
    ProhibitedPattern(
        "v6-status-prose",
        r"\bformal\s+v6\b|\bv6\s+(?:policy\s+)?outcomes?\b|\bv6\s+runtime\b",
        "final status prose must identify frozen v7 artifacts",
    ),
    ProhibitedPattern(
        "v19-runtime-prose",
        r"\bv19\s+(?:local\s+and\s+server\s+execution\s+gates|deployment\s+bundle)\b",
        "final runtime prose must identify the frozen v7 admission and deployment",
    ),
    ProhibitedPattern(
        "local-h2oplus-placement",
        r"H2O\+[^.\n]{0,200}\blocal\s+(?:graphics\s+processing\s+unit\s*\(GPU\)|GPU)\b|"
        r"\blocal\s+(?:graphics\s+processing\s+unit\s*\(GPU\)|GPU)\b[^.\n]{0,200}H2O\+",
        "canonical H2O+ training is server-only",
    ),
)


REQUIRED_BODY_PATTERNS = (
    (
        "canonical-width",
        r"\b229-dimensional\s+canonical\b",
        "state the 229-dimensional canonical state",
    ),
    (
        "raw-event-width",
        r"\b16-(?:feature|dimensional)\s+raw\s+(?:SUMO\s+)?event\b",
        "distinguish the 16-feature raw event observation",
    ),
    (
        "local-event-width",
        r"\b13-(?:feature|dimensional)\s+(?:deduplicated\s+)?local\s+(?:event\s+)?(?:observation|vector)\b",
        "state the 13-feature deduplicated local event vector",
    ),
    (
        "prefix-width",
        r"\bfive(?:-value|\s+value)?\s+(?:canonical\s+)?(?:batch\s+)?prefix\b",
        "state the five-value canonical batch prefix",
    ),
    (
        "physical-reward-unit",
        r"\bphysical\s+(?:generalized\s+)?passenger-seconds?\b",
        "state the physical unit of stored reward",
    ),
    (
        "training-reward-scale",
        r"(?:reward|passenger-second)[^.\n]{0,100}(?:10\^\{-4\}|10\^\{-?4\})|"
        r"(?:10\^\{-4\}|10\^\{-?4\})[^.\n]{0,100}(?:reward|passenger-second)",
        "state the admitted 1e-4 training reward scale",
    ),
    (
        "huber-loss",
        r"\bHuber\b",
        "state the admitted Huber critic loss",
    ),
    (
        "huber-delta",
        r"Huber[^.\n]{0,100}(?:delta|\\delta)[^0-9.\n]{0,30}1(?:\.0)?\b|"
        r"(?:delta|\\delta)[^0-9.\n]{0,30}1(?:\.0)?[^.\n]{0,100}Huber",
        "state Huber delta 1",
    ),
    (
        "hybrid-entropy-target",
        r"target\s+entropy[^.\n]{0,40}(?:is|of|=)?\s*\$?0(?:\.0)?\$?",
        "state the admitted hybrid-policy target entropy of 0",
    ),
    (
        "episode-count",
        r"\b240[^.\n]{0,80}(?:complete\s+)?episodes\b",
        "state the planned and audited v7 episode count",
    ),
    (
        "eight-policies",
        r"\beight\s+behavior\s+policies\b",
        "state the eight-policy v7 offline design",
    ),
    (
        "server-h2oplus-placement",
        r"H2O\+[^.]{0,240}\bGPU\s+servers?\b|"
        r"\bGPU\s+servers?\b[^.]{0,240}H2O\+",
        "state that H2O+ runs only on registered GPU servers",
    ),
    (
        "server-baseline-placement",
        r"(?:WSRL|RLPD)[^.]{0,240}\bGPU\s+servers?\b|"
        r"\bGPU\s+servers?\b[^.]{0,240}(?:WSRL|RLPD)",
        "state that WSRL/RLPD run on GPU servers",
    ),
    (
        "h2oplus-update-order",
        r"actor[^.]{0,200}\bprecede\w*[^.]{0,100}critic\s+optimizer",
        "state that the H2O+ actor update precedes the critic optimizer step",
    ),
)


FIRST_DEFINITION_RULES = (
    FirstDefinitionRule(
        "stored-reward",
        r"r_k",
        r"The stored dataset reward is\s*\\begin\{equation\}\s*r_k=-C_k",
        "r_k must first appear in the definition of the stored physical reward",
    ),
    FirstDefinitionRule(
        "latent-normal-mean",
        r"\\mu_\{\\phi\}",
        r"Let\s+\$\\mu_\{\\phi\}\(s\)\$\s+and\s+"
        r"\$\\sigma_\{\\phi\}\(s\)>0\$\s+be the\s+"
        r"actor-produced latent-normal mean and standard deviation",
        "mu_phi must first appear where the actor-produced latent-normal mean is defined",
    ),
    FirstDefinitionRule(
        "latent-normal-standard-deviation",
        r"\\sigma_\{\\phi\}",
        r"Let\s+\$\\mu_\{\\phi\}\(s\)\$\s+and\s+"
        r"\$\\sigma_\{\\phi\}\(s\)>0\$\s+be the\s+"
        r"actor-produced latent-normal mean and standard deviation",
        "sigma_phi must first appear where the positive latent-normal standard deviation is defined",
    ),
    FirstDefinitionRule(
        "critic-parameter-collections",
        r"(?<!\\bar)\\theta(?!\s*_)",
        r"\$\\theta=\(\\theta_1,\\theta_2\)\$\s+and\s+"
        r"\$\\bar\\theta=\(\\bar\\theta_1,\\bar\\theta_2\)\$\s+collect the "
        r"online- and target-critic\s+parameter vectors, respectively",
        "theta must first appear in the online-critic parameter-collection definition",
    ),
    FirstDefinitionRule(
        "target-critic-parameter-collection",
        r"\\bar\\theta(?!\s*_)",
        r"\$\\theta=\(\\theta_1,\\theta_2\)\$\s+and\s+"
        r"\$\\bar\\theta=\(\\bar\\theta_1,\\bar\\theta_2\)\$\s+collect the "
        r"online- and target-critic\s+parameter vectors, respectively",
        "bar-theta must first appear in the target-critic parameter-collection definition",
    ),
    FirstDefinitionRule(
        "value-parameter",
        r"\\nu",
        r"one value\s+function\s+\$V_\{\\nu\}\$, where "
        r"\$\\nu\$\s+is the value-function parameter vector",
        "nu must first appear where the value-function parameter vector is defined",
    ),
    FirstDefinitionRule(
        "ratio-parameter-collection",
        r"\\psi(?!\s*_)",
        r"\$\\psi=\(\\psi_J,\\psi_M\)\$\s+denotes the ordered collection of "
        r"both classifier\s+parameter vectors",
        "psi must first appear in the two-classifier parameter-collection definition",
    ),
    FirstDefinitionRule(
        "polyak-coefficient",
        r"\\kappa_\{\\mathrm\{tgt\}\}",
        r"Polyak averaging with coefficient\s+"
        r"\$\\kappa_\{\\mathrm\{tgt\}\}=0\.005\$",
        "kappa_tgt must first appear where the Polyak coefficient is defined",
    ),
)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def executable_contract_findings() -> tuple[dict[str, object], list[Finding]]:
    observed: dict[str, object] = {
        "raw_event_state_dim": RAW_EVENT_STATE_DIM,
        "local_event_state_dim": LOCAL_EVENT_STATE_DIM,
        "raw_state_dim": RAW_STATE_DIM,
        "canonical_state_dim": CANONICAL_STATE_DIM,
        "canonical_prefix_width": len(CANONICAL_PREFIX_NAMES),
        "canonical_slot_width": CANONICAL_SLOT_WIDTH,
        "max_simultaneous_decisions": MAX_SIMULTANEOUS_DECISIONS,
        "reward_unit_passenger_seconds": DEFAULT_CONFIG_V7.reward_unit_passenger_seconds,
        "horizon_seconds": DEFAULT_CONFIG_V7.horizon_seconds,
        "max_hold_seconds": DEFAULT_CONFIG_V7.max_hold_seconds,
    }
    expected: dict[str, object] = {
        "raw_event_state_dim": 16,
        "local_event_state_dim": 13,
        "raw_state_dim": 245,
        "canonical_state_dim": 229,
        "canonical_prefix_width": 5,
        "canonical_slot_width": 14,
        "max_simultaneous_decisions": 16,
        "reward_unit_passenger_seconds": 1.0,
        "horizon_seconds": 24_000.0,
        "max_hold_seconds": 60.0,
    }
    findings = [
        Finding(
            check="executable-contract",
            message=f"{name}: expected {expected[name]!r}, observed {value!r}",
        )
        for name, value in observed.items()
        if value != expected[name]
    ]
    return observed, findings


def prohibited_source_findings() -> list[Finding]:
    findings: list[Finding] = []
    for path in SOURCE_FILES:
        if not path.is_file():
            findings.append(
                Finding(
                    check="missing-manuscript-source",
                    message="registered manuscript source is absent",
                    path=relative(path),
                )
            )
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        for rule in PROHIBITED_PATTERNS:
            for match in re.finditer(rule.pattern, text, flags=re.IGNORECASE):
                findings.append(
                    Finding(
                        check=rule.name,
                        message=rule.explanation,
                        path=relative(path),
                        line=line_number(text, match.start()),
                    )
                )
    return findings


def required_body_findings(source: str) -> list[Finding]:
    findings: list[Finding] = []
    for name, pattern, explanation in REQUIRED_BODY_PATTERNS:
        if not re.search(pattern, source, flags=re.IGNORECASE | re.DOTALL):
            findings.append(check_missing(name, explanation))
    return findings


def first_definition_findings(source: str) -> list[Finding]:
    findings: list[Finding] = []
    for rule in FIRST_DEFINITION_RULES:
        first_use = re.search(rule.use_pattern, source)
        definition = re.search(
            rule.definition_pattern,
            source,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if first_use is None:
            findings.append(
                Finding(
                    check=f"missing-symbol-use-{rule.name}",
                    message=rule.explanation,
                )
            )
            continue
        if definition is None:
            findings.append(
                Finding(
                    check=f"missing-symbol-definition-{rule.name}",
                    message=rule.explanation,
                )
            )
            continue
        if not (definition.start() <= first_use.start() < definition.end()):
            findings.append(
                Finding(
                    check=f"symbol-before-definition-{rule.name}",
                    message=rule.explanation,
                )
            )
    return findings


def release_wiring_findings() -> list[Finding]:
    findings: list[Finding] = []
    structural_audit = ROOT / "audit_manuscript.py"
    structural_text = structural_audit.read_text(encoding="utf-8")
    if "TERMINOLOGY_LEDGER_V7.md" not in structural_text:
        findings.append(
            Finding(
                check="legacy-structural-ledger-wiring",
                message="the final structural audit must use TERMINOLOGY_LEDGER_V7.md",
                path=relative(structural_audit),
            )
        )
    math_commands = ROOT / "math_commands.tex"
    if re.search(
        r"\bv6\b|TERMINOLOGY_LEDGER_V6",
        math_commands.read_text(encoding="utf-8"),
    ):
        findings.append(
            Finding(
                check="legacy-math-command-wiring",
                message="math command provenance still identifies the v6 notation contract",
                path=relative(math_commands),
            )
        )
    return findings


def citation_release_findings() -> list[Finding]:
    findings: list[Finding] = []
    bibliography_bytes = REFERENCES.read_bytes()
    bibliography_digest = hashlib.sha256(bibliography_bytes).hexdigest()
    if bibliography_digest != REFERENCE_BIB_SHA256:
        findings.append(
            Finding(
                check="reference-metadata-registry-drift",
                message=(
                    "references.bib changed after the 23-entry authoritative "
                    "metadata audit; reverify every changed record before "
                    "updating the registered digest"
                ),
                path=relative(REFERENCES),
            )
        )
    bibliography_keys = set(
        re.findall(
            r"@\w+\s*\{\s*([^,\s]+)",
            bibliography_bytes.decode("utf-8"),
        )
    )
    citations_by_source: dict[str, set[str]] = {}
    citation_counts: dict[str, int] = {}
    placement_rows: list[tuple[str, int, int, str, str]] = []
    for path in SOURCE_FILES:
        text = strip_comments(path.read_text(encoding="utf-8"))
        source_name = relative(path)
        if source_name.startswith("paper/"):
            source_name = source_name[len("paper/") :]
        for paragraph_index, paragraph in enumerate(re.split(r"\n\s*\n", text)):
            normalized_paragraph = " ".join(paragraph.split())
            paragraph_groups = re.findall(
                r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\]\s*){0,2}\{([^}]+)\}",
                paragraph,
            )
            for group_index, group in enumerate(paragraph_groups):
                for key in (item.strip() for item in group.split(",")):
                    if key:
                        placement_rows.append(
                            (
                                source_name,
                                paragraph_index,
                                group_index,
                                key,
                                normalized_paragraph,
                            )
                        )
        cited_groups = re.findall(
            r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\]\s*){0,2}\{([^}]+)\}",
            text,
        )
        cited_occurrences = [
            key.strip()
            for group in cited_groups
            for key in group.split(",")
            if key.strip()
        ]
        keys = set(cited_occurrences)
        for key in cited_occurrences:
            citation_counts[key] = citation_counts.get(key, 0) + 1
        citations_by_source[source_name] = keys

    placement_payload = json.dumps(
        placement_rows,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    placement_digest = hashlib.sha256(placement_payload).hexdigest()
    if placement_digest != REFERENCE_PLACEMENT_SHA256:
        findings.append(
            Finding(
                check="reference-semantic-placement-drift",
                message=(
                    "a reviewed citation paragraph or citation position changed; "
                    "recheck the supported claim against the primary source before "
                    "updating the registered placement digest"
                ),
                path="paper",
            )
        )

    registered_keys = set(REFERENCE_PLACEMENT_REQUIREMENTS)
    if bibliography_keys != registered_keys:
        missing = sorted(bibliography_keys - registered_keys)
        stale = sorted(registered_keys - bibliography_keys)
        findings.append(
            Finding(
                check="incomplete-reference-placement-registry",
                message=(
                    f"unregistered bibliography keys={missing}; "
                    f"registry keys absent from bibliography={stale}"
                ),
                path=relative(REFERENCES),
            )
        )
    if set(REFERENCE_CITATION_COUNTS) != registered_keys:
        findings.append(
            Finding(
                check="incomplete-reference-count-registry",
                message="citation-count and placement registries must contain identical keys",
                path=relative(REFERENCES),
            )
        )

    for key, required_sources in REFERENCE_PLACEMENT_REQUIREMENTS.items():
        if key not in bibliography_keys:
            findings.append(
                Finding(
                    check="missing-v7-reference",
                    message=f"source-verified required bibliography record is absent: {key}",
                    path=relative(REFERENCES),
                )
            )
            continue
        expected_sources = set(required_sources)
        actual_sources = {
            source
            for source, source_keys in citations_by_source.items()
            if key in source_keys
        }
        if actual_sources != expected_sources:
            findings.append(
                Finding(
                    check="invalid-v7-citation-placement",
                    message=(
                        f"{key}: expected sources={sorted(expected_sources)}, "
                        f"observed sources={sorted(actual_sources)}"
                    ),
                    path=relative(REFERENCES),
                )
            )
        expected_count = REFERENCE_CITATION_COUNTS.get(key)
        observed_count = citation_counts.get(key, 0)
        if observed_count != expected_count:
            findings.append(
                Finding(
                    check="invalid-v7-citation-count",
                    message=(
                        f"{key}: expected {expected_count} reviewed placements, "
                        f"observed {observed_count}"
                    ),
                    path=relative(REFERENCES),
                )
            )
    return findings


def intervention_fidelity_findings() -> tuple[dict[str, object], list[Finding]]:
    """Bind the manuscript's worst-case intervention table to its artifact."""

    findings: list[Finding] = []
    if not INTERVENTION_FIDELITY.is_file():
        return {}, [
            Finding(
                check="missing-intervention-fidelity-artifact",
                message="the formal intervention-fidelity audit is absent",
                path=relative(INTERVENTION_FIDELITY),
            )
        ]
    try:
        payload = json.loads(INTERVENTION_FIDELITY.read_text(encoding="utf-8"))
        comparisons = payload["comparison"]["comparisons"]
        policies = payload["design"]["policies"]
        matrix_jobs = int(payload["matrix_manifest"]["jobs"])
        sumo_rows = len(payload["sumo_rows"])
        simulator_episodes = len(payload["simulator_episodes"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {}, [
            Finding(
                check="invalid-intervention-fidelity-artifact",
                message=f"cannot parse the formal intervention audit: {error}",
                path=relative(INTERVENTION_FIDELITY),
            )
        ]

    if not (
        payload.get("schema_version")
        == "transit-simulator-intervention-fidelity-v7-formal-v1"
        and payload.get("study_stage") == "formal"
        and payload.get("passed") is True
        and payload.get("invariants_passed") is True
        and payload.get("comparison", {}).get("passed") is True
        and len(policies) == 8
        and len(comparisons) == 3
        and all(len(policy_rows) == 7 for policy_rows in comparisons.values())
        and matrix_jobs == 240
        and sumo_rows == 240
        and simulator_episodes == 240
    ):
        findings.append(
            Finding(
                check="invalid-intervention-fidelity-design",
                message=(
                    "the manuscript requires a passed 8-policy, 3-demand, "
                    "240-pair formal intervention audit"
                ),
                path=relative(INTERVENTION_FIDELITY),
            )
        )

    table_source = (ROOT / "sections" / "5_experiments.tex").read_text(
        encoding="utf-8"
    )
    table_match = re.search(
        r"\\label\{tab:intervention_fidelity\}(.*?)\\end\{table\}",
        strip_comments(table_source),
        flags=re.DOTALL,
    )
    if table_match is None:
        findings.append(
            Finding(
                check="missing-intervention-fidelity-table",
                message="the intervention-fidelity table cannot be isolated",
                path="paper/sections/5_experiments.tex",
            )
        )
        return {}, findings

    summary: dict[str, object] = {
        "artifact": relative(INTERVENTION_FIDELITY),
        "matrix_jobs": matrix_jobs,
        "sumo_rows": sumo_rows,
        "simulator_episodes": simulator_episodes,
        "strata_per_metric": 21,
        "metrics": {},
    }
    table = table_match.group(1)
    for label, metric_name in INTERVENTION_TABLE_ROWS:
        metric_rows = [
            policy["metrics"][metric_name]
            for policies_by_demand in comparisons.values()
            for policy in policies_by_demand.values()
        ]
        if len(metric_rows) != 21:
            findings.append(
                Finding(
                    check="incomplete-intervention-fidelity-strata",
                    message=f"{metric_name} has {len(metric_rows)} rather than 21 strata",
                    path=relative(INTERVENTION_FIDELITY),
                )
            )
            continue
        thresholds = {float(row["threshold"]) for row in metric_rows}
        if len(thresholds) != 1 or any(
            row.get("point_passed") is not True
            or row.get("interval_passed") is not True
            for row in metric_rows
        ):
            findings.append(
                Finding(
                    check="failed-intervention-fidelity-metric",
                    message=f"{metric_name} is failed or has inconsistent thresholds",
                    path=relative(INTERVENTION_FIDELITY),
                )
            )
            continue
        threshold = thresholds.pop()
        largest_point = max(
            abs(
                float(
                    row.get("normalized_response_gap", row.get("normalized_gap"))
                )
            )
            for row in metric_rows
        )
        largest_endpoint = max(
            abs(float(endpoint))
            for row in metric_rows
            for endpoint in row["gap_bootstrap_95_ci"]
        )
        summary["metrics"][metric_name] = {
            "threshold": threshold,
            "largest_absolute_point": largest_point,
            "largest_absolute_interval_endpoint": largest_endpoint,
        }
        row_match = re.search(
            rf"{re.escape(label)}\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)"
            rf"\s*&\s*([0-9.]+)\s*\\\\",
            table,
        )
        if row_match is None:
            findings.append(
                Finding(
                    check="missing-intervention-fidelity-row",
                    message=f"table row is absent or malformed: {label}",
                    path="paper/sections/5_experiments.tex",
                )
            )
            continue
        observed = tuple(float(value) for value in row_match.groups())
        expected = (
            float(f"{threshold:.3f}"),
            float(f"{largest_point:.5f}"),
            float(f"{largest_endpoint:.5f}"),
        )
        if any(
            not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12)
            for first, second in zip(observed, expected)
        ):
            findings.append(
                Finding(
                    check="stale-intervention-fidelity-row",
                    message=f"{label}: expected {expected}, observed {observed}",
                    path="paper/sections/5_experiments.tex",
                )
            )
    return summary, findings


def check_missing(name: str, explanation: str) -> Finding:
    return Finding(check=f"missing-{name}", message=explanation)


def ledger_findings(source: str) -> tuple[dict[str, int], list[Finding]]:
    if not LEDGER.exists():
        return {}, [
            Finding(
                check="missing-v7-ledger",
                message="create the strict one-symbol-one-meaning v7 ledger",
                path=relative(LEDGER),
            )
        ]

    text = LEDGER.read_text(encoding="utf-8")
    section = re.search(
        r"^## Canonical symbols\s*$\n(.*?)(?=^##\s|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if section is None:
        return {}, [
            Finding(
                check="invalid-v7-ledger",
                message="canonical-symbol section is absent",
                path=relative(LEDGER),
            )
        ]

    rows = re.findall(
        r"^\|\s*(\$.*?\$)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$",
        section.group(1),
        flags=re.MULTILINE,
    )
    rows = [row for row in rows if row[0] != "$Symbol$"]
    if not rows:
        return {}, [
            Finding(
                check="invalid-v7-ledger",
                message="no three-column canonical-symbol rows were parsed",
                path=relative(LEDGER),
            )
        ]

    findings: list[Finding] = []
    normalized_symbols = [re.sub(r"\s+", "", row[0]) for row in rows]
    normalized_meanings = [re.sub(r"\s+", " ", row[1]).strip().casefold() for row in rows]
    for symbol in sorted(set(normalized_symbols)):
        if normalized_symbols.count(symbol) > 1:
            findings.append(
                Finding(
                    check="duplicate-ledger-symbol",
                    message=f"canonical symbol is repeated: {symbol}",
                    path=relative(LEDGER),
                )
            )
    for meaning in sorted(set(normalized_meanings)):
        if normalized_meanings.count(meaning) > 1:
            findings.append(
                Finding(
                    check="duplicate-ledger-meaning",
                    message=f"canonical meaning is repeated: {meaning}",
                    path=relative(LEDGER),
                )
            )
    for symbol, _, _ in rows:
        content = symbol[1:-1]
        standalone_operators = {"=", ",", "<", ">", "+", "-", "/"}
        if content.strip() not in standalone_operators and (
            has_top_level_token(content, ",") or has_top_level_token(content, "=")
        ):
            findings.append(
                Finding(
                    check="grouped-ledger-symbol",
                    message=f"ledger row must contain one symbol, not an expression: {symbol}",
                    path=relative(LEDGER),
                )
            )
    inline_math = re.findall(r"(?<!\\)\$(.+?)(?<!\\)\$", source, flags=re.DOTALL)
    display_math = re.findall(
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
        source,
        flags=re.DOTALL,
    )
    source_atoms = math_atoms("\n".join((*inline_math, *display_math)))
    ledger_atoms = math_atoms("\n".join(row[0] for row in rows))
    unknown_atoms = sorted(source_atoms - ledger_atoms)
    if unknown_atoms:
        findings.append(
            Finding(
                check="unregistered-math-atoms",
                message=f"mathematical atoms absent from the v7 ledger: {unknown_atoms}",
                path=relative(LEDGER),
            )
        )
    return {"canonical_symbol_rows": len(rows)}, findings


def has_top_level_token(text: str, token: str) -> bool:
    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0
    escaped = False
    for character in text:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth = max(0, brace_depth - 1)
        elif character == "(":
            paren_depth += 1
        elif character == ")":
            paren_depth = max(0, paren_depth - 1)
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif (
            character == token
            and brace_depth == 0
            and paren_depth == 0
            and bracket_depth == 0
        ):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-report",
        type=Path,
        help="write the complete semantic audit report to this path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = expand_inputs(MAIN)
    contract, findings = executable_contract_findings()
    findings.extend(prohibited_source_findings())
    findings.extend(required_body_findings(source))
    findings.extend(first_definition_findings(source))
    findings.extend(release_wiring_findings())
    findings.extend(citation_release_findings())
    intervention_summary, intervention_errors = intervention_fidelity_findings()
    findings.extend(intervention_errors)
    ledger_summary, ledger_errors = ledger_findings(source)
    findings.extend(ledger_errors)

    report = {
        "schema_version": "manuscript-semantic-audit-v7",
        "status": "PASS" if not findings else "FAIL",
        "executable_contract": contract,
        "intervention_fidelity": intervention_summary,
        "ledger": ledger_summary,
        "finding_count": len(findings),
        "findings": [asdict(item) for item in findings],
    }
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if findings:
        print(f"V7 MANUSCRIPT SEMANTIC AUDIT FAILED: {len(findings)} finding(s)")
        for finding in findings:
            location = ""
            if finding.path is not None:
                location = f" {finding.path}"
                if finding.line is not None:
                    location += f":{finding.line}"
            print(f"- [{finding.check}]{location}: {finding.message}")
        return 1

    print(
        "V7 MANUSCRIPT SEMANTIC AUDIT PASSED: "
        f"state={CANONICAL_STATE_DIM}, symbols={ledger_summary['canonical_symbol_rows']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
