#!/usr/bin/env python3
"""Fail-fast structural audit for the v7 manuscript source."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.tex"
BIB = ROOT / "references.bib"
SYMBOL_LEDGER = ROOT / "TERMINOLOGY_LEDGER_V7.md"


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def expand_inputs(path: Path, stack: tuple[Path, ...] = ()) -> str:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise RuntimeError(f"cyclic TeX input: {cycle}")
    text = strip_comments(path.read_text(encoding="utf-8"))

    def replace(match: re.Match[str]) -> str:
        child = ROOT / match.group(1)
        if child.suffix == "":
            child = child.with_suffix(".tex")
        if not child.exists():
            raise FileNotFoundError(f"missing TeX input: {child}")
        return expand_inputs(child, (*stack, path))

    return re.sub(r"\\input\{([^}]+)\}", replace, text)


def split_csv(matches: list[str]) -> set[str]:
    return {
        item.strip()
        for match in matches
        for item in match.split(",")
        if item.strip()
    }


def check_acronyms(body: str) -> list[str]:
    checks = [
        ("RL", r"\breinforcement\s+learning\s*\(RL\)", r"\bRL\b"),
        ("H2O", r"Hybrid\s+Offline-and-Online\s*\(H2O\)", r"\bH2O(?!\+)\b"),
        (
            "SMDP",
            r"semi-Markov\s+decision\s+process\s*\(SMDP\)",
            r"\bSMDP\b",
        ),
        (
            "SUMO",
            r"Simulation\s+of\s+Urban\s+MObility\s*\(SUMO\)",
            r"\bSUMO\b",
        ),
        (
            "H2O+",
            r"improved\s+framework\s+for\s+Hybrid\s+Offline-and-Online\s+RL"
            r"\s+with\s+dynamics\s+gaps\s*\(H2O\+\)",
            r"\bH2O\+",
        ),
        ("IQL", r"implicit\s+Q-learning\s*\(IQL\)", r"\bIQL\b"),
        (
            "RLPD",
            r"Reinforcement\s+Learning\s+with\s+Prior\s+Data\s*\(RLPD\)",
            r"\bRLPD\b",
        ),
        (
            "WSRL",
            r"Warm-Start\s+Reinforcement\s+Learning\s*\(WSRL\)",
            r"\bWSRL\b",
        ),
        ("SAC", r"soft\s+actor-critic\s*\(SAC\)", r"\bSAC\b"),
        ("CQL", r"conservative\s+Q-learning\s*\(CQL\)", r"\bCQL\b"),
        (
            "Cal-QL",
            r"calibrated\s+Q-learning\s*\(Cal-QL\)",
            r"\bCal-QL\b",
        ),
        ("BC", r"behavior\s+cloning\s*\(BC\)", r"\bBC\b"),
        ("PEX", r"policy\s+expansion\s*\(PEX\)", r"\bPEX\b"),
        (
            "RG-CQL",
            r"Reward-Guided\s+Conservative\s+Q-Learning\s*\(RG-CQL\)",
            r"\bRG-CQL\b",
        ),
        ("UTD", r"update-to-data\s*\(UTD\)", r"\bUTD\b"),
        (
            "AEC",
            r"agent-environment\s+cycle\s*\(AEC\)",
            r"\bAEC\b",
        ),
        ("IS", r"importance\s+sampling\s*\(IS\)", r"\bIS\b"),
        (
            "GPU",
            r"graphics\s+processing\s+unit\s*\(GPU\)",
            r"\bGPU\b",
        ),
        (
            "AVL",
            r"Automatic\s+Vehicle\s+Location\s*\(AVL\)",
            r"\bAVL\b",
        ),
        (
            "APC",
            r"Automatic\s+Passenger\s+Counting\s*\(APC\)",
            r"\bAPC\b",
        ),
        (
            "AFC",
            r"Automated\s+Fare\s+Collection\s*\(AFC\)",
            r"\bAFC\b",
        ),
        (
            "AI",
            r"Artificial\s+Intelligence\s*\(AI\)",
            r"\bAI\b",
        ),
        (
            "CRediT",
            r"Contributor\s+Roles\s+Taxonomy\s*\(CRediT\)",
            r"\bCRediT\b",
        ),
        (
            "HDF5",
            r"Hierarchical\s+Data\s+Format\s+version\s+5\s*\(HDF5\)",
            r"\bHDF5\b",
        ),
        (
            "SHA-256",
            r"Secure\s+Hash\s+Algorithm\s+256-bit\s*\(SHA-256\)",
            r"\bSHA-256\b",
        ),
        (
            "CUDA",
            r"Compute\s+Unified\s+Device\s+Architecture\s*\(CUDA\)",
            r"\bCUDA\b",
        ),
    ]
    errors: list[str] = []
    for name, expansion_pattern, token_pattern in checks:
        first = re.search(token_pattern, body)
        expansions = list(re.finditer(expansion_pattern, body, flags=re.IGNORECASE))
        expansion = expansions[0] if expansions else None
        if first is None:
            continue
        elif expansion is None:
            errors.append(f"required first-use expansion is absent: {name}")
        elif not (expansion.start() <= first.start() < expansion.end()):
            errors.append(f"{name} appears before its required expansion")
        elif len(expansions) > 1:
            errors.append(f"full-form expansion is repeated after first use: {name}")

    command_free = re.sub(r"\\[A-Za-z@]+", " ", body)
    candidates = set(
        re.findall(
            r"(?<![A-Za-z0-9])(?:[A-Z]{2,}-[A-Z]{2,}|"
            r"[A-Z]{2,}[A-Z0-9]*(?:-\d+)?|"
            r"[A-Z][0-9][A-Z0-9]*\+?|[A-Z][a-z]+-[A-Z]{2,}"
            r")(?![A-Za-z0-9])",
            command_free,
        )
    )
    allowed = {name for name, _, _ in checks} | {"NVIDIA"}
    unknown = sorted(candidates - allowed)
    if unknown:
        errors.append(f"unregistered acronym-like tokens: {unknown}")
    return errors


def math_atoms(text: str) -> set[str]:
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", text)
    text = re.sub(r"\\label\{[^}]+\}", " ", text)
    text = re.sub(r"\\text\{[^}]*\}", " ", text)
    # TeX spacing lengths are layout metadata, not mathematical identifiers.
    text = re.sub(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
        r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em|mu)\b",
        " ",
        text,
    )
    commands = set(re.findall(r"\\([A-Za-z]+)", text))
    ignored_commands = {
        "begin",
        "end",
        "equation",
        "aligned",
        "frac",
        "tfrac",
        "left",
        "right",
        "mathbb",
        "mathcal",
        "mathrm",
        "operatorname",
        "qquad",
        "quad",
        "hspace",
        "vspace",
        "hskip",
        "vskip",
        "kern",
        "mkern",
        "text",
        "widetilde",
        "bar",
        "ldots",
    }
    commands -= ignored_commands
    without_commands = re.sub(r"\\[A-Za-z]+", " ", text)
    words = set(re.findall(r"(?<![A-Za-z])[A-Za-z]+(?![A-Za-z])", without_commands))
    return commands | words


def check_symbol_ledger(source: str) -> list[str]:
    ledger = SYMBOL_LEDGER.read_text(encoding="utf-8")
    rows = re.findall(
        r"^\|\s*(\$.*?\$)\s*\|\s*(.*?)\s*\|(?:\s*.*?\s*\|)?$",
        ledger,
        flags=re.MULTILINE,
    )
    errors: list[str] = []
    if not rows:
        return ["canonical-symbol ledger could not be parsed"]

    symbols = [re.sub(r"\s+", "", symbol) for symbol, _ in rows]
    meanings = [re.sub(r"\s+", " ", meaning.strip()).casefold() for _, meaning in rows]
    duplicate_symbols = sorted({item for item in symbols if symbols.count(item) > 1})
    duplicate_meanings = sorted({item for item in meanings if meanings.count(item) > 1})
    if duplicate_symbols:
        errors.append(f"duplicate canonical symbols: {duplicate_symbols}")
    if duplicate_meanings:
        errors.append(f"duplicate canonical symbol meanings: {duplicate_meanings}")

    inline_math = re.findall(r"(?<!\\)\$(.+?)(?<!\\)\$", source, flags=re.DOTALL)
    display_math = re.findall(
        r"\\begin\{equation\}(.*?)\\end\{equation\}", source, flags=re.DOTALL
    )
    source_atoms = math_atoms("\n".join((*inline_math, *display_math)))
    ledger_atoms = math_atoms("\n".join(symbol for symbol, _ in rows))
    unknown_atoms = sorted(source_atoms - ledger_atoms)
    if unknown_atoms:
        errors.append(f"math atoms absent from canonical-symbol ledger: {unknown_atoms}")
    return errors


def main() -> int:
    source = expand_inputs(MAIN)
    errors: list[str] = []
    errors.extend(check_symbol_ledger(source))

    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", BIB.read_text(encoding="utf-8")))
    cited_keys = split_csv(
        re.findall(
            r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\]\s*){0,2}\{([^}]+)\}",
            source,
        )
    )
    undefined_citations = sorted(cited_keys - bib_keys)
    unused_bibliography = sorted(bib_keys - cited_keys)
    if undefined_citations:
        errors.append(f"undefined citation keys: {undefined_citations}")
    if unused_bibliography:
        errors.append(f"uncited bibliography entries: {unused_bibliography}")

    labels = re.findall(r"\\label\{([^}]+)\}", source)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        errors.append(f"duplicate labels: {duplicate_labels}")
    referenced_labels = split_csv(
        re.findall(r"\\(?:[Cc]ref|ref|eqref|autoref)\{([^}]+)\}", source)
    )
    undefined_references = sorted(referenced_labels - set(labels))
    if undefined_references:
        errors.append(f"undefined cross-references: {undefined_references}")
    numbered_labels = {
        label
        for label in labels
        if label.startswith(("eq:", "fig:", "tab:", "alg:"))
    }
    unreferenced_numbered = sorted(numbered_labels - referenced_labels)
    if unreferenced_numbered:
        errors.append(f"unreferenced numbered objects: {unreferenced_numbered}")

    for asset in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
        asset_path = ROOT / asset
        candidates = [asset_path] if asset_path.suffix else [
            asset_path.with_suffix(suffix) for suffix in (".pdf", ".png", ".jpg")
        ]
        if not any(candidate.exists() for candidate in candidates):
            errors.append(f"missing figure asset: {asset}")

    abstract_start = source.find(r"\begin{abstract}")
    bibliography_start = source.find(r"\bibliographystyle")
    if abstract_start < 0 or bibliography_start < 0:
        errors.append("could not isolate manuscript body for acronym audit")
    else:
        body = source[abstract_start:bibliography_start]
        body = re.sub(
            r"\\begin\{keyword\}.*?\\end\{keyword\}", "", body, flags=re.DOTALL
        )
        errors.extend(check_acronyms(body))

        prohibited = [
            "real environment",
            r"\\mathcal\{M\}_\{\\mathrm\{real\}\}",
            r"P_\{\\mathrm\{real\}\}",
            "675,000",
            "18,000 s",
            "InfoNCE",
            "Q-floor",
            "LCB target",
            "200 GB",
            "headway-deviation reward",
            "reward-scale ratio",
            "five-critic H2O+",
        ]
        for pattern in prohibited:
            if re.search(pattern, body, flags=re.IGNORECASE):
                errors.append(f"prohibited pre-v6 term or value: {pattern}")

    if errors:
        print("MANUSCRIPT AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "MANUSCRIPT AUDIT PASSED: "
        f"{len(bib_keys)} citations, {len(labels)} labels, "
        f"{len(numbered_labels)} referenced numbered objects"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
