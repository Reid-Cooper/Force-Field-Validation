#!/usr/bin/env python3
# getMetrics.py flags
# -------------------
# --system-dir PATH: FAIRMD analysis/system directory to inspect. Required.
# --output-title NAME: Name of the output folder. Defaults to the system folder name.
# --output-root PATH: Parent directory for the output folder. Defaults to this Scripts directory.
# --python EXE: Python executable used for scripts that are launched as subprocesses.
# --metrics LIST: Comma/space separated metrics to run. Use all, bending, surface_tension,
#   tail_dihedral, apl, or order_parameters. Defaults to all.
# --only LIST: Backward-compatible alias for --metrics.
# --skip LIST: Comma/space separated metrics to skip after --metrics/--only is applied.
# --apl-json PATH: Override detected apl.json for bending modulus and APL plotting.
# --lipids-per-leaflet INT: Override detected lipid count for bending modulus.
# --bilayer-thickness-nm FLOAT: Override detected thickness.json value for bending modulus.
# --edr PATH: Override detected .edr energy file for surface tension.
# --gmx EXE: GROMACS command used by calc_surface_tension.py. Use auto to search
#   PATH, the Python/conda env, and nearby conda envs. Defaults to auto.
# --surface-length-unit UNIT: Length unit passed to calc_surface_tension.py.
# --surface-pressure-unit UNIT: Pressure unit passed to calc_surface_tension.py.
# --surface-output-unit UNIT: Surface tension unit written by calc_surface_tension.py.
# --surface-precision INT: Decimal precision for surface tension output.
# --topology PATH: Override detected md_2.tpr/md.tpr/conf.gro for tail dihedral ratios.
# --trajectory PATH: Add a trajectory file for tail dihedral ratios. Can be repeated.
# --lipid-select TEXT: Atom selection for lipid residues in tail dihedral ratios.
# --tail-mode MODE: Tail source for ratios: auto, guess, charmm_popc, openff_popc, or manual.
# --tail DEF: Tail definition such as sn1:C3[1-16]. Can be repeated; implies manual tails.
# --window-deg FLOAT: Window half-width for dihedral state classification. Always used with
#   --state-definition window when running tail dihedral ratios.
# --start INT: First frame index for tail dihedral ratios.
# --stop INT: Stop-before frame index for tail dihedral ratios.
# --step INT: Frame step for tail dihedral ratios.
# --progress-every INT: Print tail-dihedral progress every N frames. 0 disables progress.
# --apl-time-unit UNIT: Time unit in apl.json for APL plotting, ps or ns.
# --apl-production-start-ns FLOAT: Drop APL frames before this time in ns.
# --order-json PATH: Order-parameter JSON to plot. Can be repeated; overrides autodetected
#   simulation order-parameter files when provided.
# --simulation-label LABEL: Base label for autodetected simulation order-parameter files.
# --order-label LABEL: Label for an order-parameter dataset. Can be repeated by dataset order.
# --order-color COLOR: Matplotlib color for an order-parameter dataset. Can be repeated.
# --include-experimental-reference: Add the first experimental order-parameter reference
#   discovered through README.yaml, or use --experimental-order-json when supplied.
# --experimental-order-json PATH: Explicit experimental order-parameter JSON for plotting.
# --experimental-label LABEL: Legend label for the experimental reference; defaults to the
#   README.yaml reference path when autodetected.
"""
Flag-driven FAIRMD metric runner.

This script lives beside the FAIRMD metric scripts and gathers the inputs
needed to run them. Results, plots, and logs are written into one user-named
output folder.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar


SCRIPTS_DIR = Path(__file__).resolve().parent
T = TypeVar("T")

SCRIPT_FILES = {
    "bending": "calc_bending_modulus.py",
    "surface_tension": "calc_surface_tension.py",
    "tail_dihedral": "calc_tail_dihedral_ratios.py",
    "apl": "graphAPL.py",
    "order_parameters": "JH_OrderParameterPlotting.py",
}

METRIC_ORDER = [
    "bending",
    "surface_tension",
    "tail_dihedral",
    "apl",
    "order_parameters",
]

METRIC_LABELS = {
    "bending": "bending modulus",
    "surface_tension": "surface tension",
    "tail_dihedral": "tail dihedral ratios",
    "apl": "area-per-lipid plot",
    "order_parameters": "order parameter plot",
}

METRIC_ALIASES = {
    "bending": "bending",
    "bending_modulus": "bending",
    "calc_bending_modulus": "bending",
    "surface": "surface_tension",
    "surface_tension": "surface_tension",
    "calc_surface_tension": "surface_tension",
    "tail": "tail_dihedral",
    "tail_dihedral": "tail_dihedral",
    "tail_dihedral_ratios": "tail_dihedral",
    "calc_tail_dihedral_ratios": "tail_dihedral",
    "dihedral": "tail_dihedral",
    "apl": "apl",
    "graphapl": "apl",
    "graph_apl": "apl",
    "graphAPL": "apl",
    "op": "order_parameters",
    "order": "order_parameters",
    "order_parameters": "order_parameters",
    "order_parameter_plotting": "order_parameters",
    "jh_orderparameterplotting": "order_parameters",
    "jh_orderparametersplotting": "order_parameters",
    "JH_OrderParameterPlotting": "order_parameters",
    "JH_OrderParametersPlotting": "order_parameters",
}

CHARMM_POPC_TAILS = ["sn1:C3[1-16]", "sn2:C2[1-18]"]
OPENFF_POPC_TAILS = [
    "sn1:C27x,C28x,C29x,C30x,C31x,C32x,C33x,C34x,C35x,C36x,C37x,C38x,C39x,C40x,C41x,C42x",
    "sn2:C8x,C9x,C10x,C11x,C12x,C13x,C14x,C15x,C16x,C17x,C18x,C19x,C20x,C21x,C22x,C23x,C24x,C25x",
]

PLOT_COLORS = [
    "#015480",
    "#F08521",
    "#3B26C4",
    "#53C411",
    "#C41111",
    "#8BC34A",
    "black",
]

GMX_AUTO_VALUES = {"", "auto", "gmx"}
GMX_DEFAULT_NAMES = ["gmx", "gmx_mpi"]


@dataclass
class MetricResult:
    name: str
    status: str
    artifacts: list[Path]
    message: str = ""


@dataclass
class FairmdSystem:
    path: Path
    apl_json: Path | None
    thickness_json: Path | None
    readme_yaml: Path | None
    topology: Path | None
    structure: Path | None
    trajectory: Path | None
    edr: Path | None
    order_parameter_jsons: list[Path]
    lipid_resname: str | None
    lipids_per_leaflet: int | None
    thickness_nm: float | None
    tail_mode: str


def sanitize_folder_name(title: str) -> str:
    name = re.sub(r"[\x00-\x1f/\\:]+", "_", title.strip())
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name or f"metrics_{datetime.now():%Y%m%d_%H%M%S}"


def resolve_path(value: str | Path) -> Path:
    """Resolve CLI paths from the current directory, then from the Scripts folder."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (SCRIPTS_DIR / path).resolve()


def existing_path(value: str | Path | None, description: str) -> Path | None:
    """Convert a flag value into a real path and fail early if it is missing."""
    if value is None:
        return None
    path = resolve_path(value)
    if not path.exists():
        raise RuntimeError(f"{description} was not found: {path}")
    return path


def require_value(value: T | None, message: str) -> T:
    """Raise a clear CLI error when autodetection and overrides both fail."""
    if value is None:
        raise RuntimeError(message)
    return value


def first_existing(base_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = base_dir / name
        if path.exists():
            return path
    return None


def first_glob(base_dir: Path, patterns: list[str], exclude: tuple[str, ...] = ()) -> Path | None:
    for pattern in patterns:
        for path in sorted(base_dir.glob(pattern)):
            lower_name = path.name.lower()
            if any(token in lower_name for token in exclude):
                continue
            if path.exists():
                return path
    return None


def find_order_parameter_jsons(base_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(base_dir.glob("*OrderParameters.json")):
        lower_name = path.name.lower()
        if "quality" in lower_name or "fragment" in lower_name:
            continue
        paths.append(path)
    return paths


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_second_column(data: object) -> float | None:
    if not isinstance(data, list):
        return None
    values: list[float] = []
    for row in data:
        if isinstance(row, (list, tuple)) and len(row) >= 2 and is_number(row[1]):
            values.append(float(row[1]))
        else:
            return None
    if not values:
        return None
    return statistics.fmean(values)


def find_numeric_series_or_value(data: object, preferred_keys: tuple[str, ...]) -> float | None:
    row_mean = mean_second_column(data)
    if row_mean is not None:
        return row_mean

    if is_number(data):
        return float(data)

    if isinstance(data, dict):
        for key, value in data.items():
            key_text = str(key).lower()
            if any(token in key_text for token in preferred_keys):
                direct = find_numeric_series_or_value(value, preferred_keys)
                if direct is not None:
                    return direct
        for value in data.values():
            nested = find_numeric_series_or_value(value, preferred_keys)
            if nested is not None:
                return nested

    if isinstance(data, list):
        numbers = [float(value) for value in data if is_number(value)]
        if numbers and len(numbers) == len(data):
            return statistics.fmean(numbers)
        for item in data:
            nested = find_numeric_series_or_value(item, preferred_keys)
            if nested is not None:
                return nested

    return None


def load_xy_series_json(path: Path) -> tuple[list[float], list[float]]:
    data = read_json(path)

    if isinstance(data, dict):
        for x_key in ("time", "times", "Time", "x"):
            for y_key in ("apl", "area", "values", "data", "y"):
                x_value = data.get(x_key)
                y_value = data.get(y_key)
                if isinstance(x_value, list) and isinstance(y_value, list) and len(x_value) == len(y_value):
                    if all(is_number(value) for value in x_value) and all(is_number(value) for value in y_value):
                        return [float(value) for value in x_value], [float(value) for value in y_value]
        for key in ("data", "values", "apl", "area_per_lipid"):
            if key in data:
                try:
                    return load_xy_series_from_object(data[key])
                except ValueError:
                    pass

    return load_xy_series_from_object(data)


def load_xy_series_from_object(data: object) -> tuple[list[float], list[float]]:
    if not isinstance(data, list):
        raise ValueError("Expected a list of [time, value] rows.")

    x_values: list[float] = []
    y_values: list[float] = []
    for row in data:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError("Expected every row to have at least two columns.")
        if not is_number(row[0]) or not is_number(row[1]):
            raise ValueError("Expected numeric time/value columns.")
        x_values.append(float(row[0]))
        y_values.append(float(row[1]))
    if not x_values:
        raise ValueError("No rows found.")
    return x_values, y_values


def parse_gro_residue_counts(gro_path: Path) -> dict[str, int]:
    counts: dict[str, set[str]] = {}
    try:
        lines = gro_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    for line in lines[2:-1]:
        if len(line) < 10:
            continue
        resid = line[:5].strip()
        resname = line[5:10].strip()
        if not resid or not resname:
            continue
        counts.setdefault(resname, set()).add(resid)
    return {resname: len(resids) for resname, resids in counts.items()}


def infer_lipid_resname(system_dir: Path, structure: Path | None, order_jsons: list[Path]) -> str | None:
    for path in order_jsons:
        match = re.match(r"([A-Za-z0-9]+)_?OrderParameters$", path.stem)
        if match:
            return match.group(1)

    if structure is not None and structure.suffix.lower() == ".gro":
        ignored = {
            "SOL",
            "WAT",
            "HOH",
            "TIP3",
            "TIP3P",
            "NA",
            "CL",
            "K",
            "CA",
            "MG",
        }
        counts = parse_gro_residue_counts(structure)
        candidates = [(resname, count) for resname, count in counts.items() if resname.upper() not in ignored]
        if candidates:
            return sorted(candidates, key=lambda item: item[1], reverse=True)[0][0]

    readme = system_dir / "README.yaml"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"\b(POPC|POPE|POPG|DPPC|DMPC|DLPC|DOPC|DSPC)\b", text)
        if match:
            return match.group(1)

    return None


def infer_lipids_per_leaflet(structure: Path | None, lipid_resname: str | None) -> int | None:
    if structure is None or structure.suffix.lower() != ".gro" or lipid_resname is None:
        return None
    counts = parse_gro_residue_counts(structure)
    total = counts.get(lipid_resname)
    if total is None:
        total = counts.get(lipid_resname.upper()) or counts.get(lipid_resname.lower())
    if total is None or total <= 0:
        return None
    return max(1, round(total / 2))


def infer_thickness_nm(thickness_json: Path | None) -> float | None:
    if thickness_json is None:
        return None
    try:
        raw_value = find_numeric_series_or_value(
            read_json(thickness_json),
            ("thickness", "bilayer", "mean", "average", "value"),
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if raw_value is None:
        return None
    # Bilayer thickness is commonly around 3-5 nm or 30-50 A.
    return raw_value / 10.0 if raw_value > 15.0 else raw_value


def infer_tail_mode(structure: Path | None, lipid_resname: str | None) -> str:
    if structure is None or structure.suffix.lower() != ".gro" or lipid_resname is None:
        return "guess"
    try:
        lines = structure.read_text(encoding="utf-8", errors="replace").splitlines()[2:-1]
    except OSError:
        return "guess"

    atom_names: set[str] = set()
    for line in lines:
        if len(line) < 15:
            continue
        if line[5:10].strip() != lipid_resname:
            continue
        atom_names.add(line[10:15].strip())
        if len(atom_names) > 80:
            break

    if {"C27x", "C28x", "C8x", "C9x"}.issubset(atom_names):
        return "openff_popc"
    if {"C31", "C32", "C21", "C22"}.issubset(atom_names):
        return "charmm_popc"
    return "guess"


def bilayerdata_root(system_dir: Path) -> Path | None:
    for path in (system_dir, *system_dir.parents):
        if path.name == "BilayerData":
            return path
        nested = path / "BilayerData"
        if nested.is_dir():
            return nested
        if (path / "experiments").is_dir() and (path / "Simulations").is_dir():
            return path
    return None


def strip_yaml_scalar(value: str) -> str:
    return value.strip().strip("\"'")


def parse_readme_orderparameter_refs(
    readme_path: Path | None,
    lipid_resname: str | None,
) -> list[tuple[str, str]]:
    if readme_path is None or not readme_path.exists():
        return []

    lines = readme_path.read_text(encoding="utf-8", errors="replace").splitlines()
    refs: list[tuple[str, str]] = []
    in_orderparameter = False
    orderparameter_indent = 0
    current_lipid: str | None = None
    wanted_lipid = lipid_resname.upper() if lipid_resname else None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if stripped == "ORDERPARAMETER:":
            in_orderparameter = True
            orderparameter_indent = indent
            current_lipid = None
            continue

        if not in_orderparameter:
            continue

        if indent <= orderparameter_indent:
            break

        lipid_match = re.match(r"^([A-Za-z0-9_+\-]+):\s*(.*)$", stripped)
        if lipid_match:
            current_lipid = lipid_match.group(1)
            trailing_value = lipid_match.group(2).strip()
            if trailing_value and trailing_value != "[]":
                values = re.findall(r"['\"]?([^,'\"\[\]\s]+/[^,'\"\[\]\s]+(?:/[^,'\"\[\]\s]+)?)['\"]?", trailing_value)
                for value in values:
                    if wanted_lipid is None or current_lipid.upper() == wanted_lipid:
                        refs.append((current_lipid, strip_yaml_scalar(value)))
            continue

        if current_lipid is None:
            continue

        list_match = re.match(r"^-\s*(.+)$", stripped)
        if list_match and (wanted_lipid is None or current_lipid.upper() == wanted_lipid):
            value = strip_yaml_scalar(list_match.group(1))
            if value and value != "[]":
                refs.append((current_lipid, value))

    return refs


def resolve_experimental_order_parameter_paths(system: FairmdSystem) -> list[tuple[Path, str]]:
    # README.yaml stores experiment references relative to BilayerData. This
    # resolver walks back to that root, then tests the common FAIRMD experiment
    # locations and keeps the README reference string for labeling.
    root = bilayerdata_root(system.path)
    if root is None:
        return []

    refs = parse_readme_orderparameter_refs(system.readme_yaml, system.lipid_resname)
    resolved: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for lipid, ref in refs:
        clean_ref = strip_yaml_scalar(ref).strip("/")
        if not clean_ref:
            continue

        ref_path = Path(clean_ref)
        candidate_dirs = [
            root / "experiments" / "OrderParameters" / ref_path,
            root / "experiments" / ref_path,
        ]
        candidate_files = [
            root / "experiments" / "OrderParameters" / ref_path / f"{lipid}_OrderParameters.json",
            root / "experiments" / ref_path / f"{lipid}_OrderParameters.json",
        ]

        for candidate in candidate_files:
            if candidate.exists() and candidate not in seen:
                resolved.append((candidate, clean_ref))
                seen.add(candidate)

        for candidate_dir in candidate_dirs:
            if not candidate_dir.is_dir():
                continue
            for candidate in sorted(candidate_dir.glob(f"{lipid}_OrderParameters.json")):
                if candidate.exists() and candidate not in seen:
                    resolved.append((candidate, clean_ref))
                    seen.add(candidate)
            if any(path == candidate for path, _label in resolved):
                continue
            for candidate in sorted(candidate_dir.glob("*OrderParameters.json")):
                if candidate.exists() and candidate not in seen:
                    resolved.append((candidate, clean_ref))
                    seen.add(candidate)

    return resolved


def discover_fairmd_system(system_dir: Path) -> FairmdSystem:
    if not system_dir.exists() or not system_dir.is_dir():
        raise ValueError(f"System directory does not exist or is not a directory: {system_dir}")

    # Collect the common FAIRMD outputs first. Later runners can override these
    # with flags, but the normal path is zero extra file flags for a complete run.
    apl_json = first_existing(system_dir, ["apl.json", "APL.json"])
    thickness_json = first_existing(system_dir, ["thickness.json", "Thickness.json"])
    readme_yaml = first_existing(system_dir, ["README.yaml", "README.yml"])
    topology = first_existing(system_dir, ["md_2.tpr", "md.tpr", "topol.tpr", "production.tpr"])
    structure = first_existing(system_dir, ["conf.gro", "conf.pdb", "structure.gro", "system.gro"])
    trajectory = first_existing(
        system_dir,
        ["md_every_100_ps.xtc", "centered.xtc", "whole.xtc", "md.xtc", "traj.xtc"],
    )
    edr = first_glob(system_dir, ["*.edr", "**/*.edr"])
    order_jsons = find_order_parameter_jsons(system_dir)
    lipid_resname = infer_lipid_resname(system_dir, structure, order_jsons)
    lipids_per_leaflet = infer_lipids_per_leaflet(structure, lipid_resname)
    thickness_nm = infer_thickness_nm(thickness_json)
    tail_mode = infer_tail_mode(structure, lipid_resname)

    return FairmdSystem(
        path=system_dir,
        apl_json=apl_json,
        thickness_json=thickness_json,
        readme_yaml=readme_yaml,
        topology=topology,
        structure=structure,
        trajectory=trajectory,
        edr=edr,
        order_parameter_jsons=order_jsons,
        lipid_resname=lipid_resname,
        lipids_per_leaflet=lipids_per_leaflet,
        thickness_nm=thickness_nm,
        tail_mode=tail_mode,
    )


def describe_path(path: Path | None) -> str:
    return str(path) if path is not None else "not detected"


def print_system_detection(system: FairmdSystem) -> None:
    print("\nDetected FAIRMD inputs:")
    print(f"  system directory: {system.path}")
    print(f"  apl.json: {describe_path(system.apl_json)}")
    print(f"  thickness.json: {describe_path(system.thickness_json)}")
    print(f"  topology: {describe_path(system.topology)}")
    print(f"  structure: {describe_path(system.structure)}")
    print(f"  trajectory: {describe_path(system.trajectory)}")
    print(f"  energy file: {describe_path(system.edr)}")
    print(f"  lipid residue: {system.lipid_resname or 'not detected'}")
    print(f"  lipids per leaflet: {system.lipids_per_leaflet or 'not detected'}")
    print(f"  thickness nm: {system.thickness_nm if system.thickness_nm is not None else 'not detected'}")
    if system.order_parameter_jsons:
        print("  order-parameter JSONs:")
        for path in system.order_parameter_jsons:
            print(f"    {path}")
    else:
        print("  order-parameter JSONs: not detected")


def write_detected_inputs(output_dir: Path, system: FairmdSystem) -> Path:
    # The detection report goes into info/ so generated metrics, plots, and run
    # metadata stay separated in predictable folders.
    path = info_dir(output_dir) / "detected_fairmd_inputs.txt"
    lines = [
        "Detected FAIRMD Inputs",
        "======================",
        f"System directory: {system.path}",
        f"apl.json: {describe_path(system.apl_json)}",
        f"thickness.json: {describe_path(system.thickness_json)}",
        f"README.yaml: {describe_path(system.readme_yaml)}",
        f"topology: {describe_path(system.topology)}",
        f"structure: {describe_path(system.structure)}",
        f"trajectory: {describe_path(system.trajectory)}",
        f"energy file: {describe_path(system.edr)}",
        f"lipid residue: {system.lipid_resname or 'not detected'}",
        f"lipids per leaflet: {system.lipids_per_leaflet or 'not detected'}",
        f"thickness nm: {system.thickness_nm if system.thickness_nm is not None else 'not detected'}",
        f"tail mode: {system.tail_mode}",
        "order-parameter JSONs:",
    ]
    if system.order_parameter_jsons:
        lines.extend(f"  {path}" for path in system.order_parameter_jsons)
    else:
        lines.append("  not detected")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_metric_list(value: str | None) -> set[str]:
    if not value:
        return set()
    metrics: set[str] = set()
    for raw_item in re.split(r"[,;\s]+", value):
        if not raw_item:
            continue
        key = raw_item.strip()
        if key.lower() == "all":
            metrics.update(METRIC_ORDER)
            continue
        normalized = key.lower().replace("-", "_").replace(".py", "")
        metric = METRIC_ALIASES.get(key) or METRIC_ALIASES.get(normalized)
        if metric is None:
            valid = ", ".join(METRIC_ORDER)
            raise SystemExit(f"Unknown metric {key!r}. Valid metric keys: {valid}")
        metrics.add(metric)
    return metrics


def select_metrics(args: argparse.Namespace) -> list[str]:
    """Resolve --metrics/--only/--skip into the fixed execution order."""
    selected = parse_metric_list(args.only or args.metrics)
    if not selected:
        selected = set(METRIC_ORDER)
    skip = parse_metric_list(args.skip)
    return [metric for metric in METRIC_ORDER if metric in selected and metric not in skip]


def script_path(metric: str) -> Path:
    path = SCRIPTS_DIR / SCRIPT_FILES[metric]
    if not path.exists():
        raise FileNotFoundError(f"Expected script not found: {path}")
    return path


def empirical_dir(output_dir: Path) -> Path:
    return output_dir / "empirical_results"


def graphs_dir(output_dir: Path) -> Path:
    return output_dir / "graphs"


def info_dir(output_dir: Path) -> Path:
    return output_dir / "info"


def ensure_output_dirs(output_dir: Path) -> None:
    # The folder split matches the requested layout:
    # empirical_results/ for text metrics, graphs/ for plots/csv, info/ for logs.
    output_dir.mkdir(parents=True, exist_ok=True)
    empirical_dir(output_dir).mkdir(parents=True, exist_ok=True)
    graphs_dir(output_dir).mkdir(parents=True, exist_ok=True)
    info_dir(output_dir).mkdir(parents=True, exist_ok=True)


def append_log(output_dir: Path, text: str) -> None:
    log_path = info_dir(output_dir) / "getMetrics_run_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def append_completed_process_log(
    output_dir: Path,
    *,
    label: str,
    command: list[str],
    cwd: Path,
    completed: subprocess.CompletedProcess[str],
) -> None:
    append_log(
        output_dir,
        f"""
        ================================================================================
        [{datetime.now():%Y-%m-%d %H:%M:%S}] {label}
        cwd: {cwd}
        command: {shlex.join(command)}
        exit code: {completed.returncode}

        stdout:
        {completed.stdout or "(empty)"}

        stderr:
        {completed.stderr or "(empty)"}
        """,
    )


def run_command(label: str, command: list[str], cwd: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    # All subprocess output is captured and mirrored into the run log so the
    # terminal stays readable while failures remain diagnosable.
    print(f"\nRunning {label}...")
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    append_completed_process_log(
        output_dir,
        label=label,
        command=command,
        cwd=cwd,
        completed=completed,
    )
    if completed.returncode != 0:
        stderr_tail = completed.stderr.strip().splitlines()[-6:]
        stdout_tail = completed.stdout.strip().splitlines()[-6:]
        tail = "\n".join(stderr_tail or stdout_tail)
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}."
            + (f"\nLast output:\n{tail}" if tail else "")
        )
    return completed


def copy_if_exists(src: Path, dst: Path) -> Path | None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst
    return None


def path_from_flag_or_detection(
    flag_value: str | Path | None,
    detected_path: Path | None,
    description: str,
) -> Path | None:
    """Prefer an explicit flag, then fall back to the FAIRMD autodetected path."""
    if flag_value is not None:
        path = existing_path(flag_value, description)
        print(f"Using {description} from flag: {path}")
        return path
    if detected_path is not None:
        print(f"Using detected {description}: {detected_path}")
    return detected_path


def value_from_flag_or_detection(
    flag_value: T | None,
    detected_value: T | None,
    description: str,
) -> T | None:
    """Prefer explicit numeric/text flags while still reporting autodetected defaults."""
    if flag_value is not None:
        print(f"Using {description} from flag: {flag_value}")
        return flag_value
    if detected_value is not None:
        print(f"Using detected {description}: {detected_value}")
    return detected_value


def gromacs_name_candidates(requested: str) -> list[str]:
    """Return executable names to try for --gmx, keeping explicit requests first."""
    value = requested.strip()
    if value in GMX_AUTO_VALUES:
        return GMX_DEFAULT_NAMES
    return [value]


def conda_root_candidates(args: argparse.Namespace) -> list[Path]:
    """Find likely conda roots from the chosen Python executable and environment."""
    roots: list[Path] = []

    python_path = Path(args.python).expanduser()
    if python_path.exists():
        resolved = python_path.resolve()
        for parent in resolved.parents:
            if parent.name in {"anaconda3", "miniconda3", "miniforge3", "mambaforge"}:
                roots.append(parent)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).expanduser()
        roots.append(prefix)
        if prefix.parent.name == "envs":
            roots.append(prefix.parent.parent)

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        exe_path = Path(conda_exe).expanduser()
        if exe_path.exists():
            roots.append(exe_path.resolve().parents[1])

    roots.extend(
        [
            Path("/opt/anaconda3"),
            Path("/opt/miniconda3"),
            Path.home() / "anaconda3",
            Path.home() / "miniconda3",
            Path.home() / "miniforge3",
            Path.home() / "mambaforge",
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve() if root.exists() else root
        if resolved not in seen and resolved.exists():
            unique.append(resolved)
            seen.add(resolved)
    return unique


def conda_env_sort_key(bin_dir: Path) -> tuple[int, str]:
    """Prefer FAIRMD/GROMACS-named envs when several conda envs have gmx."""
    name = bin_dir.parent.name.lower()
    if "fairmd" in name:
        priority = 0
    elif "gromacs" in name or "gmx" in name:
        priority = 1
    elif "openff" in name:
        priority = 2
    else:
        priority = 9
    return priority, name


def gromacs_search_dirs(args: argparse.Namespace) -> list[Path]:
    """Build bin directories that commonly contain conda-installed GROMACS."""
    dirs: list[Path] = []

    python_path = Path(args.python).expanduser()
    if python_path.exists():
        dirs.append(python_path.resolve().parent)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        dirs.append(Path(conda_prefix).expanduser() / "bin")

    for root in conda_root_candidates(args):
        dirs.append(root / "bin")
        envs_dir = root / "envs"
        if envs_dir.is_dir():
            dirs.extend(sorted(envs_dir.glob("*/bin"), key=conda_env_sort_key))
        pkgs_dir = root / "pkgs"
        if pkgs_dir.is_dir():
            dirs.extend(sorted(pkgs_dir.glob("gromacs-*/bin"), reverse=True))

    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        if not directory.exists():
            continue
        resolved = directory.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def runnable_gromacs(path: Path) -> bool:
    """Confirm a candidate binary can start before using it for gmx energy."""
    if not path.exists() or not os.access(path, os.X_OK):
        return False
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def resolve_gromacs_executable_for_surface(args: argparse.Namespace) -> str:
    """Resolve --gmx to an executable path, including conda env fallbacks."""
    requested = args.gmx.strip()
    for name in gromacs_name_candidates(requested):
        if "/" in name:
            candidate = resolve_path(name)
            if runnable_gromacs(candidate):
                return str(candidate)
            raise RuntimeError(f"GROMACS executable is not runnable: {candidate}")

        path_from_shell = shutil.which(name)
        if path_from_shell:
            candidate = Path(path_from_shell).resolve()
            if runnable_gromacs(candidate):
                return str(candidate)

    for directory in gromacs_search_dirs(args):
        for name in gromacs_name_candidates(requested):
            if "/" in name:
                continue
            candidate = directory / name
            if runnable_gromacs(candidate):
                return str(candidate.resolve())

    checked_names = ", ".join(gromacs_name_candidates(requested))
    raise RuntimeError(
        f"Could not find a runnable GROMACS executable for --gmx {requested!r}. "
        f"Checked PATH and conda envs for: {checked_names}. "
        "Pass --gmx /absolute/path/to/gmx if it lives somewhere else."
    )


def run_bending_modulus(output_dir: Path, args: argparse.Namespace, system: FairmdSystem) -> list[Path]:
    # Bending modulus needs APL rows, lipids per leaflet, and bilayer thickness.
    # Each value is taken from a flag first, then from FAIRMD autodetection.
    apl_json = require_value(
        path_from_flag_or_detection(args.apl_json, system.apl_json, "APL JSON"),
        "Bending modulus requires apl.json. Use --apl-json if it was not autodetected.",
    )
    lipids_per_leaflet = require_value(
        value_from_flag_or_detection(args.lipids_per_leaflet, system.lipids_per_leaflet, "lipids per leaflet"),
        "Bending modulus requires lipids per leaflet. Use --lipids-per-leaflet.",
    )
    bilayer_thickness = require_value(
        value_from_flag_or_detection(args.bilayer_thickness_nm, system.thickness_nm, "bilayer thickness (nm)"),
        "Bending modulus requires bilayer thickness in nm. Use --bilayer-thickness-nm.",
    )

    command = [
        args.python,
        str(script_path("bending")),
        str(apl_json),
        str(lipids_per_leaflet),
        f"{bilayer_thickness:g}",
    ]
    completed = run_command("calc_bending_modulus.py", command, empirical_dir(output_dir), output_dir)

    output_path = empirical_dir(output_dir) / "bending_modulus.txt"
    output_path.write_text(completed.stdout, encoding="utf-8")
    return [output_path]


def collect_surface_tension_args(output_dir: Path, args: argparse.Namespace, system: FairmdSystem) -> list[str]:
    # Surface tension follows calc_surface_tension.py's .edr workflow only.
    edr = path_from_flag_or_detection(args.edr, system.edr, "GROMACS .edr energy file")
    if edr is None:
        raise RuntimeError(
            "Surface tension requires a .edr energy file. "
            "No .edr file was detected; use --edr or omit surface_tension."
        )

    gmx = resolve_gromacs_executable_for_surface(args)
    print("Using detected surface tension input mode: edr")
    print(f"Using GROMACS executable: {gmx}")
    return [
        "--edr",
        str(edr),
        "--gmx",
        gmx,
        "--work-dir",
        str(info_dir(output_dir) / "surface_tension_gmx_work"),
        "--energy-xvg",
        str(info_dir(output_dir) / "surface_tension_pressure_tensor.xvg"),
        "--length-unit",
        args.surface_length_unit,
        "--pressure-unit",
        args.surface_pressure_unit,
        "--output-unit",
        args.surface_output_unit,
        "--precision",
        str(args.surface_precision),
        "--output",
        str(empirical_dir(output_dir) / "surface_tension.txt"),
    ]


def run_surface_tension(output_dir: Path, args: argparse.Namespace, system: FairmdSystem) -> list[Path]:
    surface_args = collect_surface_tension_args(output_dir, args, system)
    command = [args.python, str(script_path("surface_tension")), *surface_args]
    run_command("calc_surface_tension.py", command, output_dir, output_dir)
    artifacts = [empirical_dir(output_dir) / "surface_tension.txt"]
    xvg = info_dir(output_dir) / "surface_tension_pressure_tensor.xvg"
    if xvg.exists():
        artifacts.append(xvg)
    return [path for path in artifacts if path.exists()]


def tail_definitions_from_args(args: argparse.Namespace, system: FairmdSystem) -> list[str]:
    """Choose tail definitions from flags or from recognizable FAIRMD atom names."""
    if args.tail:
        print("Using tail definitions from --tail.")
        return args.tail

    mode = system.tail_mode if args.tail_mode == "auto" else args.tail_mode
    if mode == "charmm_popc":
        print("Using CHARMM-style POPC tail definitions.")
        return CHARMM_POPC_TAILS
    if mode == "openff_popc":
        print("Using OpenFF POPC tail definitions.")
        return OPENFF_POPC_TAILS
    if mode == "manual":
        raise RuntimeError("Tail mode manual requires at least one --tail definition.")

    print("Using automatic tail guessing.")
    return []


def run_tail_dihedral_ratios(output_dir: Path, args: argparse.Namespace, system: FairmdSystem) -> list[Path]:
    # Tail ratios need a topology/structure and usually a trajectory. The FAIRMD
    # directory supplies defaults, while flags allow a different tpr/gro/xtc set.
    default_topology = system.topology or system.structure
    topology = require_value(
        path_from_flag_or_detection(args.topology, default_topology, "topology/structure"),
        "Tail dihedral ratios require a topology or structure. Use --topology.",
    )

    if args.trajectory:
        trajectories = [existing_path(path, "trajectory") for path in args.trajectory]
        print("Using trajectory file(s) from --trajectory.")
    else:
        trajectories = [system.trajectory] if system.trajectory is not None else []
        if trajectories:
            print(f"Using detected trajectory: {trajectories[0]}")

    default_lipid_select = f"resname {system.lipid_resname or 'POPC'}"
    lipid_select = args.lipid_select or default_lipid_select
    print(f"Using lipid atom selection: {lipid_select}")

    tails = tail_definitions_from_args(args, system)

    command_args: list[str] = ["--topology", str(topology)]
    if trajectories:
        command_args += ["--trajectory", *[str(path) for path in trajectories]]
    command_args += ["--lipid-select", lipid_select]
    if tails:
        for tail in tails:
            command_args += ["--tail", tail]
        command_args += ["--no-guess-tails"]
    else:
        command_args += ["--guess-tails"]

    # The state definition is intentionally fixed to the requested window method.
    command_args += ["--state-definition", "window", "--window-deg", f"{args.window_deg:g}"]
    if args.start is not None:
        command_args += ["--start", str(args.start)]
    if args.stop is not None:
        command_args += ["--stop", str(args.stop)]
    command_args += ["--step", str(args.step)]
    if args.progress_every:
        command_args += ["--progress-every", str(args.progress_every)]

    graph_dir = graphs_dir(output_dir)
    output_csv = graph_dir / "tail_dihedral_ratios.csv"
    transition_csv = graph_dir / "tail_dihedral_ratios_transitions_count.csv"
    plot_base = graph_dir / "tail_dihedral_ratios_tg_simple_percent_per_carbon.png"
    command_args += [
        "--output",
        str(output_csv),
        "--transition-output",
        str(transition_csv),
        "--plot-output",
        str(plot_base),
    ]
    command = [args.python, str(script_path("tail_dihedral")), *command_args]
    run_command("calc_tail_dihedral_ratios.py", command, output_dir, output_dir)

    artifacts = [output_csv, transition_csv]
    artifacts.extend(sorted(graph_dir.glob("tail_dihedral_ratios_tg_simple_percent_per_carbon*.png")))
    return [path for path in artifacts if path.exists()]


def run_apl_plot(output_dir: Path, args: argparse.Namespace, system: FairmdSystem) -> list[Path]:
    # APL plotting is implemented here so the plot and summary can be written
    # directly into the requested graphs/ and empirical_results/ folders.
    apl_json = require_value(
        path_from_flag_or_detection(args.apl_json, system.apl_json, "APL JSON"),
        "APL plotting requires apl.json. Use --apl-json if it was not autodetected.",
    )
    time_unit = args.apl_time_unit
    production_start = args.apl_production_start_ns

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Missing dependency needed for graphAPL.py behavior: matplotlib.") from exc

    try:
        time_values, apl_values = load_xy_series_json(apl_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read APL data from {apl_json}: {exc}") from exc

    time_ns = [value / 1000.0 for value in time_values] if time_unit == "ps" else time_values
    if production_start is not None:
        filtered = [
            (time_value, apl_value)
            for time_value, apl_value in zip(time_ns, apl_values)
            if time_value >= production_start
        ]
        time_ns = [time_value for time_value, _apl_value in filtered]
        apl_values = [apl_value for _time_value, apl_value in filtered]
    if len(apl_values) == 0:
        raise RuntimeError("No APL data remained after filtering.")

    avg_apl = statistics.fmean(apl_values)
    plot_path = graphs_dir(output_dir) / "APL_vs_time.png"
    summary_path = empirical_dir(output_dir) / "APL_summary.txt"

    plt.figure(figsize=(8, 5))
    plt.plot(time_ns, apl_values, linewidth=1.5)
    plt.axhline(
        avg_apl,
        linestyle="--",
        linewidth=2,
        color="black",
        label=f"Average = {avg_apl:.2f} A^2",
    )
    plt.xlabel("Time (ns)")
    plt.ylabel("Area per Lipid (A^2)")
    plt.title("Area per Lipid vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    summary_lines = [
        "Area per Lipid Summary",
        "----------------------",
        f"System path: {system.path}",
        f"APL JSON: {apl_json}",
        f"Input time unit: {time_unit}",
        f"Frames used: {len(apl_values)}",
        f"Average APL: {avg_apl:.6g} A^2",
    ]
    if production_start is not None:
        summary_lines.append(f"Production start filter: {production_start:g} ns")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    append_log(output_dir, f"graphAPL.py behavior wrote {plot_path} and {summary_path}")
    return [plot_path, summary_path]


def load_order_parameter_module():
    module_path = script_path("order_parameters")
    spec = importlib.util.spec_from_file_location("JH_OrderParameterPlotting_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repeated_option_value(values: list[str] | None, index: int, default: str) -> str:
    """Return the matching repeated flag value, or a stable default."""
    if values and index < len(values) and values[index]:
        return values[index]
    return default


def run_order_parameter_plot(output_dir: Path, args: argparse.Namespace, system: FairmdSystem) -> list[Path]:
    # Order-parameter datasets are assembled as triples expected by
    # JH_OrderParameterPlotting.py: (json file, matplotlib color, legend label).
    data_files: list[tuple[str, str, str]] = []

    if args.order_json:
        order_jsons = [existing_path(path, "order-parameter JSON") for path in args.order_json]
        print("Using order-parameter JSON file(s) from --order-json.")
    else:
        order_jsons = system.order_parameter_jsons
        for path in order_jsons:
            print(f"Using detected order-parameter JSON: {path}")

    if not order_jsons:
        raise RuntimeError("Order-parameter plotting requires at least one JSON file. Use --order-json.")

    simulation_label = args.simulation_label or system.path.name
    for index, json_path in enumerate(order_jsons):
        lipid_label = json_path.stem.replace("OrderParameters", "").strip("_")
        default_label = simulation_label if len(order_jsons) == 1 else f"{simulation_label} {lipid_label or json_path.stem}"
        color = repeated_option_value(args.order_color, index, PLOT_COLORS[index % len(PLOT_COLORS)])
        label = repeated_option_value(args.order_label, index, default_label)
        data_files.append((str(json_path), color, label))

    # Experimental OP references are optional. When discovered through README.yaml,
    # the exact README reference path is kept as the legend label by default.
    if args.include_experimental_reference or args.experimental_order_json:
        if args.experimental_order_json:
            ref_path = require_value(
                existing_path(args.experimental_order_json, "experimental order-parameter JSON"),
                "Experimental order-parameter JSON was not found.",
            )
            ref_label = args.experimental_label or str(ref_path)
            print(f"Using experimental order-parameter reference from --experimental-order-json: {ref_path}")
            data_files.append((str(ref_path), "black", ref_label))
        else:
            experimental_refs = resolve_experimental_order_parameter_paths(system)
            if experimental_refs:
                ref_path, ref_label = experimental_refs[0]
                if len(experimental_refs) > 1:
                    print("Multiple experimental order-parameter references were found; using the first listed one.")
                label = args.experimental_label or ref_label
                print(f"Using detected experimental order-parameter reference: {ref_path}")
                data_files.append((str(ref_path), "black", label))
            else:
                print("No experimental order-parameter reference file was found from README.yaml.")

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError as exc:
        raise RuntimeError("Missing dependency: matplotlib") from exc

    old_cwd = Path.cwd()
    graph_dir = graphs_dir(output_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(graph_dir)
        module = load_order_parameter_module()
        if hasattr(module, "plt"):
            module.plt.show = lambda *args, **kwargs: None
        module.plot_data(data_files)
    finally:
        os.chdir(old_cwd)

    candidates = [
        graph_dir / "ORDERPARAMETERS_PLOT.png",
        graph_dir / "ORDERPARAMETERS_PLOT",
    ]
    artifacts: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        final_path = graph_dir / "order_parameters_plot.png"
        if candidate != final_path:
            candidate.replace(final_path)
        artifacts.append(final_path)
        break

    append_log(output_dir, f"JH_OrderParameterPlotting.py plot_data inputs: {data_files!r}")
    if not artifacts:
        raise RuntimeError("Order parameter plotting finished, but no plot file was found.")
    return artifacts


RUNNERS: dict[str, Callable[[Path, argparse.Namespace, FairmdSystem], list[Path]]] = {
    "bending": run_bending_modulus,
    "surface_tension": run_surface_tension,
    "tail_dihedral": run_tail_dihedral_ratios,
    "apl": run_apl_plot,
    "order_parameters": run_order_parameter_plot,
}


def write_summary(output_dir: Path, system: FairmdSystem, results: list[MetricResult]) -> Path:
    summary_path = info_dir(output_dir) / "getMetrics_summary.txt"
    lines = [
        "getMetrics Summary",
        "==================",
        f"Run time: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"User: {getpass.getuser()}",
        f"System folder: {system.path}",
        f"Output folder: {output_dir}",
        f"Lipid residue: {system.lipid_resname or 'not detected'}",
        "",
    ]
    for result in results:
        label = METRIC_LABELS.get(result.name, result.name.replace("_", " "))
        lines.append(f"{label}: {result.status}")
        if result.message:
            lines.append(f"  {result.message}")
        for artifact in result.artifacts:
            lines.append(f"  {artifact}")
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FAIRMD metric scripts with flags and write all outputs to one folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--system-dir",
        type=Path,
        required=True,
        help="FAIRMD analysis/system directory containing files such as apl.json, conf.gro, md_2.tpr, md.tpr, and POPCOrderParameters.json.",
    )
    parser.add_argument(
        "--output-title",
        help="Name/title for the output folder. Defaults to the system folder name.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPTS_DIR,
        help="Directory where the output-title folder is created.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable or "python3",
        help="Python executable used when calling subprocess metric scripts.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma/space separated metrics to run: all, bending, surface_tension, tail_dihedral, apl, order_parameters.",
    )
    parser.add_argument(
        "--only",
        help="Backward-compatible alias for --metrics. If supplied, it overrides --metrics.",
    )
    parser.add_argument(
        "--skip",
        help="Comma/space separated metric keys to skip.",
    )

    # Bending modulus inputs. FAIRMD autodetection supplies defaults when possible.
    parser.add_argument(
        "--apl-json",
        type=Path,
        help="APL JSON input for bending modulus and APL plotting.",
    )
    parser.add_argument(
        "--lipids-per-leaflet",
        type=int,
        help="Lipids per leaflet for bending modulus.",
    )
    parser.add_argument(
        "--bilayer-thickness-nm",
        type=float,
        help="Bilayer thickness in nm for bending modulus.",
    )

    # Surface tension remains disabled unless an .edr is detected or explicitly supplied.
    parser.add_argument("--edr", type=Path, help="GROMACS .edr energy file for surface tension.")
    parser.add_argument(
        "--gmx",
        default="auto",
        help="GROMACS executable for energy extraction, or auto to search PATH/conda envs.",
    )
    parser.add_argument("--surface-length-unit", default="nm", help="Length unit passed to calc_surface_tension.py.")
    parser.add_argument("--surface-pressure-unit", default="bar", help="Pressure unit passed to calc_surface_tension.py.")
    parser.add_argument("--surface-output-unit", default="mN/m", help="Output unit passed to calc_surface_tension.py.")
    parser.add_argument("--surface-precision", type=int, default=6, help="Decimal precision for surface tension output.")

    # Tail dihedral ratio controls. The state definition is fixed to window in the runner.
    parser.add_argument("--topology", type=Path, help="Topology/structure file for tail dihedral ratios.")
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        help="Trajectory file for tail dihedral ratios. Can be repeated.",
    )
    parser.add_argument("--lipid-select", help="Atom selection for lipid residues.")
    parser.add_argument(
        "--tail-mode",
        choices=["auto", "guess", "charmm_popc", "openff_popc", "manual"],
        default="auto",
        help="Tail definition source for tail dihedral ratios.",
    )
    parser.add_argument(
        "--tail",
        action="append",
        help="Tail definition such as sn1:C3[1-16]. Can be repeated; implies manual tails.",
    )
    parser.add_argument(
        "--window-deg",
        type=float,
        default=60.0,
        help="Window half-width in degrees for tail dihedral classification.",
    )
    parser.add_argument("--start", type=int, help="First frame index for tail dihedral ratios.")
    parser.add_argument("--stop", type=int, help="Stop-before frame index for tail dihedral ratios.")
    parser.add_argument("--step", type=int, default=1, help="Frame step for tail dihedral ratios.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print tail-dihedral progress every N frames. 0 disables progress.",
    )

    # APL plot controls.
    parser.add_argument(
        "--apl-time-unit",
        choices=["ps", "ns"],
        default="ps",
        help="Time unit used by apl.json.",
    )
    parser.add_argument(
        "--apl-production-start-ns",
        type=float,
        help="Drop APL frames before this time in ns.",
    )

    # Order parameter plot controls.
    parser.add_argument(
        "--order-json",
        type=Path,
        action="append",
        help="Order-parameter JSON to plot. Can be repeated; overrides autodetected simulation files.",
    )
    parser.add_argument(
        "--simulation-label",
        help="Base label for autodetected/simulation order-parameter datasets.",
    )
    parser.add_argument(
        "--order-label",
        action="append",
        help="Legend label for an order-parameter dataset. Can be repeated by dataset order.",
    )
    parser.add_argument(
        "--order-color",
        action="append",
        help="Matplotlib color for an order-parameter dataset. Can be repeated by dataset order.",
    )
    parser.add_argument(
        "--include-experimental-reference",
        action="store_true",
        help="Include the first experimental order-parameter reference found through README.yaml.",
    )
    parser.add_argument(
        "--experimental-order-json",
        type=Path,
        help="Explicit experimental order-parameter JSON for plotting.",
    )
    parser.add_argument(
        "--experimental-label",
        help="Legend label for the experimental reference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Discovery happens once, then every runner uses the same FairmdSystem
    # record plus any metric-specific flag overrides.
    system_path = args.system_dir.expanduser().resolve()
    system = discover_fairmd_system(system_path)
    print_system_detection(system)

    title = args.output_title or system.path.name
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / sanitize_folder_name(title)
    ensure_output_dirs(output_dir)
    detected_inputs_path = write_detected_inputs(output_dir, system)

    append_log(
        output_dir,
        f"""
        ================================================================================
        [{datetime.now():%Y-%m-%d %H:%M:%S}] getMetrics.py started
        scripts directory: {SCRIPTS_DIR}
        system directory: {system.path}
        output directory: {output_dir}
        python executable: {args.python}
        """,
    )

    print(f"\nOutput folder: {output_dir}")
    print(f"Detected inputs written to: {detected_inputs_path}")
    print("Relative input paths are resolved from your current directory, then from the Scripts folder.")

    # Metrics run in a stable order even when the user supplies them as a
    # comma-separated flag list.
    selected_metrics = select_metrics(args)
    results: list[MetricResult] = [
        MetricResult("detected_inputs", "complete", [detected_inputs_path]),
    ]
    for metric in selected_metrics:
        label = METRIC_LABELS[metric]
        if metric == "surface_tension" and system.edr is None and args.edr is None:
            message = (
                "Skipped because no .edr energy file was detected in the FAIRMD system directory. "
                "Surface tension is only run through calc_surface_tension.py --edr."
            )
            print(f"\nSkipping {label}: {message}")
            append_log(output_dir, f"{label} skipped: {message}")
            results.append(MetricResult(metric, "skipped", [], message))
            continue

        try:
            artifacts = RUNNERS[metric](output_dir, args, system)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            message = str(exc)
            print(f"\n{label} failed: {message}")
            append_log(output_dir, f"{label} failed: {message}")
            results.append(MetricResult(metric, "failed", [], message))
        else:
            print(f"{label} complete.")
            for artifact in artifacts:
                print(f"  {artifact}")
            results.append(MetricResult(metric, "complete", artifacts))

    summary_path = write_summary(output_dir, system, results)
    print(f"\nSummary written to: {summary_path}")
    print(f"Run log written to: {info_dir(output_dir) / 'getMetrics_run_log.txt'}")
    return 0 if all(result.status != "failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
