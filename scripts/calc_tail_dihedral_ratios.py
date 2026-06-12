#!/usr/bin/env python3
"""
Calculate trans/gauche ratios for dihedral angles down lipid tails.

This script uses MDAnalysis to read an MD topology/trajectory, compute every
four-atom torsion along lipid tail carbon chains, and summarize the fraction of
samples in trans versus gauche states.

By default, states are assigned by rotamer basins: trans is centered at
+/-180 degrees and uses |phi| >= 120 degrees, gauche+ is centered at +60
degrees and uses 0 < phi < 120 degrees, and gauche- is centered at -60 degrees
and uses -120 < phi < 0 degrees. Use --state-definition window for stricter
classification around the ideal trans and gauche centers only.

The script also writes a PNG plot of the selected ratio/metric versus tail
dihedral position. Each point corresponds to the central carbon-carbon bond of
one four-carbon dihedral.

Install dependencies:
    python -m pip install MDAnalysis numpy matplotlib

Example with explicit CHARMM-style POPC tail atoms:
    python calc_tail_dihedral_ratios.py \
        -s topol.tpr -f traj.xtc \
        --lipid-select "resname POPC" \
        --tail "sn1:C3[1-16]" \
        --tail "sn2:C2[1-18]" \
        -o popc_tail_trans_gauche.csv

Example with GROMOS-style chain names:
    python calc_tail_dihedral_ratios.py \
        -s system.gro -f traj.xtc \
        --lipid-select "resname DPPC" \
        --tail "sn1:C[1-16]A" \
        --tail "sn2:C[1-16]B"

If --tail is omitted, the script attempts to infer common atom-name patterns.
For unusual lipid or force-field naming, explicit --tail arguments are safer.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple


Key = Tuple[str, str, int, str, str, str, str]


@dataclass(frozen=True)
class TailDefinition:
    label: str
    atom_names: Tuple[str, ...]


@dataclass(frozen=True)
class DihedralSpec:
    key: Key
    atom_indices: Tuple[int, int, int, int]


@dataclass
class Accumulator:
    key: Key
    total: int = 0
    trans_count: int = 0
    gauche_plus_count: int = 0
    gauche_minus_count: int = 0
    other_count: int = 0
    sin_sum: float = 0.0
    cos_sum: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate trans/gauche ratios for every four-carbon dihedral down "
            "lipid tails in an MD trajectory."
        )
    )
    parser.add_argument(
        "-s",
        "--topology",
        required=True,
        help="Topology/structure file readable by MDAnalysis, such as TPR, PSF, GRO, PDB, or PRMTOP.",
    )
    parser.add_argument(
        "-f",
        "--trajectory",
        nargs="*",
        default=[],
        help="Trajectory file(s), such as XTC, TRR, DCD, NC, or other MDAnalysis-readable formats.",
    )
    parser.add_argument(
        "--lipid-select",
        default="all",
        help='MDAnalysis atom selection used to choose lipid residues. Default: "all".',
    )
    parser.add_argument(
        "--tail",
        action="append",
        default=[],
        help=(
            "Tail atom names in order from glycerol/headgroup end toward tail end. "
            "Use LABEL:ATOM1,ATOM2,... or LABEL:PREFIX[START-END]SUFFIX. "
            'Examples: "sn1:C3[1-16]", "sn2:C2[1-18]", "sn1:C31,C32,C33,C34". '
            "Repeat for each tail."
        ),
    )
    parser.add_argument(
        "--guess-tails",
        action="store_true",
        default=None,
        help="Infer common lipid tail atom-name patterns if --tail is omitted.",
    )
    parser.add_argument(
        "--no-guess-tails",
        action="store_false",
        dest="guess_tails",
        help="Do not infer tails automatically. Requires at least one --tail.",
    )
    parser.add_argument(
        "--trans-cutoff-deg",
        type=float,
        default=120.0,
        help=(
            "Basin boundary for trans classification in degrees. In the default basin mode, "
            "abs(phi) >= this value is assigned to the trans basin centered at +/-180. "
            "Default: 120."
        ),
    )
    parser.add_argument(
        "--cis-cutoff-deg",
        type=float,
        default=0.0,
        help=(
            "Optional exclusion window around 0 degrees. If > 0, angles with "
            "abs(phi) <= this value are counted as other instead of gauche. Default: 0."
        ),
    )
    parser.add_argument(
        "--state-definition",
        choices=("basin", "window"),
        default="basin",
        help=(
            "How to assign states. 'basin' uses rotamer basins centered at trans +/-180 "
            "and gauche +/-60. 'window' counts only angles within --window-deg of "
            "those ideal centers and marks the rest as other. Default: basin."
        ),
    )
    parser.add_argument(
        "--window-deg",
        type=float,
        default=30.0,
        help="Half-width around ideal centers for --state-definition window. Default: 30.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First trajectory frame to analyze, using Python slice indexing.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop before this trajectory frame, using Python slice indexing.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Analyze every Nth frame. Default: 1.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an explicitly requested tail atom is missing from any selected residue.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N analyzed frames. Default: 0, disabled.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tail_dihedral_trans_gauche.csv",
        help="Output CSV path. Default: tail_dihedral_trans_gauche.csv.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help=(
            "PNG path for a per-carbon plot. If multiple tails are plotted, the tail label "
            "is added before the file extension. Default: save next to the CSV as "
            "<csv_stem>_<plot_metric>_per_carbon_<tail>.png."
        ),
    )
    parser.add_argument(
        "--plot-metric",
        choices=("trans_gauche_ratio", "trans_fraction", "gauche_fraction"),
        default="trans_gauche_ratio",
        help="Metric to graph versus tail dihedral position. Default: trans_gauche_ratio.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write the CSV and skip the PNG plot.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error status messages.",
    )
    return parser


def expand_atom_token(token: str) -> List[str]:
    """Expand compact atom-name ranges such as C3[1-16] or C[1-18]A."""
    token = token.strip()
    match = re.fullmatch(r"(?P<prefix>.*)\[(?P<start>\d+)-(?P<end>\d+)\](?P<suffix>.*)", token)
    if not match:
        return [token] if token else []

    prefix = match.group("prefix")
    suffix = match.group("suffix")
    start_text = match.group("start")
    end_text = match.group("end")
    start = int(start_text)
    end = int(end_text)
    step = 1 if end >= start else -1
    width = max(len(start_text), len(end_text)) if start_text.startswith("0") or end_text.startswith("0") else 0

    names = []
    for value in range(start, end + step, step):
        number = f"{value:0{width}d}" if width else str(value)
        names.append(f"{prefix}{number}{suffix}")
    return names


def parse_tail_definition(text: str, index: int) -> TailDefinition:
    if ":" in text:
        label, atom_text = text.split(":", 1)
        label = label.strip()
    else:
        label = f"tail{index}"
        atom_text = text

    atom_names: List[str] = []
    for token in re.split(r"[,\s]+", atom_text.strip()):
        atom_names.extend(expand_atom_token(token))

    atom_names = [name.strip() for name in atom_names if name.strip()]
    if not label:
        raise ValueError(f"Tail definition {text!r} has an empty label.")
    if len(atom_names) < 4:
        raise ValueError(f"Tail {label!r} needs at least 4 atom names to form a dihedral.")
    return TailDefinition(label=label, atom_names=tuple(atom_names))


def parse_tail_definitions(values: Sequence[str]) -> List[TailDefinition]:
    return [parse_tail_definition(value, index + 1) for index, value in enumerate(values)]


def residue_atom_lookup(residue) -> Tuple[Dict[str, object], List[str]]:
    atoms: Dict[str, object] = {}
    duplicates: List[str] = []
    for atom in residue.atoms:
        name = atom.name.strip()
        if name in atoms:
            duplicates.append(name)
        else:
            atoms[name] = atom
    return atoms, duplicates


def add_dihedral_specs(specs: List[DihedralSpec], resname: str, tail_label: str, atoms: Sequence[object]) -> None:
    for offset in range(len(atoms) - 3):
        quartet = atoms[offset : offset + 4]
        atom_names = tuple(atom.name.strip() for atom in quartet)
        key: Key = (resname, tail_label, offset + 1, atom_names[0], atom_names[1], atom_names[2], atom_names[3])
        atom_indices = tuple(int(atom.index) for atom in quartet)
        specs.append(DihedralSpec(key=key, atom_indices=atom_indices))  # type: ignore[arg-type]


def build_explicit_specs(
    residues,
    tail_definitions: Sequence[TailDefinition],
    strict: bool,
) -> Tuple[List[DihedralSpec], Dict[Tuple[str, str], int], Dict[Tuple[str, str], List[str]]]:
    specs: List[DihedralSpec] = []
    skipped: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    missing_examples: Dict[Tuple[str, str], List[str]] = {}

    for residue in residues:
        resname = residue.resname.strip()
        atom_lookup, _duplicates = residue_atom_lookup(residue)
        for tail in tail_definitions:
            missing = [name for name in tail.atom_names if name not in atom_lookup]
            if missing:
                key = (resname, tail.label)
                skipped[key] += 1
                missing_examples.setdefault(key, missing[:8])
                if strict:
                    resid = getattr(residue, "resid", "?")
                    raise ValueError(
                        f"Residue {resname} {resid} is missing atom(s) for tail {tail.label}: "
                        f"{', '.join(missing)}"
                    )
                continue

            atoms = [atom_lookup[name] for name in tail.atom_names]
            add_dihedral_specs(specs, resname, tail.label, atoms)

    return specs, dict(skipped), missing_examples


def candidate_tail_chains(residue, min_atoms: int = 4) -> List[Tuple[str, List[object]]]:
    """Infer common lipid tail atom-name patterns within one residue."""
    pattern_specs = [
        # CHARMM/Slipids-style: C21..C218 and C31..C316.
        (re.compile(r"^C([23])(\d+)$"), lambda m: f"C{m.group(1)}", lambda m: int(m.group(2))),
        # GROMOS-style: C1A..C16A and C1B..C16B.
        (re.compile(r"^C(\d+)([A-Za-z])$"), lambda m: f"chain_{m.group(2)}", lambda m: int(m.group(1))),
        # Alternative: CA1..CA16 and CB1..CB16.
        (re.compile(r"^C([A-Za-z])(\d+)$"), lambda m: f"chain_{m.group(1)}", lambda m: int(m.group(2))),
        # Alternative: AC1..AC16 and BC1..BC16.
        (re.compile(r"^([A-Za-z])C(\d+)$"), lambda m: f"chain_{m.group(1)}", lambda m: int(m.group(2))),
    ]

    candidates: List[Tuple[str, List[object]]] = []
    for regex, label_fn, order_fn in pattern_specs:
        groups: DefaultDict[str, List[Tuple[int, object]]] = defaultdict(list)
        for atom in residue.atoms:
            name = atom.name.strip()
            match = regex.fullmatch(name)
            if not match:
                continue
            groups[label_fn(match)].append((order_fn(match), atom))

        for label, entries in groups.items():
            by_order: Dict[int, object] = {}
            duplicate_order = False
            for order, atom in entries:
                if order in by_order:
                    duplicate_order = True
                    break
                by_order[order] = atom
            if duplicate_order:
                continue

            ordered = [atom for _order, atom in sorted(by_order.items())]
            if len(ordered) >= min_atoms:
                candidates.append((label, ordered))

    kept: List[Tuple[str, List[object]]] = []
    used_atom_indices = set()
    for label, atoms in sorted(candidates, key=lambda item: (-len(item[1]), item[0])):
        atom_indices = {int(atom.index) for atom in atoms}
        if atom_indices & used_atom_indices:
            continue
        used_atom_indices.update(atom_indices)
        kept.append((label, atoms))

    return sorted(kept, key=lambda item: item[0])


def build_guessed_specs(residues) -> Tuple[List[DihedralSpec], Dict[str, int]]:
    specs: List[DihedralSpec] = []
    guessed_residue_counts: DefaultDict[str, int] = defaultdict(int)

    for residue in residues:
        resname = residue.resname.strip()
        chains = candidate_tail_chains(residue)
        if not chains:
            continue
        guessed_residue_counts[resname] += 1
        for label, atoms in chains:
            add_dihedral_specs(specs, resname, label, atoms)

    return specs, dict(guessed_residue_counts)


def import_analysis_dependencies():
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Missing dependency: numpy. Install with: python -m pip install numpy") from exc

    try:
        import MDAnalysis as mda
        from MDAnalysis.lib.distances import calc_dihedrals
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: MDAnalysis. Install with: python -m pip install MDAnalysis"
        ) from exc

    return np, mda, calc_dihedrals


def has_valid_box(dimensions, np_module) -> bool:
    if dimensions is None:
        return False
    dims = np_module.asarray(dimensions)
    if dims.size < 6:
        return False
    return bool(np_module.all(np_module.isfinite(dims[:6])) and np_module.all(dims[:3] > 0.0))


def update_accumulator(
    accumulator: Accumulator,
    angles_rad,
    np_module,
    trans_cutoff_deg: float,
    cis_cutoff_deg: float,
    state_definition: str,
    window_deg: float,
) -> None:
    finite_angles = angles_rad[np_module.isfinite(angles_rad)]
    if finite_angles.size == 0:
        return

    angles_deg = np_module.degrees(finite_angles)
    abs_angles = np_module.abs(angles_deg)

    if state_definition == "window":
        # calc_dihedrals returns angles in [-180, 180]. Near-trans values can
        # sit on either side of that periodic boundary, so use abs(phi).
        trans = (180.0 - abs_angles) <= window_deg
        gauche_plus = np_module.abs(angles_deg - 60.0) <= window_deg
        gauche_minus = np_module.abs(angles_deg + 60.0) <= window_deg
        other = ~(trans | gauche_plus | gauche_minus)
    else:
        if cis_cutoff_deg > 0.0:
            other = abs_angles <= cis_cutoff_deg
        else:
            other = np_module.zeros(angles_deg.shape, dtype=bool)

        trans = (~other) & (abs_angles >= trans_cutoff_deg)
        gauche = (~other) & (~trans)
        gauche_plus = gauche & (angles_deg > 0.0)
        gauche_minus = gauche & (angles_deg < 0.0)
        other = other | (gauche & (angles_deg == 0.0))

    accumulator.total += int(finite_angles.size)
    accumulator.trans_count += int(np_module.count_nonzero(trans))
    accumulator.gauche_plus_count += int(np_module.count_nonzero(gauche_plus))
    accumulator.gauche_minus_count += int(np_module.count_nonzero(gauche_minus))
    accumulator.other_count += int(np_module.count_nonzero(other))
    accumulator.sin_sum += float(np_module.sin(finite_angles).sum())
    accumulator.cos_sum += float(np_module.cos(finite_angles).sum())


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.8g}"


def trans_gauche_ratio(trans_count: int, gauche_count: int) -> float:
    if gauche_count:
        return trans_count / gauche_count
    if trans_count:
        return math.inf
    return math.nan


def accumulator_metric(accumulator: Accumulator, metric: str) -> float:
    gauche_count = accumulator.gauche_plus_count + accumulator.gauche_minus_count
    total = accumulator.total

    if metric == "trans_gauche_ratio":
        return trans_gauche_ratio(accumulator.trans_count, gauche_count)
    if metric == "trans_fraction":
        return accumulator.trans_count / total if total else math.nan
    if metric == "gauche_fraction":
        return gauche_count / total if total else math.nan
    raise ValueError(f"Unknown plot metric: {metric}")


def accumulator_row(accumulator: Accumulator, n_lipids: int, n_frames: int) -> Dict[str, object]:
    resname, tail, dihedral_index, atom1, atom2, atom3, atom4 = accumulator.key
    gauche_count = accumulator.gauche_plus_count + accumulator.gauche_minus_count
    total = accumulator.total

    trans_fraction = accumulator.trans_count / total if total else math.nan
    gauche_fraction = gauche_count / total if total else math.nan
    gauche_plus_fraction = accumulator.gauche_plus_count / total if total else math.nan
    gauche_minus_fraction = accumulator.gauche_minus_count / total if total else math.nan
    ratio = trans_gauche_ratio(accumulator.trans_count, gauche_count)

    mean_angle = math.degrees(math.atan2(accumulator.sin_sum, accumulator.cos_sum)) if total else math.nan

    return {
        "resname": resname,
        "tail": tail,
        "dihedral_index": dihedral_index,
        "atom1": atom1,
        "atom2": atom2,
        "atom3": atom3,
        "atom4": atom4,
        "central_bond": f"{atom2}-{atom3}",
        "n_lipids": n_lipids,
        "n_frames": n_frames,
        "n_observations": total,
        "trans_count": accumulator.trans_count,
        "gauche_count": gauche_count,
        "gauche_plus_count": accumulator.gauche_plus_count,
        "gauche_minus_count": accumulator.gauche_minus_count,
        "other_count": accumulator.other_count,
        "trans_fraction": format_float(trans_fraction),
        "gauche_fraction": format_float(gauche_fraction),
        "gauche_plus_fraction": format_float(gauche_plus_fraction),
        "gauche_minus_fraction": format_float(gauche_minus_fraction),
        "trans_gauche_ratio": format_float(ratio),
        "mean_angle_deg": format_float(mean_angle),
    }


def write_summary_csv(
    output_path: Path,
    accumulators: Dict[Key, Accumulator],
    key_to_spec_indices: Dict[Key, List[int]],
    n_frames: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "resname",
        "tail",
        "dihedral_index",
        "atom1",
        "atom2",
        "atom3",
        "atom4",
        "central_bond",
        "n_lipids",
        "n_frames",
        "n_observations",
        "trans_count",
        "gauche_count",
        "gauche_plus_count",
        "gauche_minus_count",
        "other_count",
        "trans_fraction",
        "gauche_fraction",
        "gauche_plus_fraction",
        "gauche_minus_fraction",
        "trans_gauche_ratio",
        "mean_angle_deg",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(accumulators):
            writer.writerow(accumulator_row(accumulators[key], len(key_to_spec_indices[key]), n_frames))


def default_plot_path(output_path: Path, plot_metric: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{plot_metric}_per_carbon.png")


def safe_filename_part(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe_value.strip("._") or "tail"


def tail_plot_path(base_path: Path, tail: str, multiple_tails: bool) -> Path:
    if not multiple_tails:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{safe_filename_part(tail)}{base_path.suffix}")


def plot_summary_pngs(
    output_path: Path,
    accumulators: Dict[Key, Accumulator],
    plot_metric: str,
    state_definition: str,
    trans_cutoff_deg: float,
    cis_cutoff_deg: float,
    window_deg: float,
) -> List[Path]:
    cache_root = Path(tempfile.gettempdir()) / "tail_dihedral_plot_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install with: python -m pip install matplotlib"
        ) from exc

    metric_labels = {
        "trans_gauche_ratio": "Trans/Gauche ratio",
        "trans_fraction": "Trans fraction",
        "gauche_fraction": "Gauche fraction",
    }

    tail_groups: DefaultDict[str, DefaultDict[str, List[Accumulator]]] = defaultdict(lambda: defaultdict(list))
    for accumulator in accumulators.values():
        resname, tail, _dihedral_index, _atom1, _atom2, _atom3, _atom4 = accumulator.key
        tail_groups[tail][resname].append(accumulator)

    if state_definition == "window":
        cutoff_text = (
            f"State definition: window\n"
            f"trans: within {window_deg:g} deg of +/-180\n"
            f"gauche: within {window_deg:g} deg of +/-60"
        )
    else:
        cutoff_text = (
            f"State definition: basin\n"
            f"trans: |phi| >= {trans_cutoff_deg:g} deg\n"
            f"gauche: remaining non-cis angles"
        )
        if cis_cutoff_deg > 0.0:
            cutoff_text += f"\nother/cis: |phi| <= {cis_cutoff_deg:g} deg"

    written_paths: List[Path] = []
    multiple_tails = len(tail_groups) > 1
    for tail, resname_groups in sorted(tail_groups.items()):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        finite_points = 0
        all_positions = set()

        for resname, group_accumulators in sorted(resname_groups.items()):
            sorted_group = sorted(group_accumulators, key=lambda item: item.key[2])
            x_values = []
            y_values = []
            for accumulator in sorted_group:
                dihedral_index = accumulator.key[2]
                value = accumulator_metric(accumulator, plot_metric)
                x_values.append(dihedral_index)
                y_values.append(value if math.isfinite(value) else math.nan)
                all_positions.add(dihedral_index)
                if math.isfinite(value):
                    finite_points += 1

            ax.scatter(x_values, y_values, s=42, label=resname)

        if not finite_points:
            plt.close(fig)
            continue

        if all_positions:
            ax.set_xticks(sorted(all_positions))
        ax.set_xlabel("Dihedral position along tail (central C-C bond)")
        ax.set_ylabel(metric_labels[plot_metric])
        ax.set_title(f"{tail}: {metric_labels[plot_metric]} per tail carbon position")
        if plot_metric == "trans_gauche_ratio":
            ax.set_ylim(0.0, 4.0)
            ax.set_yticks([0, 1, 2, 3, 4])
        ax.text(
            0.02,
            0.98,
            cutoff_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
        )
        ax.grid(True, alpha=0.3)
        if len(resname_groups) > 1:
            ax.legend(frameon=False)

        fig.tight_layout()
        plot_path = tail_plot_path(output_path, tail, multiple_tails)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        written_paths.append(plot_path)

    return written_paths


def validate_args(args: argparse.Namespace) -> None:
    if args.step <= 0:
        raise SystemExit("--step must be a positive integer.")
    if args.trans_cutoff_deg <= 0.0 or args.trans_cutoff_deg > 180.0:
        raise SystemExit("--trans-cutoff-deg must be in the range (0, 180].")
    if args.cis_cutoff_deg < 0.0 or args.cis_cutoff_deg >= args.trans_cutoff_deg:
        raise SystemExit("--cis-cutoff-deg must be >= 0 and smaller than --trans-cutoff-deg.")
    if args.window_deg <= 0.0 or args.window_deg > 60.0:
        raise SystemExit("--window-deg must be in the range (0, 60].")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    if args.guess_tails is None:
        args.guess_tails = not bool(args.tail)

    tail_definitions = parse_tail_definitions(args.tail)
    if not tail_definitions and not args.guess_tails:
        raise SystemExit("Provide at least one --tail, or omit --no-guess-tails to infer common tail names.")

    np_module, mda, calc_dihedrals = import_analysis_dependencies()

    if args.trajectory:
        universe = mda.Universe(args.topology, *args.trajectory)
    else:
        universe = mda.Universe(args.topology)

    try:
        selected_atoms = universe.select_atoms(args.lipid_select)
    except Exception as exc:
        raise SystemExit(f"Could not apply --lipid-select {args.lipid_select!r}: {exc}") from exc

    residues = selected_atoms.residues
    if len(residues) == 0:
        raise SystemExit(f"--lipid-select {args.lipid_select!r} selected no residues.")

    if tail_definitions:
        specs, skipped, missing_examples = build_explicit_specs(residues, tail_definitions, args.strict)
        if skipped and not args.quiet:
            for (resname, tail_label), count in sorted(skipped.items()):
                missing = ", ".join(missing_examples.get((resname, tail_label), []))
                print(
                    f"Skipped {count} {resname} residues for tail {tail_label}; missing atom(s): {missing}",
                    file=sys.stderr,
                )
    else:
        specs, guessed_counts = build_guessed_specs(residues)
        if guessed_counts and not args.quiet:
            guessed_text = ", ".join(f"{resname}: {count}" for resname, count in sorted(guessed_counts.items()))
            print(f"Inferred tail chains for residues: {guessed_text}", file=sys.stderr)

    if not specs:
        raise SystemExit(
            "No tail dihedrals were found. Pass explicit tails, for example "
            '--tail "sn1:C3[1-16]" --tail "sn2:C2[1-18]", and check --lipid-select.'
        )

    atom_index_array = np_module.asarray([spec.atom_indices for spec in specs], dtype=np_module.int64)
    key_to_spec_indices: DefaultDict[Key, List[int]] = defaultdict(list)
    for spec_index, spec in enumerate(specs):
        key_to_spec_indices[spec.key].append(spec_index)

    accumulators = {key: Accumulator(key=key) for key in key_to_spec_indices}

    n_frames = 0
    frame_iterator = universe.trajectory[args.start : args.stop : args.step]
    for timestep in frame_iterator:
        positions = universe.atoms.positions
        box = timestep.dimensions if has_valid_box(timestep.dimensions, np_module) else None
        angles = calc_dihedrals(
            positions[atom_index_array[:, 0]],
            positions[atom_index_array[:, 1]],
            positions[atom_index_array[:, 2]],
            positions[atom_index_array[:, 3]],
            box=box,
        )

        for key, spec_indices in key_to_spec_indices.items():
            update_accumulator(
                accumulators[key],
                angles[spec_indices],
                np_module,
                args.trans_cutoff_deg,
                args.cis_cutoff_deg,
                args.state_definition,
                args.window_deg,
            )

        n_frames += 1
        if args.progress_every and n_frames % args.progress_every == 0:
            print(f"Analyzed {n_frames} frames...", file=sys.stderr)

    if n_frames == 0:
        raise SystemExit("The requested frame slice produced zero frames.")

    output_path = Path(args.output)
    write_summary_csv(output_path, accumulators, dict(key_to_spec_indices), n_frames)

    plot_path = None
    plot_paths: List[Path] = []
    if not args.no_plot:
        plot_path = Path(args.plot_output) if args.plot_output else default_plot_path(output_path, args.plot_metric)
        plot_paths = plot_summary_pngs(
            plot_path,
            accumulators,
            args.plot_metric,
            args.state_definition,
            args.trans_cutoff_deg,
            args.cis_cutoff_deg,
            args.window_deg,
        )

    if not args.quiet:
        print(
            f"Wrote {len(accumulators)} dihedral summaries from {n_frames} frame(s) to {output_path}",
            file=sys.stderr,
        )
        if plot_path is not None and plot_paths:
            for written_plot_path in plot_paths:
                print(f"Wrote per-carbon plot to {written_plot_path}", file=sys.stderr)
        elif plot_path is not None:
            print(
                "Skipped plot because the selected metric had no finite values.",
                file=sys.stderr,
            )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
