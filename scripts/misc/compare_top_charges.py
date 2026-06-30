#!/usr/bin/env python3
"""Compare selected atom charges between two GROMACS topology files of the same molecule type.

The parser expects standard GROMACS ``[ atoms ]`` rows:

    index atom_type resnum resname atom_name cgnr charge mass

Atom indices in GROMACS topology files are local to a moleculetype, so pass
``--molecule`` when the topology contains more than one molecule with the same
atom indices, such as a lipid plus water, which is what this script was designed to analyze.


Example Usage:
python compare_top_charges.py top_A.top top_B.top --molecule POPC --atoms 1,2,5-7

or 

python compare_top_charges.py POPC_TOPS/POPC_am1.top POPC_TOPS/POPC.top   \                                        (base) 
--molecule POPC \
--sn1 37-52 \
--sn2 16-33 \
--plot charge_compare.png

"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SECTION_RE = re.compile(r"^\s*\[\s*([^\]]+?)\s*\]\s*(?:;.*)?$")


@dataclass(frozen=True)
class AtomRecord:
    index: int
    atom_type: str
    residue_number: str
    residue_name: str
    atom_name: str
    charge_group: str
    charge: float
    mass: str | None
    molecule_type: str
    line_number: int


def strip_comment(line: str) -> str:
    return line.split(";", 1)[0].strip()


def parse_topology(path: Path) -> list[AtomRecord]:
    """Return atom records from all [ atoms ] sections in a topology file."""
    records: list[AtomRecord] = []
    section: str | None = None
    current_molecule = "<unknown>"
    expecting_molecule_name = False

    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise SystemExit(f"ERROR: Could not read {path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        section_match = SECTION_RE.match(raw_line)
        if section_match:
            section = section_match.group(1).strip().lower()
            expecting_molecule_name = section == "moleculetype"
            continue

        line = strip_comment(raw_line)
        if not line:
            continue

        if expecting_molecule_name:
            current_molecule = line.split()[0]
            expecting_molecule_name = False
            continue

        if section != "atoms":
            continue

        fields = line.split()
        if len(fields) < 7:
            print(
                f"WARNING: Skipping short [ atoms ] row in {path} at line "
                f"{line_number}: {raw_line}",
                file=sys.stderr,
            )
            continue

        try:
            atom_index = int(fields[0])
            charge = float(fields[6])
        except ValueError:
            print(
                f"WARNING: Skipping unparseable [ atoms ] row in {path} at "
                f"line {line_number}: {raw_line}",
                file=sys.stderr,
            )
            continue

        records.append(
            AtomRecord(
                index=atom_index,
                atom_type=fields[1],
                residue_number=fields[2],
                residue_name=fields[3],
                atom_name=fields[4],
                charge_group=fields[5],
                charge=charge,
                mass=fields[7] if len(fields) > 7 else None,
                molecule_type=current_molecule,
                line_number=line_number,
            )
        )

    if not records:
        raise SystemExit(f"ERROR: No [ atoms ] records found in {path}")

    return records


def parse_indices(values: Iterable[str]) -> list[int]:
    """Parse atom indices from values like '1 2 5-8' or '1,2,5-8'."""
    indices: list[int] = []
    seen: set[int] = set()

    tokens: list[str] = []
    for value in values:
        tokens.extend(part for part in value.split(",") if part.strip())

    for token in tokens:
        token = token.strip()
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise SystemExit(f"ERROR: Invalid atom index range: {token}") from exc
            if start <= 0 or end <= 0:
                raise SystemExit(f"ERROR: Atom indices must be positive: {token}")
            step = 1 if start <= end else -1
            expanded = range(start, end + step, step)
        else:
            try:
                atom_index = int(token)
            except ValueError as exc:
                raise SystemExit(f"ERROR: Invalid atom index: {token}") from exc
            if atom_index <= 0:
                raise SystemExit(f"ERROR: Atom indices must be positive: {token}")
            expanded = [atom_index]

        for atom_index in expanded:
            if atom_index not in seen:
                seen.add(atom_index)
                indices.append(atom_index)

    if not indices:
        raise SystemExit("ERROR: No atom indices were supplied")

    return indices


def first_molecule(records: list[AtomRecord]) -> str:
    return records[0].molecule_type


def filter_records(
    records: list[AtomRecord],
    path: Path,
    molecule: str | None,
    all_molecules: bool,
) -> tuple[list[AtomRecord], str]:
    if all_molecules:
        return records, "all molecules"

    selected_molecule = molecule or first_molecule(records)
    filtered = [record for record in records if record.molecule_type == selected_molecule]
    if not filtered:
        molecules = ", ".join(sorted({record.molecule_type for record in records}))
        raise SystemExit(
            f"ERROR: Molecule '{selected_molecule}' was not found in {path}. "
            f"Available molecule types: {molecules}"
        )

    return filtered, selected_molecule


def by_index(records: list[AtomRecord]) -> dict[int, list[AtomRecord]]:
    lookup: dict[int, list[AtomRecord]] = {}
    for record in records:
        lookup.setdefault(record.index, []).append(record)
    return lookup


def one_record(
    lookup: dict[int, list[AtomRecord]],
    atom_index: int,
    path: Path,
    selected_scope: str,
) -> AtomRecord | None:
    matches = lookup.get(atom_index, [])
    if not matches:
        return None
    if len(matches) > 1:
        details = ", ".join(
            f"{record.molecule_type}:{record.atom_name}@line{record.line_number}"
            for record in matches
        )
        raise SystemExit(
            f"ERROR: Atom index {atom_index} is ambiguous in {path} "
            f"({selected_scope}): {details}. Use --molecule or omit "
            "--all-molecules to restrict the comparison."
        )
    return matches[0]


def format_charge(value: float | None, precision: int) -> str:
    if value is None:
        return "MISSING"
    return f"{value:.{precision}f}"


def compare_records(
    indices: list[int],
    records_a: list[AtomRecord],
    records_b: list[AtomRecord],
    path_a: Path,
    path_b: Path,
    scope_a: str,
    scope_b: str,
    tolerance: float,
    group: str = "",
) -> list[dict[str, object]]:
    lookup_a = by_index(records_a)
    lookup_b = by_index(records_b)
    rows: list[dict[str, object]] = []

    for atom_index in indices:
        record_a = one_record(lookup_a, atom_index, path_a, scope_a)
        record_b = one_record(lookup_b, atom_index, path_b, scope_b)

        charge_a = record_a.charge if record_a else None
        charge_b = record_b.charge if record_b else None
        delta = None if charge_a is None or charge_b is None else charge_b - charge_a

        notes: list[str] = []
        if record_a is None:
            notes.append("missing in topology A")
        if record_b is None:
            notes.append("missing in topology B")
        if record_a and record_b:
            if record_a.atom_name != record_b.atom_name:
                notes.append(f"atom name differs ({record_a.atom_name} vs {record_b.atom_name})")
            if record_a.residue_name != record_b.residue_name:
                notes.append(
                    f"residue differs ({record_a.residue_name} vs {record_b.residue_name})"
                )
            if abs(delta or 0.0) <= tolerance:
                notes.append("same within tolerance")
            else:
                notes.append("different")

        rows.append(
            {
                "group": group,
                "index": atom_index,
                "molecule_a": record_a.molecule_type if record_a else "",
                "atom_a": record_a.atom_name if record_a else "",
                "residue_a": record_a.residue_name if record_a else "",
                "charge_a": charge_a,
                "molecule_b": record_b.molecule_type if record_b else "",
                "atom_b": record_b.atom_name if record_b else "",
                "residue_b": record_b.residue_name if record_b else "",
                "charge_b": charge_b,
                "delta_b_minus_a": delta,
                "abs_delta": None if delta is None else abs(delta),
                "notes": "; ".join(notes),
            }
        )

    return rows


def print_table(rows: list[dict[str, object]], label_a: str, label_b: str, precision: int) -> None:
    include_group = any(row.get("group") for row in rows)
    headers = [
        "mol_A",
        "atom_A",
        "res_A",
        f"charge_A ({label_a})",
        "mol_B",
        "atom_B",
        "res_B",
        f"charge_B ({label_b})",
        "delta_B-A",
        "notes",
    ]
    if include_group:
        headers.insert(0, "group")
    headers.insert(0, "index")

    printable_rows: list[list[str]] = []
    for row in rows:
        printable_row = [
            str(row["index"]),
            str(row["molecule_a"]),
            str(row["atom_a"]),
            str(row["residue_a"]),
            format_charge(row["charge_a"], precision),
            str(row["molecule_b"]),
            str(row["atom_b"]),
            str(row["residue_b"]),
            format_charge(row["charge_b"], precision),
            format_charge(row["delta_b_minus_a"], precision),
            str(row["notes"]),
        ]
        if include_group:
            printable_row.insert(1, str(row.get("group", "")))
        printable_rows.append(printable_row)

    widths = [
        max(len(header), *(len(row[column]) for row in printable_rows))
        for column, header in enumerate(headers)
    ]

    def emit(values: list[str]) -> None:
        print("  ".join(value.ljust(widths[column]) for column, value in enumerate(values)))

    emit(headers)
    emit(["-" * width for width in widths])
    for printable_row in printable_rows:
        emit(printable_row)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "group",
        "index",
        "molecule_a",
        "atom_a",
        "residue_a",
        "charge_a",
        "molecule_b",
        "atom_b",
        "residue_b",
        "charge_b",
        "delta_b_minus_a",
        "abs_delta",
        "notes",
    ]
    try:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise SystemExit(f"ERROR: Could not write CSV file {path}: {exc}") from exc


def setup_matplotlib():
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache_dir = Path(tempfile.gettempdir()) / "compare_top_charges_cache"
        xdg_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path(tempfile.gettempdir()) / "compare_top_charges_mpl"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "ERROR: Plotting requires matplotlib. Install it with "
            "'python3 -m pip install matplotlib' or run without --plot."
        ) from exc

    return plt


def numeric_or_nan(value: object) -> float:
    return float("nan") if value is None else float(value)


def finite_numbers(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def padded_limits(values: Iterable[float], minimum_span: float = 0.02) -> tuple[float, float]:
    finite_values = finite_numbers(values)
    if not finite_values:
        return (-minimum_span / 2.0, minimum_span / 2.0)

    lower = min(finite_values)
    upper = max(finite_values)
    span = upper - lower
    if span < minimum_span:
        midpoint = (lower + upper) / 2.0
        lower = midpoint - minimum_span / 2.0
        upper = midpoint + minimum_span / 2.0
        span = minimum_span

    padding = span * 0.08
    return (lower - padding, upper + padding)


def symmetric_delta_limits(
    values: Iterable[float], tolerance: float, minimum_half_range: float = 1e-5
) -> tuple[float, float]:
    finite_values = finite_numbers(values)
    max_abs = max((abs(value) for value in finite_values), default=0.0)
    half_range = max(max_abs, tolerance, minimum_half_range)
    half_range *= 1.20
    return (-half_range, half_range)


def validate_axis_limits(values: list[float] | None, name: str) -> tuple[float, float] | None:
    if values is None:
        return None

    lower, upper = values
    if lower >= upper:
        raise SystemExit(f"ERROR: {name} requires MIN to be less than MAX")

    return (lower, upper)


def default_plot_label(path: Path) -> str:
    if path.suffix.lower() == ".top":
        return path.stem
    return path.name


def atom_label(row: dict[str, object], tail_position: int) -> str:
    atom_name = row.get("atom_a") or row.get("atom_b") or ""
    return f"{tail_position}\n{atom_name}"


def plot_tail_charges(
    rows: list[dict[str, object]],
    output_path: Path,
    label_a: str,
    label_b: str,
    title: str | None,
    tolerance: float,
    charge_ylim: tuple[float, float] | None,
    delta_ylim: tuple[float, float] | None,
) -> None:
    tail_groups = [
        group for group in ("SN1", "SN2") if any(row.get("group") == group for row in rows)
    ]
    if not tail_groups:
        raise SystemExit("ERROR: --plot requires --sn1 and/or --sn2 atom specifications")

    plt = setup_matplotlib()
    figure, axes = plt.subplots(
        nrows=2,
        ncols=len(tail_groups),
        figsize=(7.0 * len(tail_groups), 7.0),
        squeeze=False,
        constrained_layout=True,
    )

    colors = {
        "A": "#1f77b4",
        "B": "#d62728",
        "delta": "#444444",
        "zero": "#888888",
        "tolerance": "#999999",
    }

    for column, group in enumerate(tail_groups):
        group_rows = [row for row in rows if row.get("group") == group]
        x_values = list(range(1, len(group_rows) + 1))
        x_a = [value - 0.035 for value in x_values]
        x_b = [value + 0.035 for value in x_values]
        charge_a = [numeric_or_nan(row["charge_a"]) for row in group_rows]
        charge_b = [numeric_or_nan(row["charge_b"]) for row in group_rows]
        deltas = [numeric_or_nan(row["delta_b_minus_a"]) for row in group_rows]
        tick_labels = [
            atom_label(row, tail_position)
            for tail_position, row in enumerate(group_rows, start=1)
        ]

        charge_axis = axes[0][column]
        charge_axis.plot(
            x_a,
            charge_a,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=colors["A"],
            label=label_a,
        )
        charge_axis.plot(
            x_b,
            charge_b,
            marker="s",
            linewidth=1.8,
            markersize=4,
            color=colors["B"],
            label=label_b,
        )
        charge_axis.set_title(f"{group} Charge Profile")
        charge_axis.set_ylabel("Charge (e)")
        charge_axis.grid(True, alpha=0.25)
        charge_axis.legend()
        charge_axis.set_ylim(charge_ylim or padded_limits([*charge_a, *charge_b]))

        delta_axis = axes[1][column]
        delta_axis.bar(x_values, deltas, color=colors["delta"], alpha=0.8)
        delta_axis.axhline(0.0, color=colors["zero"], linewidth=1.0)
        if tolerance > 0:
            delta_axis.axhline(tolerance, color=colors["tolerance"], linestyle="--", linewidth=0.8)
            delta_axis.axhline(-tolerance, color=colors["tolerance"], linestyle="--", linewidth=0.8)
        delta_axis.set_title(f"{group} Delta ({label_b} - {label_a})")
        delta_axis.set_ylabel("Delta Charge (e)")
        delta_axis.set_xlabel("Tail position / atom name")
        delta_axis.grid(True, axis="y", alpha=0.25)
        delta_axis.set_ylim(delta_ylim or symmetric_delta_limits(deltas, tolerance))

        for axis in (charge_axis, delta_axis):
            axis.set_xticks(x_values)
            axis.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=8)

    figure.suptitle(title or "SN1/SN2 Tail Charge Comparison", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected atom charges between two GROMACS .top/.itp files. "
            "Atom numbers are interpreted as moleculetype-local [ atoms ] indices."
        )
    )
    parser.add_argument("topology_a", type=Path, help="First GROMACS topology file")
    parser.add_argument("topology_b", type=Path, help="Second GROMACS topology file")
    parser.add_argument(
        "-a",
        "--atoms",
        nargs="+",
        help="Atom indices to compare, e.g. --atoms 1 2 5-8 or --atoms 1,2,5-8",
    )
    parser.add_argument(
        "--sn1",
        nargs="+",
        metavar="ATOMS",
        help="SN1 tail atom indices to compare/plot, e.g. --sn1 16-33 or --sn1 16,17,18",
    )
    parser.add_argument(
        "--sn2",
        nargs="+",
        metavar="ATOMS",
        help="SN2 tail atom indices to compare/plot, e.g. --sn2 37-52 or --sn2 37,38,39",
    )
    parser.add_argument(
        "-m",
        "--molecule",
        help=(
            "Molecule type to compare, e.g. POPC or DPPC. Defaults to the first "
            "moleculetype found in each file."
        ),
    )
    parser.add_argument(
        "--all-molecules",
        action="store_true",
        help=(
            "Search all molecule types instead of the selected/first molecule. "
            "This fails if a requested atom index appears more than once."
        ),
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute charge-difference tolerance for reporting sameness (default: 1e-6)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Decimal places to print for charges and deltas (default: 12)",
    )
    parser.add_argument("--csv", type=Path, help="Optional output CSV path")
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional PNG/PDF/SVG plot path for SN1/SN2 charge profiles and B-A deltas",
    )
    parser.add_argument("--plot-title", help="Optional title for the generated plot")
    parser.add_argument(
        "--charge-ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Manual y-axis limits for charge-profile panels, e.g. --charge-ylim -0.2 0.1",
    )
    parser.add_argument(
        "--delta-ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Manual y-axis limits for delta panels, e.g. --delta-ylim -0.01 0.01",
    )
    parser.add_argument("--label-a", help="Display label for topology A in tables/plots")
    parser.add_argument("--label-b", help="Display label for topology B in tables/plots")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    comparison_groups: list[tuple[str, list[int]]] = []
    if args.atoms:
        comparison_groups.append(("", parse_indices(args.atoms)))
    if args.sn1:
        comparison_groups.append(("SN1", parse_indices(args.sn1)))
    if args.sn2:
        comparison_groups.append(("SN2", parse_indices(args.sn2)))

    if not comparison_groups:
        raise SystemExit("ERROR: Supply --atoms and/or --sn1/--sn2 atom specifications")
    if args.plot and not (args.sn1 or args.sn2):
        raise SystemExit("ERROR: --plot requires --sn1 and/or --sn2 atom specifications")

    if args.molecule and args.all_molecules:
        raise SystemExit("ERROR: Use either --molecule or --all-molecules, not both")
    if args.tolerance < 0:
        raise SystemExit("ERROR: --tolerance must be non-negative")
    if args.precision < 0:
        raise SystemExit("ERROR: --precision must be non-negative")
    charge_ylim = validate_axis_limits(args.charge_ylim, "--charge-ylim")
    delta_ylim = validate_axis_limits(args.delta_ylim, "--delta-ylim")

    records_a = parse_topology(args.topology_a)
    records_b = parse_topology(args.topology_b)
    filtered_a, scope_a = filter_records(
        records_a, args.topology_a, args.molecule, args.all_molecules
    )
    filtered_b, scope_b = filter_records(
        records_b, args.topology_b, args.molecule, args.all_molecules
    )

    if args.molecule is None and not args.all_molecules and scope_a != scope_b:
        print(
            "WARNING: Default molecule choices differ: "
            f"{args.topology_a.name} -> {scope_a}, {args.topology_b.name} -> {scope_b}. "
            "Use --molecule to choose explicitly.",
            file=sys.stderr,
        )

    rows: list[dict[str, object]] = []
    for group, indices in comparison_groups:
        rows.extend(
            compare_records(
                indices=indices,
                records_a=filtered_a,
                records_b=filtered_b,
                path_a=args.topology_a,
                path_b=args.topology_b,
                scope_a=scope_a,
                scope_b=scope_b,
                tolerance=args.tolerance,
                group=group,
            )
        )

    label_a = args.label_a or args.topology_a.name
    label_b = args.label_b or args.topology_b.name
    plot_label_a = args.label_a or default_plot_label(args.topology_a)
    plot_label_b = args.label_b or default_plot_label(args.topology_b)
    print(f"Compared molecule scope: A={scope_a}, B={scope_b}")
    print_table(rows, label_a, label_b, args.precision)

    comparable_rows = [
        row for row in rows if row["delta_b_minus_a"] is not None
    ]
    different = [
        row
        for row in comparable_rows
        if float(row["abs_delta"]) > args.tolerance
    ]
    missing = len(rows) - len(comparable_rows)
    print()
    print(
        f"Summary: {len(comparable_rows)} compared, {len(different)} different "
        f"above tolerance {args.tolerance:g}, {missing} missing."
    )

    if args.csv:
        write_csv(args.csv, rows)
        print(f"Wrote CSV: {args.csv}")

    if args.plot:
        plot_tail_charges(
            rows,
            args.plot,
            plot_label_a,
            plot_label_b,
            args.plot_title,
            args.tolerance,
            charge_ylim,
            delta_ylim,
        )
        print(f"Wrote plot: {args.plot}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
