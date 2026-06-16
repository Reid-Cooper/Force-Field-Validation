#!/usr/bin/env python3
"""
Calculate lipid-bilayer surface tension from pressure tensor components.

The calculations are as follows:

    pressure_anisotropy = <Pzz> - ( <Pxx> + <Pyy> ) / 2
        units: pressure_unit, bar by default

    Native value = gamma_native = 0.5 * <Lz> * pressure_anisotropy
        units: pressure_unit*length_unit, bar*nm by default

    Surface tension = gamma = gamma_native * pressure_to_Pa * length_to_m
        units: N/m

    Reported surface tension = gamma converted to --output-unit
        units: mN/m by default

For frame-by-frame data:

    gamma_i = 0.5 * Lz_i * ( Pzz_i - ( Pxx_i + Pyy_i ) / 2 )

    Frame-averaged instantaneous gamma = mean(gamma_i)
        units: --output-unit, mN/m by default

    Frame SD = sample standard deviation of gamma_i
        units: --output-unit, mN/m by default

    Frame standard error = Frame SD / sqrt(number_of_frames)
        units: --output-unit, mN/m by default

where Lz is the simulation-box height normal to the bilayer, Pzz is the
normal pressure, and Pxx/Pyy are lateral pressures.

Examples
--------
Direct averaged values, using the default pressure unit of bar and length
unit of nm:

    python calc_surface_tension.py --lz 12.0 --pxx -15 --pyy -20 --pzz 35

(Recommended use): Use GROMACS to extract pressure tensor terms from an energy file which should have been created when the simulation was run:

    python calc_surface_tension.py --edr ener.edr

---------------------------------------------------------------------------------------------------------------------

IF NEEDED (Not recommended): Use GROMACS to rerun a trajectory and create the pressure tensor terms. A
.top and .xtc are not enough by themselves because XTC files do not store
forces, velocities, atom names, or pressure tensor data. If you do not already
have a .tpr file, provide the .mdp, .top, and starting .gro/.pdb structure
needed by gmx grompp. The rerun is serial by default to avoid thread-MPI
collective mismatch errors during analysis:

    python calc_surface_tension.py --xtc traj.xtc --tpr topol.tpr

    python calc_surface_tension.py --xtc traj.xtc --top topol.top \
        --mdp production.mdp --structure conf.gro

Read columns from a whitespace, CSV, or GROMACS XVG file. Column numbers are
1-based:

    python calc_surface_tension.py pressure_box.xvg --lz-col 2 --pxx-col 3 --pyy-col 4 --pzz-col 5

If the input file has column headers or GROMACS legends such as Box-Z,
Pres-XX, Pres-YY, and Pres-ZZ, the script will try to infer the columns:

    python calc_surface_tension.py pressure_box.xvg
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PRESSURE_TO_PA = {
    "pa": 1.0,
    "kpa": 1.0e3,
    "mpa": 1.0e6,
    "gpa": 1.0e9,
    "bar": 1.0e5,
    "atm": 101325.0,
}

LENGTH_TO_M = {
    "m": 1.0,
    "cm": 1.0e-2,
    "mm": 1.0e-3,
    "um": 1.0e-6,
    "micrometer": 1.0e-6,
    "nm": 1.0e-9,
    "angstrom": 1.0e-10,
    "a": 1.0e-10,
}

OUTPUT_FROM_N_PER_M = {
    "n/m": 1.0,
    "mn/m": 1.0e3,
    "dyne/cm": 1.0e3,
}

UNIT_DISPLAY = {
    "pa": "Pa",
    "kpa": "kPa",
    "mpa": "MPa",
    "gpa": "GPa",
    "bar": "bar",
    "atm": "atm",
    "m": "m",
    "cm": "cm",
    "mm": "mm",
    "um": "um",
    "micrometer": "micrometer",
    "nm": "nm",
    "angstrom": "angstrom",
    "a": "angstrom",
    "n/m": "N/m",
    "mn/m": "mN/m",
    "dyne/cm": "dyne/cm",
}

ALIASES = {
    "lz": {
        "lz",
        "boxz",
        "box-z",
        "box_z",
        "boxheightz",
        "height",
        "zbox",
    },
    "pxx": {
        "pxx",
        "p_xx",
        "presxx",
        "pres-xx",
        "pressurexx",
        "pressure_xx",
        "pressuretensorxx",
    },
    "pyy": {
        "pyy",
        "p_yy",
        "presyy",
        "pres-yy",
        "pressureyy",
        "pressure_yy",
        "pressuretensoryy",
    },
    "pzz": {
        "pzz",
        "p_zz",
        "preszz",
        "pres-zz",
        "pressurezz",
        "pressure_zz",
        "pressuretensorzz",
    },
    "time": {
        "time",
        "t",
        "step",
    },
}


@dataclass(frozen=True)
class TableData:
    path: Path
    labels: list[str]
    rows: list[list[float]]


@dataclass(frozen=True)
class SurfaceTensionResult:
    n_rows: int
    length_unit: str
    pressure_unit: str
    output_unit: str
    mean_lz: float
    mean_pxx: float
    mean_pyy: float
    mean_pzz: float
    anisotropy_native: float
    gamma_native: float
    gamma_n_per_m: float
    gamma_output: float
    frame_mean_output: float | None
    frame_std_output: float | None
    frame_sem_output: float | None


def normalize_unit(unit: str) -> str:
    return unit.strip().lower().replace(" ", "").replace("micro", "u").replace("µ", "u")


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def validate_unit(unit: str, known_units: dict[str, float], unit_type: str) -> str:
    normalized = normalize_unit(unit)
    if normalized not in known_units:
        choices = ", ".join(sorted(known_units))
        raise ValueError(f"Unknown {unit_type} unit '{unit}'. Valid choices: {choices}")
    return normalized


def unit_label(unit: str) -> str:
    return UNIT_DISPLAY.get(unit, unit)


def mean(values: list[float], name: str) -> float:
    if not values:
        raise ValueError(f"No values available for {name}.")
    return statistics.fmean(values)


def split_data_line(line: str) -> list[str]:
    if "," in line:
        return [token.strip().strip("\"'") for token in line.split(",") if token.strip()]
    return [token.strip().strip("\"'") for token in line.split() if token.strip()]


def parse_float_row(tokens: list[str]) -> list[float] | None:
    row: list[float] = []
    for token in tokens:
        try:
            row.append(float(token))
        except ValueError:
            return None
    return row


def parse_table(path: Path) -> TableData:
    rows: list[list[float]] = []
    labels: list[str] | None = None
    xvg_legends: dict[int, str] = {}
    legend_re = re.compile(r"@\s+s(\d+)\s+legend\s+\"([^\"]+)\"")

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("@"):
                match = legend_re.search(line)
                if match:
                    xvg_legends[int(match.group(1))] = match.group(2)
                continue

            if line.startswith("#"):
                continue

            tokens = split_data_line(line)
            if not tokens:
                continue

            numeric_row = parse_float_row(tokens)
            if numeric_row is None:
                if labels is None:
                    labels = tokens
                continue

            rows.append(numeric_row)

    if not rows:
        raise ValueError(f"No numeric data rows found in {path}.")

    n_columns = max(len(row) for row in rows)
    if labels is None and xvg_legends:
        labels = [f"column_{idx + 1}" for idx in range(n_columns)]
        labels[0] = "time"
        for series_index, legend in xvg_legends.items():
            column_index = series_index + 1
            if column_index < n_columns:
                labels[column_index] = legend

    if labels is None:
        labels = [f"column_{idx + 1}" for idx in range(n_columns)]
    elif len(labels) < n_columns:
        labels = labels + [f"column_{idx + 1}" for idx in range(len(labels), n_columns)]

    return TableData(path=path, labels=labels, rows=rows)


def resolve_column(
    spec: str | None,
    labels: list[str],
    n_columns: int,
    role: str,
    required: bool = True,
) -> int | None:
    if spec:
        try:
            column_number = int(spec)
        except ValueError:
            normalized_spec = normalize_label(spec)
            for index, label in enumerate(labels):
                if normalize_label(label) == normalized_spec:
                    return index
            raise ValueError(f"Could not find a column named '{spec}' for {role}.")

        if column_number < 1 or column_number > n_columns:
            raise ValueError(
                f"{role} column {column_number} is outside the available 1-{n_columns} range."
            )
        return column_number - 1

    role_aliases = {normalize_label(alias) for alias in ALIASES[role]}
    for index, label in enumerate(labels):
        if normalize_label(label) in role_aliases:
            return index

    if required:
        label_list = ", ".join(f"{idx + 1}:{label}" for idx, label in enumerate(labels))
        raise ValueError(
            f"Could not infer the {role} column. Use --{role}-col to specify it. "
            f"Available columns: {label_list}"
        )

    return None


def column_values(rows: list[list[float]], column_index: int, role: str) -> list[float]:
    values: list[float] = []
    for row_number, row in enumerate(rows, start=1):
        if column_index >= len(row):
            raise ValueError(
                f"Row {row_number} does not have column {column_index + 1} needed for {role}."
            )
        values.append(row[column_index])
    return values


def select_rows(
    table: TableData,
    time_col: str | None,
    start: float | None,
    end: float | None,
) -> list[list[float]]:
    if start is None and end is None:
        return table.rows

    time_index = resolve_column(
        time_col or "1",
        table.labels,
        max(len(row) for row in table.rows),
        "time",
        required=True,
    )
    assert time_index is not None

    selected: list[list[float]] = []
    for row in table.rows:
        if time_index >= len(row):
            continue
        time_value = row[time_index]
        if start is not None and time_value < start:
            continue
        if end is not None and time_value > end:
            continue
        selected.append(row)

    if not selected:
        raise ValueError("No rows remain after applying --start/--end filtering.")
    return selected


def repeated_value(value: float, n_values: int) -> list[float]:
    return [value for _ in range(n_values)]


def collect_values_from_table(
    args: argparse.Namespace,
    table: TableData,
) -> tuple[list[float], list[float], list[float], list[float], TableData]:
    rows = select_rows(table, args.time_col, args.start, args.end)
    n_columns = max(len(row) for row in rows)
    n_rows = len(rows)

    def values_for(role: str) -> list[float]:
        direct_value = getattr(args, role)
        if direct_value is not None:
            return repeated_value(direct_value, n_rows)

        column_spec = getattr(args, f"{role}_col")
        column_index = resolve_column(column_spec, table.labels, n_columns, role, required=True)
        assert column_index is not None
        return column_values(rows, column_index, role)

    return values_for("lz"), values_for("pxx"), values_for("pyy"), values_for("pzz"), table


def collect_values(args: argparse.Namespace) -> tuple[list[float], list[float], list[float], list[float], TableData | None]:
    table: TableData | None = None

    if args.input is None:
        missing = [
            name
            for name in ("lz", "pxx", "pyy", "pzz")
            if getattr(args, name) is None
        ]
        if missing:
            missing_args = ", ".join(f"--{name}" for name in missing)
            raise ValueError(f"Direct-value mode requires {missing_args}.")
        return [args.lz], [args.pxx], [args.pyy], [args.pzz], None

    table = parse_table(args.input)
    return collect_values_from_table(args, table)


def require_existing_path(path: Path | None, description: str) -> Path:
    if path is None:
        raise ValueError(f"{description} is required.")
    if not path.exists():
        raise ValueError(f"{description} does not exist: {path}")
    return path


def resolve_gromacs_executable(gmx: str) -> str:
    executable = shutil.which(gmx)
    if executable is None:
        raise ValueError(
            f"Could not find GROMACS executable '{gmx}'. Install GROMACS, load its "
            "environment, or pass --gmx with the executable name/path."
        )
    return str(Path(executable).resolve())


def command_tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return "... output truncated ...\n" + text[-max_chars:]


def run_gromacs_command(
    command: list[str],
    input_text: str | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            cwd=cwd,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"Could not run {' '.join(command)}: {exc}") from exc

    if completed.returncode != 0:
        parts = [
            f"GROMACS command failed with exit code {completed.returncode}:",
            shlex.join(command),
        ]
        if completed.stdout:
            parts.extend(["stdout:", command_tail(completed.stdout)])
        if completed.stderr:
            parts.extend(["stderr:", command_tail(completed.stderr)])
        raise ValueError("\n".join(parts))

    return completed


def build_tpr_with_grompp(args: argparse.Namespace, gmx: str, work_dir: Path) -> Path:
    if args.tpr is not None:
        return require_existing_path(args.tpr, ".tpr file")

    missing = [
        option
        for option, value in (
            ("--top", args.top),
            ("--mdp", args.mdp),
            ("--structure", args.structure),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "To rerun an .xtc without --tpr, provide the files needed by "
            f"gmx grompp: {', '.join(missing)}. A .top and .xtc alone are not "
            "enough to build a runnable system because the trajectory does not "
            "store atom names or topology settings."
        )

    top = require_existing_path(args.top, ".top file")
    mdp = require_existing_path(args.mdp, ".mdp file")
    structure = require_existing_path(args.structure, "starting structure")
    output_tpr = work_dir / "surface_tension_rerun.tpr"

    command = [
        gmx,
        "grompp",
        "-f",
        str(mdp.resolve()),
        "-c",
        str(structure.resolve()),
        "-p",
        str(top.resolve()),
        "-o",
        str(output_tpr.resolve()),
        "-po",
        str((work_dir / "surface_tension_grompp_out.mdp").resolve()),
    ]
    if args.index is not None:
        command.extend(["-n", str(require_existing_path(args.index, "index file").resolve())])
    if args.maxwarn:
        command.extend(["-maxwarn", str(args.maxwarn)])

    run_gromacs_command(command, cwd=top.parent)
    return output_tpr


def rerun_trajectory_with_gromacs(args: argparse.Namespace, gmx: str, work_dir: Path) -> Path:
    xtc = require_existing_path(args.xtc, ".xtc trajectory")
    tpr = build_tpr_with_grompp(args, gmx, work_dir)
    output_prefix = work_dir / "surface_tension_rerun"
    output_edr = output_prefix.with_suffix(".edr")

    command = [
        gmx,
        "mdrun",
    ]
    if args.rerun_ntmpi > 0:
        command.extend(["-ntmpi", str(args.rerun_ntmpi)])
    if args.rerun_ntomp > 0:
        command.extend(["-ntomp", str(args.rerun_ntomp)])
    command.extend(
        [
        "-s",
        str(tpr.resolve()),
        "-rerun",
        str(xtc.resolve()),
        "-deffnm",
        str(output_prefix.resolve()),
        ]
    )
    run_gromacs_command(command, cwd=work_dir)

    if not output_edr.exists():
        raise ValueError(f"GROMACS rerun finished but did not create {output_edr}.")
    return output_edr


def extract_pressure_terms_with_gmx_energy(
    gmx: str,
    edr: Path,
    output_xvg: Path,
) -> Path:
    terms = ["Box-Z", "Pres-XX", "Pres-YY", "Pres-ZZ"]
    selection = "\n".join(terms + ["0", ""])
    command = [
        gmx,
        "energy",
        "-f",
        str(edr.resolve()),
        "-o",
        str(output_xvg.resolve()),
    ]
    run_gromacs_command(command, input_text=selection, cwd=output_xvg.parent)

    if not output_xvg.exists():
        raise ValueError(f"gmx energy finished but did not create {output_xvg}.")
    return output_xvg


def using_gromacs_inputs(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.edr,
            args.xtc,
            args.tpr,
            args.top,
            args.mdp,
            args.structure,
        )
    )


def collect_values_from_gromacs(
    args: argparse.Namespace,
) -> tuple[list[float], list[float], list[float], list[float], TableData]:
    if args.input is not None:
        raise ValueError("Use either a table/XVG positional input or GROMACS inputs, not both.")

    gmx = resolve_gromacs_executable(args.gmx)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.edr is not None:
        edr = require_existing_path(args.edr, ".edr energy file")
    else:
        if args.xtc is None:
            raise ValueError("GROMACS rerun mode requires --xtc and either --tpr or --top/--mdp/--structure.")
        edr = rerun_trajectory_with_gromacs(args, gmx, work_dir)

    output_xvg = args.energy_xvg or (work_dir / "surface_tension_pressure_tensor.xvg")
    output_xvg.parent.mkdir(parents=True, exist_ok=True)
    xvg = extract_pressure_terms_with_gmx_energy(gmx, edr, output_xvg)
    table = parse_table(xvg)
    return collect_values_from_table(args, table)


def calculate_surface_tension(
    lz_values: list[float],
    pxx_values: list[float],
    pyy_values: list[float],
    pzz_values: list[float],
    length_unit: str,
    pressure_unit: str,
    output_unit: str,
) -> SurfaceTensionResult:
    if not (len(lz_values) == len(pxx_values) == len(pyy_values) == len(pzz_values)):
        raise ValueError("Lz, Pxx, Pyy, and Pzz must have the same number of values.")

    length_unit = validate_unit(length_unit, LENGTH_TO_M, "length")
    pressure_unit = validate_unit(pressure_unit, PRESSURE_TO_PA, "pressure")
    output_unit = validate_unit(output_unit, OUTPUT_FROM_N_PER_M, "output")

    mean_lz = mean(lz_values, "Lz")
    mean_pxx = mean(pxx_values, "Pxx")
    mean_pyy = mean(pyy_values, "Pyy")
    mean_pzz = mean(pzz_values, "Pzz")

    anisotropy_native = mean_pzz - 0.5 * (mean_pxx + mean_pyy)
    gamma_native = 0.5 * mean_lz * anisotropy_native
    gamma_n_per_m = gamma_native * LENGTH_TO_M[length_unit] * PRESSURE_TO_PA[pressure_unit]
    gamma_output = gamma_n_per_m * OUTPUT_FROM_N_PER_M[output_unit]

    frame_mean_output: float | None = None
    frame_std_output: float | None = None
    frame_sem_output: float | None = None

    if len(lz_values) > 1:
        frame_values = [
            0.5 * lz * (pzz - 0.5 * (pxx + pyy))
            * LENGTH_TO_M[length_unit]
            * PRESSURE_TO_PA[pressure_unit]
            * OUTPUT_FROM_N_PER_M[output_unit]
            for lz, pxx, pyy, pzz in zip(lz_values, pxx_values, pyy_values, pzz_values)
        ]
        frame_mean_output = statistics.fmean(frame_values)
        frame_std_output = statistics.stdev(frame_values)
        frame_sem_output = frame_std_output / math.sqrt(len(frame_values))

    return SurfaceTensionResult(
        n_rows=len(lz_values),
        length_unit=length_unit,
        pressure_unit=pressure_unit,
        output_unit=output_unit,
        mean_lz=mean_lz,
        mean_pxx=mean_pxx,
        mean_pyy=mean_pyy,
        mean_pzz=mean_pzz,
        anisotropy_native=anisotropy_native,
        gamma_native=gamma_native,
        gamma_n_per_m=gamma_n_per_m,
        gamma_output=gamma_output,
        frame_mean_output=frame_mean_output,
        frame_std_output=frame_std_output,
        frame_sem_output=frame_sem_output,
    )


def format_number(value: float, precision: int) -> str:
    return f"{value:.{precision}g}"


def build_report(
    result: SurfaceTensionResult,
    precision: int,
    input_path: Path | None,
) -> str:
    p = precision
    length_unit = unit_label(result.length_unit)
    pressure_unit = unit_label(result.pressure_unit)
    output_unit = unit_label(result.output_unit)
    lines = [
        "-----------------------------------------",
        "Lipid Bilayer Surface Tension",
        "Equation: gamma = 0.5 * Lz * (<Pzz> - (<Pxx> + <Pyy>) / 2)",
        "-----------------------------------------",
    ]

    if input_path is not None:
        lines.append(f"Input file = {input_path}")

    lines.extend(
        [
            f"Rows used = {result.n_rows}",
            f"<Lz> = {format_number(result.mean_lz, p)} {length_unit}",
            f"<Pxx> = {format_number(result.mean_pxx, p)} {pressure_unit}",
            f"<Pyy> = {format_number(result.mean_pyy, p)} {pressure_unit}",
            f"<Pzz> = {format_number(result.mean_pzz, p)} {pressure_unit}",
            (
                "Pressure anisotropy = "
                f"{format_number(result.anisotropy_native, p)} {pressure_unit}"
            ),
            "-----------------------------------------",
            (
                "Surface tension = "
                f"{format_number(result.gamma_output, p)} {output_unit}"
            ),
            f"Surface tension = {format_number(result.gamma_n_per_m, p)} N/m",
            (
                "Native value = "
                f"{format_number(result.gamma_native, p)} "
                f"{pressure_unit}*{length_unit}"
            ),
        ]
    )

    if result.frame_mean_output is not None:
        lines.extend(
            [
                "-----------------------------------------",
                (
                    "Frame-averaged instantaneous gamma = "
                    f"{format_number(result.frame_mean_output, p)} {output_unit}"
                ),
                (
                    "Frame standard deviation = "
                    f"{format_number(result.frame_std_output or 0.0, p)} {output_unit}"
                ),
                (
                    "Frame standard error = "
                    f"{format_number(result.frame_sem_output or 0.0, p)} {output_unit}"
                ),
            ]
        )

    lines.append("-----------------------------------------")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate lipid-bilayer surface tension from Lz, Pxx, Pyy, and Pzz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Optional whitespace, CSV, or GROMACS XVG table containing Lz/Pxx/Pyy/Pzz.",
    )

    parser.add_argument("--lz", type=float, help="Simulation box height normal to the bilayer.")
    parser.add_argument("--pxx", type=float, help="Average or constant Pxx lateral pressure.")
    parser.add_argument("--pyy", type=float, help="Average or constant Pyy lateral pressure.")
    parser.add_argument("--pzz", type=float, help="Average or constant Pzz normal pressure.")

    parser.add_argument("--lz-col", help="1-based Lz column number or column name.")
    parser.add_argument("--pxx-col", help="1-based Pxx column number or column name.")
    parser.add_argument("--pyy-col", help="1-based Pyy column number or column name.")
    parser.add_argument("--pzz-col", help="1-based Pzz column number or column name.")

    parser.add_argument("--xtc", type=Path, help="GROMACS trajectory to rerun for pressure tensor terms.")
    parser.add_argument(
        "--tpr",
        type=Path,
        help="GROMACS run input file. Preferred for --xtc rerun mode.",
    )
    parser.add_argument(
        "--edr",
        type=Path,
        help="Existing GROMACS energy file. If provided, no trajectory rerun is needed.",
    )
    parser.add_argument(
        "--top",
        type=Path,
        help="GROMACS topology file, used only to build a .tpr when --tpr is not provided.",
    )
    parser.add_argument(
        "--mdp",
        type=Path,
        help="GROMACS mdp file, used only to build a .tpr when --tpr is not provided.",
    )
    parser.add_argument(
        "--structure",
        type=Path,
        help="Starting .gro/.pdb structure, required with --top/--mdp if --tpr is not provided.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="Optional GROMACS index file passed to gmx grompp.",
    )
    parser.add_argument(
        "--gmx",
        default="gmx",
        help="GROMACS executable name or path, e.g. gmx, gmx_mpi, or /path/to/gmx.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("surface_tension_gmx_work"),
        help="Directory for generated .tpr/.edr/.xvg files in GROMACS mode.",
    )
    parser.add_argument(
        "--energy-xvg",
        type=Path,
        help="Optional output path for extracted Box-Z/pressure-tensor XVG data.",
    )
    parser.add_argument(
        "--maxwarn",
        type=int,
        default=0,
        help="Optional gmx grompp -maxwarn value when building a .tpr.",
    )
    parser.add_argument(
        "--rerun-ntmpi",
        type=int,
        default=1,
        help=(
            "Thread-MPI ranks for gmx mdrun -rerun. The default serial rerun "
            "avoids tMPI collective mismatch errors. Use 0 to omit this option."
        ),
    )
    parser.add_argument(
        "--rerun-ntomp",
        type=int,
        default=1,
        help="OpenMP threads for gmx mdrun -rerun. Use 0 to omit this option.",
    )

    parser.add_argument(
        "--time-col",
        help="1-based time column number or column name, used only with --start/--end.",
    )
    parser.add_argument("--start", type=float, help="Only use rows with time >= this value.")
    parser.add_argument("--end", type=float, help="Only use rows with time <= this value.")

    parser.add_argument(
        "--length-unit",
        default="nm",
        help="Unit for Lz values. Common choices: nm, angstrom, m.",
    )
    parser.add_argument(
        "--pressure-unit",
        default="bar",
        help="Unit for pressure values. Common choices: bar, atm, Pa, kPa, MPa.",
    )
    parser.add_argument(
        "--output-unit",
        default="mN/m",
        help="Output surface tension unit. Choices: mN/m, N/m, dyne/cm.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Significant digits to print.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path for saving the text report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if using_gromacs_inputs(args):
            lz_values, pxx_values, pyy_values, pzz_values, table = collect_values_from_gromacs(args)
        else:
            lz_values, pxx_values, pyy_values, pzz_values, table = collect_values(args)
        result = calculate_surface_tension(
            lz_values=lz_values,
            pxx_values=pxx_values,
            pyy_values=pyy_values,
            pzz_values=pzz_values,
            length_unit=args.length_unit,
            pressure_unit=args.pressure_unit,
            output_unit=args.output_unit,
        )
        report = build_report(result, args.precision, table.path if table else None)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(report)

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        except OSError as exc:
            print(f"Error writing output file: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
