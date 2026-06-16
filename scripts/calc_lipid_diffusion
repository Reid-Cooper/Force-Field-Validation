#!/usr/bin/env python3
"""
Calculate lipid-bilayer diffusion coefficients from GROMACS gmx msd output.

Typical workflow
----------------
1. Generate lateral MSD with GROMACS. If the bilayer normal is z:

   gmx msd -f traj.xtc -s topol.tpr -n index.ndx -o msd_lateral.xvg -lateral z

   Select the lipid, molecule, or group whose lateral diffusion you want.

2. Fit the linear part of the MSD curve:

   python gromacs_lipid_diffusion.py fit msd_lateral.xvg --fit-start 10000 --fit-end 50000

For a lipid bilayer, the default is 2D lateral diffusion:

   D = slope(MSD vs time) / (2 * dimensions) = slope / 4

Area per lipid, bilayer thickness, order parameters, and bending modulus are
accepted as metadata for the output report. They are useful for interpreting
the membrane state, but they do not replace the MSD time series.
"""

from __future__ import annotations

import argparse
import math
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TIME_TO_SECONDS = {
    "fs": 1.0e-15,
    "ps": 1.0e-12,
    "ns": 1.0e-9,
    "us": 1.0e-6,
    "mus": 1.0e-6,
    "ms": 1.0e-3,
    "s": 1.0,
}

AREA_TO_M2 = {
    "a^2": 1.0e-20,
    "angstrom^2": 1.0e-20,
    "angstrom2": 1.0e-20,
    "nm^2": 1.0e-18,
    "nm2": 1.0e-18,
    "m^2": 1.0,
    "m2": 1.0,
    "cm^2": 1.0e-4,
    "cm2": 1.0e-4,
}


@dataclass(frozen=True)
class XvgData:
    path: Path
    x_label: str | None
    y_label: str | None
    rows: list[list[float]]


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    slope_stderr: float
    intercept_stderr: float
    r_squared: float
    n_points: int
    x_min: float
    x_max: float


@dataclass(frozen=True)
class DiffusionResult:
    fit: LinearFit
    dimensions: int
    time_unit: str
    msd_unit: str
    d_native: float
    d_native_stderr: float
    d_m2_s: float
    d_m2_s_stderr: float
    d_cm2_s: float
    d_cm2_s_stderr: float
    d_nm2_ns: float
    d_nm2_ns_stderr: float


def normalize_unit(text: str) -> str:
    cleaned = (
        text.strip()
        .replace("\\S2\\N", "^2")
        .replace("\\S-2\\N", "^-2")
        .replace("\\s2\\N", "^2")
        .replace("\\N", "")
        .replace(" ", "")
    )
    cleaned = cleaned.replace("A^2", "a^2").replace("Angstrom", "angstrom")
    cleaned = cleaned.replace("micro", "u").replace("µ", "u")
    return cleaned.lower()


def parse_axis_label(line: str) -> tuple[str, str] | None:
    match = re.search(r"@\s+(xaxis|yaxis)\s+label\s+\"([^\"]+)\"", line)
    if not match:
        return None
    return match.group(1), match.group(2)


def unit_from_label(label: str | None, known_units: Iterable[str]) -> str | None:
    if not label:
        return None
    match = re.search(r"\(([^()]*)\)", label)
    if match:
        candidate = normalize_unit(match.group(1))
        if candidate in known_units:
            return candidate

    normalized = normalize_unit(label)
    for unit in known_units:
        if unit in normalized:
            return unit
    return None


def parse_xvg(path: Path) -> XvgData:
    rows: list[list[float]] = []
    x_label: str | None = None
    y_label: str | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("@"):
                parsed_label = parse_axis_label(line)
                if parsed_label:
                    axis, label = parsed_label
                    if axis == "xaxis":
                        x_label = label
                    else:
                        y_label = label
                continue

            if line.startswith("#"):
                continue

            try:
                rows.append([float(value) for value in line.split()])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric data rows found in {path}")
    if len(rows[0]) < 2:
        raise ValueError(f"Expected at least two columns in {path}: time and MSD")

    return XvgData(path=path, x_label=x_label, y_label=y_label, rows=rows)


def select_points(
    xvg: XvgData,
    msd_column: int,
    fit_start: float | None,
    fit_end: float | None,
    auto_fit_fraction: float,
) -> list[tuple[float, float]]:
    if msd_column < 1:
        raise ValueError("--msd-column is 1-based after the time column; use 1 for the first MSD column")

    row_index = msd_column
    max_columns = max(len(row) for row in xvg.rows)
    if row_index >= max_columns:
        raise ValueError(
            f"--msd-column {msd_column} requested column index {row_index}, "
            f"but the file has only {max_columns} numeric columns"
        )

    points = [(row[0], row[row_index]) for row in xvg.rows if len(row) > row_index]
    if not points:
        raise ValueError("No complete time/MSD points found for the requested column")

    if fit_start is None and fit_end is None:
        if not 0.0 < auto_fit_fraction <= 1.0:
            raise ValueError("--auto-fit-fraction must be > 0 and <= 1")
        start_index = int(math.floor(len(points) * (1.0 - auto_fit_fraction)))
        points = points[start_index:]
    else:
        if fit_start is not None:
            points = [(time, msd) for time, msd in points if time >= fit_start]
        if fit_end is not None:
            points = [(time, msd) for time, msd in points if time <= fit_end]

    if len(points) < 3:
        raise ValueError("Need at least 3 points in the fit window")

    return points


def linear_regression(points: list[tuple[float, float]]) -> LinearFit:
    n = len(points)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in points)

    if sxx == 0.0:
        raise ValueError("All selected time values are identical")

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    residuals = [y - (slope * x + intercept) for x, y in points]
    ss_res = sum(residual ** 2 for residual in residuals)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else float("nan")

    if n > 2:
        residual_variance = ss_res / (n - 2)
        slope_stderr = math.sqrt(residual_variance / sxx)
        intercept_stderr = math.sqrt(residual_variance * (1.0 / n + x_mean**2 / sxx))
    else:
        slope_stderr = float("nan")
        intercept_stderr = float("nan")

    return LinearFit(
        slope=slope,
        intercept=intercept,
        slope_stderr=slope_stderr,
        intercept_stderr=intercept_stderr,
        r_squared=r_squared,
        n_points=n,
        x_min=min(xs),
        x_max=max(xs),
    )


def calculate_diffusion(
    fit: LinearFit,
    dimensions: int,
    time_unit: str,
    msd_unit: str,
) -> DiffusionResult:
    if dimensions not in (1, 2, 3):
        raise ValueError("--dimensions must be 1, 2, or 3")

    normalized_time_unit = normalize_unit(time_unit)
    normalized_msd_unit = normalize_unit(msd_unit)

    if normalized_time_unit not in TIME_TO_SECONDS:
        raise ValueError(f"Unsupported time unit: {time_unit}")
    if normalized_msd_unit not in AREA_TO_M2:
        raise ValueError(f"Unsupported MSD unit: {msd_unit}")

    denominator = 2.0 * dimensions
    d_native = fit.slope / denominator
    d_native_stderr = fit.slope_stderr / denominator

    native_to_m2_s = AREA_TO_M2[normalized_msd_unit] / TIME_TO_SECONDS[normalized_time_unit]
    d_m2_s = d_native * native_to_m2_s
    d_m2_s_stderr = d_native_stderr * native_to_m2_s

    return DiffusionResult(
        fit=fit,
        dimensions=dimensions,
        time_unit=normalized_time_unit,
        msd_unit=normalized_msd_unit,
        d_native=d_native,
        d_native_stderr=d_native_stderr,
        d_m2_s=d_m2_s,
        d_m2_s_stderr=d_m2_s_stderr,
        d_cm2_s=d_m2_s * 1.0e4,
        d_cm2_s_stderr=d_m2_s_stderr * 1.0e4,
        d_nm2_ns=d_m2_s / 1.0e-9,
        d_nm2_ns_stderr=d_m2_s_stderr / 1.0e-9,
    )


def fmt_uncertainty(value: float, stderr: float, unit: str) -> str:
    if math.isnan(stderr):
        return f"{value:.6g} {unit}"
    return f"{value:.6g} +/- {stderr:.2g} {unit}"


def optional_line(label: str, value: float | None, unit: str = "") -> str | None:
    if value is None:
        return None
    suffix = f" {unit}" if unit else ""
    return f"{label}: {value:g}{suffix}"


def build_report(
    result: DiffusionResult,
    xvg: XvgData,
    args: argparse.Namespace,
    inferred_time_unit: str | None,
    inferred_msd_unit: str | None,
) -> str:
    native_unit = f"{result.msd_unit}/{result.time_unit}"
    lines = [
        "Lipid bilayer diffusion coefficient from GROMACS MSD",
        "=" * 56,
        f"Input XVG: {xvg.path}",
        f"MSD column: {args.msd_column} (1 = first data column after time)",
        f"Fit window: {result.fit.x_min:g} to {result.fit.x_max:g} {result.time_unit}",
        f"Fit points: {result.fit.n_points}",
        f"Dimensions: {result.dimensions}D",
        "",
        "Linear MSD fit",
        "--------------",
        f"Slope: {fmt_uncertainty(result.fit.slope, result.fit.slope_stderr, native_unit)}",
        f"Intercept: {fmt_uncertainty(result.fit.intercept, result.fit.intercept_stderr, result.msd_unit)}",
        f"R^2: {result.fit.r_squared:.6g}",
        "",
        "Diffusion coefficient",
        "---------------------",
        f"D: {fmt_uncertainty(result.d_native, result.d_native_stderr, native_unit)}",
        f"D: {fmt_uncertainty(result.d_nm2_ns, result.d_nm2_ns_stderr, 'nm^2/ns')}",
        f"D: {fmt_uncertainty(result.d_cm2_s, result.d_cm2_s_stderr, 'cm^2/s')}",
        f"D: {fmt_uncertainty(result.d_m2_s, result.d_m2_s_stderr, 'm^2/s')}",
    ]

    metadata_lines = [
        optional_line("Area per lipid", args.area_per_lipid, "nm^2/lipid"),
        optional_line("Bilayer thickness", args.bilayer_thickness, "nm"),
        optional_line("Order parameter", args.order_parameter),
        optional_line("Bending modulus", args.bending_modulus, args.bending_modulus_unit),
        optional_line("Temperature", args.temperature, "K"),
    ]
    metadata_lines = [line for line in metadata_lines if line is not None]
    if metadata_lines:
        lines.extend(["", "Membrane metadata", "-----------------", *metadata_lines])

    notes = [
        "For lateral lipid diffusion in a flat bilayer, use dimensions=2 and generate the XVG with gmx msd -lateral z when z is the bilayer normal.",
        "The standard error is the ordinary least-squares fit error. MSD points are time-correlated, so independent replicates/block analysis give a better uncertainty estimate.",
    ]
    if args.fit_start is None and args.fit_end is None:
        notes.append(
            f"No fit window was supplied, so the script fit the last {args.auto_fit_fraction:g} fraction of points. Inspect the MSD plot and set --fit-start/--fit-end for publication-quality values."
        )
    if inferred_time_unit is None or inferred_msd_unit is None:
        notes.append("At least one unit was not detected from the XVG labels; command-line defaults or supplied units were used.")
    if args.area_per_lipid or args.bilayer_thickness or args.order_parameter or args.bending_modulus:
        notes.append("The membrane metadata are reported for context only; D is calculated from the MSD slope.")

    lines.extend(["", "Notes", "-----", *notes])
    report = "\n".join(lines) + "\n"

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")

    return report


def add_metadata_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--area-per-lipid", type=float, help="Area per lipid in nm^2/lipid, reported as metadata")
    parser.add_argument("--bilayer-thickness", type=float, help="Bilayer thickness in nm, reported as metadata")
    parser.add_argument("--order-parameter", type=float, help="Representative order parameter, reported as metadata")
    parser.add_argument("--bending-modulus", type=float, help="Bending modulus, reported as metadata")
    parser.add_argument("--bending-modulus-unit", default="kBT", help="Unit for --bending-modulus, default: kBT")
    parser.add_argument("--temperature", type=float, help="Temperature in K, reported as metadata")


def run_fit(args: argparse.Namespace) -> int:
    xvg = parse_xvg(Path(args.xvg))

    inferred_time_unit = unit_from_label(xvg.x_label, TIME_TO_SECONDS)
    inferred_msd_unit = unit_from_label(xvg.y_label, AREA_TO_M2)

    time_unit = args.time_unit or inferred_time_unit or "ps"
    msd_unit = args.msd_unit or inferred_msd_unit or "nm^2"

    points = select_points(
        xvg=xvg,
        msd_column=args.msd_column,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        auto_fit_fraction=args.auto_fit_fraction,
    )
    fit = linear_regression(points)
    result = calculate_diffusion(
        fit=fit,
        dimensions=args.dimensions,
        time_unit=time_unit,
        msd_unit=msd_unit,
    )
    report = build_report(
        result=result,
        xvg=xvg,
        args=args,
        inferred_time_unit=inferred_time_unit,
        inferred_msd_unit=inferred_msd_unit,
    )

    print(report, end="")
    if args.output:
        output_path = Path(args.output)
        print(f"\nWrote report: {output_path}")

    return 0


def run_gmx_command(args: argparse.Namespace) -> int:
    command = [
        "gmx",
        "msd",
        "-f",
        args.trajectory,
        "-s",
        args.structure,
        "-o",
        args.output,
        "-lateral",
        args.lateral_axis,
    ]
    if args.index:
        command.extend(["-n", args.index])
    if args.begin is not None:
        command.extend(["-b", str(args.begin)])
    if args.end is not None:
        command.extend(["-e", str(args.end)])

    print("Run this command, then select the lipid/molecule group when GROMACS prompts:")
    print(" ".join(shlex.quote(part) for part in command))
    print()
    print("Then fit the output with:")
    print(
        "python "
        + shlex.quote(Path(__file__).name)
        + " fit "
        + shlex.quote(args.output)
        + " --dimensions 2 --fit-start START --fit-end END"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate lipid-bilayer diffusion coefficients from GROMACS gmx msd .xvg output."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit a gmx msd .xvg file and calculate D")
    fit_parser.add_argument("xvg", help="GROMACS gmx msd output file, usually .xvg")
    fit_parser.add_argument(
        "--dimensions",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="Diffusion dimensionality. Use 2 for lateral bilayer diffusion. Default: 2",
    )
    fit_parser.add_argument(
        "--msd-column",
        type=int,
        default=1,
        help="1-based MSD data column after the time column. Default: 1",
    )
    fit_parser.add_argument("--fit-start", type=float, help="Start time for the linear fit, in the XVG time unit")
    fit_parser.add_argument("--fit-end", type=float, help="End time for the linear fit, in the XVG time unit")
    fit_parser.add_argument(
        "--auto-fit-fraction",
        type=float,
        default=0.5,
        help="If no fit window is supplied, fit this final fraction of points. Default: 0.5",
    )
    fit_parser.add_argument(
        "--time-unit",
        choices=sorted(TIME_TO_SECONDS),
        help="Time unit. If omitted, the script tries to infer it from the XVG x-axis label; default fallback: ps",
    )
    fit_parser.add_argument(
        "--msd-unit",
        choices=sorted(AREA_TO_M2),
        help="MSD unit. If omitted, the script tries to infer it from the XVG y-axis label; default fallback: nm^2",
    )
    fit_parser.add_argument("--output", help="Optional path to write a text report")
    add_metadata_arguments(fit_parser)
    fit_parser.set_defaults(func=run_fit)

    gmx_parser = subparsers.add_parser("gmx-command", help="Print a gmx msd command for lateral bilayer MSD")
    gmx_parser.add_argument("--trajectory", "-f", required=True, help="Trajectory file, e.g. traj.xtc")
    gmx_parser.add_argument("--structure", "-s", required=True, help="Structure/run input file, e.g. topol.tpr")
    gmx_parser.add_argument("--index", "-n", help="Optional index file")
    gmx_parser.add_argument("--output", "-o", default="msd_lateral.xvg", help="Output XVG file. Default: msd_lateral.xvg")
    gmx_parser.add_argument(
        "--lateral-axis",
        default="z",
        choices=("x", "y", "z"),
        help="Bilayer normal axis passed to gmx msd -lateral. Default: z",
    )
    gmx_parser.add_argument("--begin", "-b", type=float, help="Optional GROMACS begin time")
    gmx_parser.add_argument("--end", "-e", type=float, help="Optional GROMACS end time")
    gmx_parser.set_defaults(func=run_gmx_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
