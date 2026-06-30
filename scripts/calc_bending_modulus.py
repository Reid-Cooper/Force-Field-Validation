import json
import re
import statistics
import sys
from pathlib import Path


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def runtime_ns_from_apl(data):
    times_ps = []
    for row in data:
        if not isinstance(row, list) or len(row) < 2 or not is_number(row[0]):
            return None
        times_ps.append(float(row[0]))

    if len(times_ps) < 2:
        return None
    return (max(times_ps) - min(times_ps)) / 1000


def runtime_ns_from_readme(system_dir):
    readme_path = system_dir / "README.yaml"
    if not readme_path.exists():
        return None

    match = re.search(
        r"^\s*TRJLENGTH\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        readme_path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        return None
    return float(match.group(1)) / 1000


def format_runtime_ns(runtime_ns):
    if runtime_ns is None:
        return "not detected"
    if abs(runtime_ns - round(runtime_ns)) < 1e-9:
        return f"{round(runtime_ns):.0f} ns"
    return f"{runtime_ns:.3f}".rstrip("0").rstrip(".") + " ns"


# ensure correct number of arguments
if len(sys.argv) < 4:
    print("Error: Missing arguments.")
    print("Usage: python CalculateBendingModulus.py <apl_input_file.json> <lipids_per_leaflet> <bilayer_thickness> [fairmd_system_dir]")
    sys.exit(1)

# set up variables and constants
kB = 1.380649e-23 # Boltzmann constant in J/K
T = 303 # Temperature in K
N = int(sys.argv[2]) # lipids per leaflet
bilayerThickness = float(sys.argv[3]) # bilayer thickness, h, in nm
A2toM2 = 1e-20 # conversion factor from Å^2 to m^2
apl_path = Path(sys.argv[1]).expanduser().resolve()
system_dir = Path(sys.argv[4]).expanduser().resolve() if len(sys.argv) > 4 else apl_path.parent

# read in data from json file
with apl_path.open() as f:
    data = json.load(f)
areas = [apl * N * A2toM2 for _, apl in data]
md_runtime_ns = runtime_ns_from_apl(data)
if md_runtime_ns is None:
    md_runtime_ns = runtime_ns_from_readme(system_dir)
md_runtime = format_runtime_ns(md_runtime_ns)

# calculate mean area, variance of area, compressibility modulus, and bending modulus
mean_A = statistics.mean(areas)
var_A = statistics.pvariance(areas)
compressibilityModulus = (kB * T * mean_A) / var_A #KA, stiffness of the membrane (N/m), (KB * T * <A>) / Var(A)
bendingModulus = (compressibilityModulus * (bilayerThickness * 1e-9)**2) / 48 #KB also known as Kappa or bending modulus (J), (KA * h^2) / 48

# print and save results
results = f"""
-----------------------------------------
Bilayer Thickness = {bilayerThickness} nm
Lipids Per Leaflet = {N}
Force Field Type = Sage 2.3.0
MD Production Runtime = {md_runtime}
-----------------------------------------
<A> = {mean_A:.3e} m²
Var(A) = {var_A:.3e} m⁴
-----------------------------------------
K_A = {compressibilityModulus:.3f} N/m
K_A = {compressibilityModulus*1000:.1f} mN/m
-----------------------------------------
K_C = {bendingModulus:.3e} J
K_C = {bendingModulus/(kB*T):.2f} kBT
-----------------------------------------
"""

print(results)

# saves results to an output file
# with open("metric_results/bending_modulus_Sage230_POPC256_bending_modulus.txt", "w") as f:
#    f.write(results)
