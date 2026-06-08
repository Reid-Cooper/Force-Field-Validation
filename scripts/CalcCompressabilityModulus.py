import json
import statistics
import sys

if len(sys.argv) < 3:
    print("Error: Missing arguments.")
    print("Usage: python CalculateCompressability.py <apl_input_file.json> <lipids_per_leaflet>")
    sys.exit(1)

kB = 1.380649e-23  # Boltzmann constant in J/K
T = 300  # Temperature in K
N = int(sys.argv[2])  # lipids per leaflet
A2toM2 = 1e-20 # conversion factor from Å^2 to m^2

with open(sys.argv[1]) as f:
    data = json.load(f)

areas = [apl * N * A2toM2 for _, apl in data]

mean_A = statistics.mean(areas)
var_A = statistics.pvariance(areas)

print(f"<A> = {mean_A:.3e} m²")
print(f"Var(A) = {var_A:.3e} m⁴")

compressibilityModulus = (kB * T * mean_A) / var_A #KA, stiffness of the membrane, in N/m

print(f"K_A = {compressibilityModulus:.3f} N/m")
print(f"K_A = {compressibilityModulus*1000:.1f} mN/m")