import json
import statistics
import sys

# ensure correct number of arguments
if len(sys.argv) < 4:
    print("Error: Missing arguments.")
    print("Usage: python CalculateBendingModulus.py <apl_input_file.json> <lipids_per_leaflet> <bilayer_thickness>")
    sys.exit(1)

# set up variables and constants
kB = 1.380649e-23 # Boltzmann constant in J/K
T = 303 # Temperature in K
N = int(sys.argv[2]) # lipids per leaflet
bilayerThickness = float(sys.argv[3]) # bilayer thickness, h, in nm
A2toM2 = 1e-20 # conversion factor from Å^2 to m^2

# read in data from json file
with open(sys.argv[1]) as f:
    data = json.load(f)
areas = [apl * N * A2toM2 for _, apl in data]

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
MD Production Runtime = 160 ns
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
with open("metric_results/bending_modulus_Sage230_POPC256_bending_modulus.txt", "w") as f:
    f.write(results)
