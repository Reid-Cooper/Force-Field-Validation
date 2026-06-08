from fairmd.lipids.api import get_ApL_data
import matplotlib.pyplot as plt

system = {
    "path": " " #paste the direct path to system here
}

apl = get_ApL_data(system)

plt.figure(figsize=(8,5))
plt.plot(apl[:,0], apl[:,1])
plt.xlabel("Time (ps)")
plt.ylabel("Area per Lipid (Å²)")
plt.title("APL vs Time")

plt.savefig("APL_vs_time.png", dpi=300, bbox_inches="tight")
plt.close()