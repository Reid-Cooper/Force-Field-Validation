#!/usr/bin/env python3

import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt

#Order parameters from your simulation, generated using FAIRMD tools.
SIM_FILE = "POPCOrderParameters.json"

#Change this to the directory of the experimental order parameters file, specificied at the top of the OrderParametersQuality file.
EXP_FILE = "/Users/reidxcooper/FAIRMD/BilayerData/experiments/OrderParameters/10.1021/acs.jpcb.4c04719/4/POPC_OrderParameters.json"  

#Output plot file.
OUT_FILE = "order_parameters_vs_experiment.png"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def value_and_error_from_entry(entry):
    """
    Simulation format:
        [[mean, std, stderr]]

    Experiment format:
        [[value, uncertainty]]

    This returns:
        value = entry[0][0]
        error = entry[0][1] if present
    """
    if isinstance(entry, list) and entry and isinstance(entry[0], list):
        value = entry[0][0]
        error = entry[0][1] if len(entry[0]) > 1 else None
        return value, error

    return None, None


def parse_key(key):
    """
    Classify FAIRMD POPC order parameter keys into:
      - headgroup: gamma, alpha, beta, g3, g2, g1
      - sn1 tail
      - sn2 tail
    """

    # Headgroup / glycerol
    if re.search(r"M_G1_M\s+M_G1H", key):
        return "head", "g1"

    if re.search(r"M_G2_M\s+M_G2H", key):
        return "head", "g2"

    if re.search(r"M_G3_M\s+M_G3H", key):
        return "head", "g3"

    # Choline / phosphate-side headgroup
    if re.search(r"M_G3C4_M\s+M_G3C4H", key):
        return "head", "β"

    if re.search(r"M_G3C5_M\s+M_G3C5H", key):
        return "head", "α"

    if re.search(r"M_G3N6C[123]_M\s+M_G3N6C[123]H", key):
        return "head", "γ"

    # sn-1 tail
    m = re.search(r"M_G1C(\d+)_M", key)
    if m:
        carbon = int(m.group(1))
        if 3 <= carbon <= 17:
            return "sn1", carbon

    # sn-2 tail
    m = re.search(r"M_G2C(\d+)_M", key)
    if m:
        carbon = int(m.group(1))
        if 3 <= carbon <= 19:
            return "sn2", carbon

    return None, None


def collect_order_parameters(data, use_abs_for_tails=True):
    values = {
        "head": defaultdict(list),
        "sn1": defaultdict(list),
        "sn2": defaultdict(list),
    }

    errors = {
        "head": defaultdict(list),
        "sn1": defaultdict(list),
        "sn2": defaultdict(list),
    }

    for key, entry in data.items():
        group, label = parse_key(key)
        value, error = value_and_error_from_entry(entry)

        if group is None or value is None:
            continue

        if group in ["sn1", "sn2"] and use_abs_for_tails:
            value = abs(value)

        values[group][label].append(value)

        if error is not None:
            errors[group][label].append(abs(error))

    averaged = {"head": {}, "sn1": {}, "sn2": {}}
    averaged_errors = {"head": {}, "sn1": {}, "sn2": {}}

    for group in values:
        for label, vals in values[group].items():
            averaged[group][label] = sum(vals) / len(vals)

            if errors[group][label]:
                averaged_errors[group][label] = (
                    sum(errors[group][label]) / len(errors[group][label])
                )

    return averaged, averaged_errors


def plot_headgroup(ax, sim_ops, exp_ops, exp_errs):
    head_order = ["γ", "α", "β", "g3", "g2", "g1"]

    exp_labels = [x for x in head_order if exp_ops["head"].get(x) is not None]
    exp_values = [exp_ops["head"][x] for x in exp_labels]
    exp_errors = [exp_errs["head"].get(x, 0.0) for x in exp_labels]

    sim_labels = [x for x in head_order if sim_ops["head"].get(x) is not None]
    sim_values = [sim_ops["head"][x] for x in sim_labels]

    if exp_labels:
        ax.errorbar(
            exp_labels,
            exp_values,
            yerr=exp_errors,
            fmt="o",
            color="black",
            capsize=3,
            label="Experiment",
        )

    if sim_labels:
        ax.scatter(
            sim_labels,
            sim_values,
            marker="s",
            color="#ff7f0e",
            label="OpenFF Sage 2.3.0",
        )

    ax.set_title("Head Group")
    ax.set_xlabel("Carbon")
    ax.set_ylabel(r"$S_{CH}$")
    ax.legend()


def plot_tail(ax, sim_ops, exp_ops, exp_errs, group, title):
    carbons = sorted(set(sim_ops[group]) | set(exp_ops[group]))

    exp_carbons = [c for c in carbons if c in exp_ops[group]]
    exp_values = [exp_ops[group][c] for c in exp_carbons]
    exp_errors = [exp_errs[group].get(c, 0.0) for c in exp_carbons]

    sim_carbons = [c for c in carbons if c in sim_ops[group]]
    sim_values = [sim_ops[group][c] for c in sim_carbons]

    if exp_carbons:
        ax.errorbar(
            exp_carbons,
            exp_values,
            yerr=exp_errors,
            fmt="o-",
            color="black",
            capsize=3,
            label="Experiment",
        )

    if sim_carbons:
        ax.plot(
            sim_carbons,
            sim_values,
            marker="s",
            linestyle="-",
            color="#ff7f0e",
            label="OpenFF Sage 2.3.0",
        )

    ax.set_title(title)
    ax.set_xlabel("Carbon Number")
    ax.set_ylabel(r"$|S_{CH}|$")
    ax.legend()


def print_debug(sim_ops, exp_ops):
    print("\nSimulation headgroup values:")
    for label in ["γ", "α", "β", "g3", "g2", "g1"]:
        print(label, sim_ops["head"].get(label))

    print("\nExperiment headgroup values:")
    for label in ["γ", "α", "β", "g3", "g2", "g1"]:
        print(label, exp_ops["head"].get(label))

    print("\nSimulation sn1 carbons:", sorted(sim_ops["sn1"]))
    print("Experiment sn1 carbons:", sorted(exp_ops["sn1"]))

    print("\nSimulation sn2 carbons:", sorted(sim_ops["sn2"]))
    print("Experiment sn2 carbons:", sorted(exp_ops["sn2"]))


def main():
    sim_data = load_json(SIM_FILE)
    exp_data = load_json(EXP_FILE)

    sim_ops, sim_errs = collect_order_parameters(sim_data)
    exp_ops, exp_errs = collect_order_parameters(exp_data)

    print_debug(sim_ops, exp_ops)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_headgroup(axes[0], sim_ops, exp_ops, exp_errs)
    plot_tail(axes[1], sim_ops, exp_ops, exp_errs, "sn1", "sn-1")
    plot_tail(axes[2], sim_ops, exp_ops, exp_errs, "sn2", "sn-2")

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300)
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
