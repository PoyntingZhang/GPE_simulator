"""CPU analysis of a power-law study: python -m gpe.io.analyze_power_law FOLDER."""

import argparse
import csv
from html import escape
import json
from pathlib import Path

import numpy as np


def power_fit(k, nk, lower=0.1, upper=2, minimum_bins=6):
    mask = (k >= lower) & (k <= upper) & (k > 0) & (nk > 0) & np.isfinite(nk)
    x, y = np.log(k[mask]), np.log(nk[mask])
    result = dict(alpha=np.nan, A=np.nan, log_rss=np.nan, log_rms=np.nan,
                  n_bins=len(x), k_min=np.nan, k_max=np.nan, log_curvature=np.nan)
    if len(x) < minimum_bins:
        return result
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope*x)
    result.update(alpha=-slope, A=np.exp(intercept), log_rss=float(residual@residual),
                  log_rms=float(np.sqrt(np.mean(residual**2))), k_min=float(np.exp(x[0])),
                  k_max=float(np.exp(x[-1])), log_curvature=float(np.polyfit(x, y, 2)[0]))
    return result


def local_slopes(k, nk, width=7):
    """Sliding log-space fits; return bounds to expose the smoothing scale."""
    output = []
    for start in range(len(k)-width+1):
        stop = start+width
        fit = power_fit(k[start:stop], nk[start:stop], k[start], k[stop-1], width)
        output.append((np.sqrt(k[start]*k[stop-1]), fit["alpha"], k[start], k[stop-1]))
    return np.asarray(output).reshape(-1, 4)


def read_spectra(folder):
    spectra = []
    for path in sorted(folder.glob("nk_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            keep = (data["k"] > 0) & (data["k"] <= 5) & (data["counts"] > 0) & data["complete_shells"]
            spectra.append(dict(time=float(data["time"]), k=data["k"][keep], nk=data["nk"][keep],
                                norm=float(data["Nk"].sum()),
                                low_k_population=float(data["Nk"][data["k"] < 0.3].sum()),
                                kp=float(data["k"][np.argmax(data["Nk"]) ])))
    return sorted(spectra, key=lambda sample: sample["time"])


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_case(folder, case, output, plt):
    from .plot_results import plot_stage

    name = case["name"]
    source = folder/"monitor"/name
    samples = read_spectra(source)
    if not samples:
        return None
    output.mkdir(parents=True, exist_ok=True)
    settings = case["config"]["stages"][-1]
    beta = 8*np.pi*settings["cloud"]["N_atom"]*settings["cloud"]["a"]*5.29177210544e-5
    volume = np.pi*settings["box"]["R"]**2*settings["box"]["L"]
    k0 = samples[0]["kp"]
    rows = []
    for sample in samples:
        fit = power_fit(sample["k"], sample["nk"])
        kxi = np.sqrt(max(0, beta*sample["norm"]/volume))
        theory = power_fit(sample["k"], sample["nk"], max(sample["kp"], kxi), k0)
        rows.append(dict(time=sample["time"], **fit, norm_spectrum=sample["norm"],
                         low_k_population=sample["low_k_population"], kp=sample["kp"],
                         kxi_nominal=kxi, restricted_alpha=theory["alpha"],
                         restricted_n_bins=theory["n_bins"]))
    save_csv(output/"fits.csv", rows)
    t = np.array([r["time"] for r in rows])
    alpha = np.array([r["alpha"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    ax = axes[0, 0]
    for index in np.unique(np.linspace(0, len(samples)-1, min(50, len(samples))).astype(int)):
        sample = samples[index]
        colour = plt.cm.viridis(index/max(1, len(samples)-1))
        ax.loglog(sample["k"], sample["nk"], color=colour, alpha=0.45, linewidth=0.8)
    for sample, colour, label in ((samples[0], "tab:blue", "First"), (samples[-1], "black", "Last")):
        ax.loglog(sample["k"], sample["nk"], color=colour, label=f"{label}: t={sample['time']:g}")
    last = rows[-1]
    if np.isfinite(last["alpha"]):
        k = np.geomspace(last["k_min"], last["k_max"], 100)
        ax.loglog(k, last["A"]*k**(-last["alpha"]), "r--", label=f"Fit: alpha={last['alpha']:.3f}")
    ax.set(xlabel="k", ylabel="n(k)", xlim=(samples[0]["k"].min(), 5))
    ax.legend(fontsize=8)
    axes[0, 1].plot(t, alpha, label="0.1 <= k <= 2")
    axes[0, 1].plot(t, [r["restricted_alpha"] for r in rows], ".", label="max(kp, kxi) <= k <= initial kp")
    axes[0, 1].set(xlabel="Time since quench", ylabel="Fitted alpha")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(t, [r["log_rms"] for r in rows])
    axes[1, 0].set(xlabel="Time since quench", ylabel="RMS natural-log residual", title="Unweighted; not reduced chi-square")
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 1].plot(t, [r["kp"] for r in rows], label="Population peak kp")
    axes[1, 1].plot(t, [r["kxi_nominal"] for r in rows], "--", label="Nominal kxi")
    axes[1, 1].set(xlabel="Time since quench", ylabel="k")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2, which="both")
    fig.suptitle(f"{name} — {case['status']}")
    fig.savefig(output/"history.png", dpi=130)
    plt.close(fig)

    # Window sensitivity at several times, not just the final endpoint.
    lows = [0.1, 0.18, 0.27, 0.36, 0.45, 0.54, 0.72]
    highs = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    selected = np.unique(np.linspace(0, len(samples)-1, min(4, len(samples))).astype(int))
    windows, local = [], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for index in selected:
        sample = samples[index]
        for low in lows:
            for high in highs:
                windows.append(dict(time=sample["time"], lower=low, upper=high,
                                    **power_fit(sample["k"], sample["nk"], low, high)))
        slope = local_slopes(sample["k"], sample["nk"])
        for center, value, low, high in slope:
            local.append(dict(time=sample["time"], k=center, alpha=value, k_min=low, k_max=high))
        if len(slope):
            axes[0].semilogx(slope[:, 0], slope[:, 1], label=f"t={sample['time']:g}")
    axes[0].set(xlabel="Geometric window centre k", ylabel="Local alpha (7-bin fits)", xlim=(0.1, 5))
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    final = samples[-1]
    matrix = np.array([[power_fit(final["k"], final["nk"], low, high)["alpha"]
                        for high in highs] for low in lows])
    mesh = axes[1].imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set(xticks=range(len(highs)), xticklabels=highs, yticks=range(len(lows)), yticklabels=lows,
                xlabel="Upper fitting limit", ylabel="Lower fitting limit",
                title=f"Final alpha; blank = fewer than 6 bins")
    fig.colorbar(mesh, ax=axes[1], label="alpha")
    fig.savefig(output/"fit_sensitivity.png", dpi=130)
    plt.close(fig)
    save_csv(output/"fit_windows.csv", windows)
    save_csv(output/"local_slopes.csv", local)

    scalar_path = source/"scalars.csv"
    if scalar_path.exists():
        scalar = np.genfromtxt(scalar_path, delimiter=",", names=True, ndmin=1)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        times = scalar["time"]
        axes[0].plot(times, scalar["norm"], ".-")
        axes[0].ticklabel_format(axis="y", useOffset=False)
        axes[0].set(ylabel="Norm", xlabel="Time since quench")
        for key in ("E_kin", "E_potential", "E_interaction", "E_total"):
            axes[1].plot(times, scalar[key], label=key)
        axes[1].set(xlabel="Time since quench", ylabel="Integrated energy")
        axes[1].legend(fontsize=8)
        for key in ("kx2_fraction", "ky2_fraction", "kz2_fraction"):
            axes[2].plot(times, scalar[key], label=key)
        axes[2].axhline(1/3, color="grey", linestyle="--")
        axes[2].set(xlabel="Time since quench", ylabel="Fraction of kinetic energy by axis")
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.suptitle(name + (" — reference energies sampled only at endpoints" if name == "reference_a20" else ""))
        fig.savefig(output/"energies_anisotropy.png", dpi=130)
        plt.close(fig)
    # Existing plotter provides labelled density/phase/TOF panels at +/-5.
    if any(source.glob("density_xy_*.npz")) or not (output/"diagnostics.png").exists():
        plot_stage(source, output, np, plt, tof_limit=5, k_max=5)
    first_images = [("density_xy", "x", "y"), ("TOF_xy", "kx", "ky"), ("TOF_yz", "ky", "kz")]
    if all((source/f"{prefix}_00000000.npz").exists() for prefix, _, _ in first_images):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, (prefix, horizontal, vertical) in zip(axes, first_images):
            with np.load(source/f"{prefix}_00000000.npz", allow_pickle=False) as data:
                mesh = ax.pcolormesh(data[horizontal], data[vertical], data["values"].T,
                                     shading="auto", cmap="magma")
                ax.set(title=f"{prefix}; t={float(data['time']):g}", xlabel=horizontal, ylabel=vertical)
                if prefix.startswith("TOF"):
                    ax.set(xlim=(-5, 5), ylim=(-5, 5))
                ax.set_aspect("equal")
                fig.colorbar(mesh, ax=ax, shrink=0.8)
        fig.suptitle(name + " — starting state; independent image colour scales")
        fig.savefig(output/"initial_state.png", dpi=130)
        plt.close(fig)
    return dict(case=case, rows=rows, samples=samples)


def analyze(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder).resolve()
    manifest = json.loads((folder/"manifest.json").read_text())
    output = folder/"analysis"
    output.mkdir(exist_ok=True)
    results = {}
    for case in manifest["cases"]:
        result = analyze_case(folder, case, output/case["name"], plt)
        if result:
            results[case["name"]] = result
    comparisons = {
        "timesteps": ["reference_a20", "dt_half", "dt_quarter"],
        "disorder_controls": ["reference_a20", "a0_control", "clean_a20", "clean_a0"],
        "interactions": ["reference_a20", "a10", "a40"],
        "realizations": ["reference_a20", "seed_43", "seed_44", "seed_45"],
        "long_time": ["reference_a20", "long_a20"],
        "resolution_clean": ["clean_a20", "clean_a20_512"],
    }
    pair_metrics = []
    for group, names in comparisons.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        plotted = False
        for name in names:
            if name not in results:
                continue
            plotted = True
            rows = results[name]["rows"]
            t = [row["time"] for row in rows]
            for ax, key in zip(axes, ("alpha", "kp", "low_k_population")):
                ax.plot(t, [row[key] for row in rows], label=name)
        for ax, label in zip(axes, ("Alpha: 0.1 <= k <= 2", "Population peak k", "Integrated population k < 0.3")):
            ax.set(xlabel="Time since quench", ylabel=label)
            ax.grid(alpha=0.2)
            if plotted:
                ax.legend(fontsize=7)
        fig.suptitle(group.replace("_", " "))
        fig.savefig(output/f"{group}.png", dpi=130)
        plt.close(fig)

        if names[0] in results:
            reference = {round(s["time"], 8): s for s in results[names[0]]["samples"]}
            for name in names[1:]:
                if name not in results:
                    continue
                for sample in results[name]["samples"]:
                    ref = reference.get(round(sample["time"], 8))
                    if ref is None or sample["k"].shape != ref["k"].shape or not np.allclose(sample["k"], ref["k"]):
                        continue
                    mask = ((ref["k"] >= 0.1) & (ref["k"] <= 2) & (ref["nk"] > 0)
                            & (sample["nk"] > 0))
                    if mask.sum() < 6:
                        continue
                    ref_fit = power_fit(ref["k"], ref["nk"])
                    fit = power_fit(sample["k"], sample["nk"])
                    pair_metrics.append(dict(group=group, reference=names[0], case=name,
                        time=sample["time"], delta_alpha=fit["alpha"]-ref_fit["alpha"],
                        spectrum_log_rms=float(np.sqrt(np.mean(np.log(sample["nk"][mask]/ref["nk"][mask])**2))),
                        delta_norm=sample["norm"]-ref["norm"]))
    save_csv(output/"paired_comparisons.csv", pair_metrics)

    summary = []
    for name, result in results.items():
        case, rows = result["case"], result["rows"]
        stage = case["config"]["stages"][-1]
        last = rows[-1]
        summary.append(dict(case=name, status=case["status"], grid=case["config"]["simulation"]["N_grids"],
                            a=stage["cloud"]["a"], dt=stage["evolution"]["dt"],
                            seed=stage["noise"]["seed"], noise=stage["noise"]["enabled"],
                            first_time=rows[0]["time"], last_time=last["time"],
                            final_alpha=last["alpha"], final_log_rms=last["log_rms"],
                            final_fit_bins=last["n_bins"], initial_kp=rows[0]["kp"],
                            final_kp=last["kp"], wall_seconds=case.get("wall_seconds", 0)))
    save_csv(output/"summary.csv", summary)
    notes = [
        "POWER-LAW STUDY: DESCRIPTIVE ANALYSIS, NOT AUTOMATIC PROOF OF A CASCADE",
        "Fits use shell-averaged n(k), not k^2*n(k) or the discrete shell population Nk.",
        "The reference fit is 0.1 <= k <= 2. Actual bin bounds are recorded in fits.csv.",
        "Residuals are unweighted natural-log residuals, not statistically calibrated chi-square.",
        "Missing exponents mean fewer than six positive bins; six bins alone do not establish an inertial range.",
        "The restricted fit uses max(kp(t), nominal kxi) to the first saved kp; long_a20 starts at t=1000, so its restricted fit is not a new-quench test.",
        "Time samples from one trajectory are correlated. Independent preparation seeds provide the realization comparison.",
        "Compare initial energy and kp before interpreting seed-to-seed differences; their initial states need not have identical energy.",
        "Static disorder is switched off only at the start of the clean branches; this is a potential quench.",
        "The 512 comparison tests the clean dynamics only. Its physical box is unchanged, so it does not test finite-size effects.",
        "E_total includes kinetic, real potential and interaction energy. Absorber loss and quenches preclude a blanket conservation claim.",
        "Momentum-axis fractions test only diagonal second moments; equality does not prove full isotropy.",
        "Low-k accumulation alone does not isolate nonlinear flux from scattering by the trap or disorder.",
        "Full states are retained in the runtime data/ folder for further analysis; the review ZIP contains spectra and scalar data.",
    ]
    lines = notes + ["", "FINAL FITS (compare full histories, not just these endpoints):"]
    for row in summary:
        lines.append(f"{row['case']}: {row['status']}; t={row['last_time']:g}; "
                     f"alpha={row['final_alpha']:.5g}; log RMS={row['final_log_rms']:.5g}")
    (output/"summary.txt").write_text("\n".join(lines)+"\n")
    html = ["<!doctype html><meta charset='utf-8'><title>Power-law study</title>",
            "<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:0 1em}img{width:100%}li{margin:.5em 0}</style>",
            "<h1>Power-law study</h1>", "<p>Source run: "+escape(manifest["source_run"])+"</p>",
            "<p><a href='summary.csv'>Summary CSV</a> · <a href='summary.txt'>Definitions and limitations</a></p>",
            "<ul>"+"".join("<li>"+escape(note)+"</li>" for note in notes[1:])+"</ul>"]
    for group in comparisons:
        html.append(f"<h2>{escape(group.replace('_', ' '))}</h2><img src='{group}.png'>")
    for name in results:
        html.append(f"<h2>{escape(name)}</h2><p><a href='{name}/fits.csv'>All-time fits</a> · "
                    f"<a href='{name}/fit_windows.csv'>Fit windows</a> · <a href='{name}/local_slopes.csv'>Local slopes</a></p>")
        for picture in ("history", "fit_sensitivity", "energies_anisotropy", "initial_state", "diagnostics"):
            if (output/name/f"{picture}.png").exists():
                html.append(f"<img loading='lazy' src='{name}/{picture}.png'>")
    (output/"index.html").write_text("\n".join(html))
    print(f"Readable report: {output/'index.html'}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    args = parser.parse_args()
    analyze(args.folder)
