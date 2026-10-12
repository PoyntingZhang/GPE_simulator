"""Controlled five-hour study: python -m gpe.power_law_study --help."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

from config.yaml_parser import dump_yaml, load_yaml


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024**2), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def build_cases(template, initial, final):
    """Every control loads the same pre-interaction state, independently."""
    cases = []

    def branch(name, a, duration, dt=0.01, noise=True, source=initial, offset=0, grid=256):
        config = copy.deepcopy(template)
        item = copy.deepcopy(template["stages"][-1])
        item["name"] = name
        item["cloud"]["a"] = a
        item["evolution"].update(type="real", dt=dt, steps=round(duration/dt))
        item["initial_state"].update(type="load", path=str(source))
        item["noise"]["enabled"] = noise
        config["simulation"]["N_grids"] = grid
        config["stages"] = [item]
        cases.append(dict(name=name, config=config, time_offset=offset,
                          requested_duration=duration, status="pending"))

    branch("a0_control", 0, 2000)
    branch("dt_half", 20, 1000, dt=0.005)
    branch("dt_quarter", 20, 1000, dt=0.0025)
    branch("long_a20", 20, 2000, source=final, offset=1000)
    branch("clean_a0", 0, 2000, noise=False)
    branch("clean_a20", 20, 2000, noise=False)
    branch("a10", 10, 4000)
    branch("a40", 40, 1000)
    for seed in (43, 44, 45):
        config = copy.deepcopy(template)
        for stage in config["stages"]:
            stage["noise"]["seed"] = seed
        cases.append(dict(name=f"seed_{seed}", config=config, time_offset=0,
                          requested_duration=1000, status="pending"))
    branch("clean_a20_512", 20, 2000, noise=False, grid=512)
    return cases


def budget_steps(requested, dt, seconds_left, seconds_per_step, reserve=600):
    """Keep at most the requested duration; stop on a ten-time-unit boundary."""
    block = max(1, round(10/dt))
    affordable = max(0, int((seconds_left-reserve)/seconds_per_step))
    return min(requested, affordable//block*block)


def publish(folder, share_dir=None):
    """A small, inspectable archive; full 3-D states stay in the runtime folder."""
    archive = folder / "review.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path == archive:
                continue
            rel = path.relative_to(folder)
            include = (rel.parts[0] in {"configs", "analysis", "source"}
                       or rel.as_posix() in {"manifest.json", "run.log", "README.txt"}
                       or (rel.parts[0] == "monitor" and
                           (path.name.startswith("nk_") or path.suffix == ".csv")))
            if include:
                bundle.write(path, rel.as_posix())
    if share_dir is not None:
        target = Path(share_dir).expanduser() / folder.name
        target.mkdir(parents=True, exist_ok=True)
        # Atomic replacement avoids exposing a half-copied ZIP to Drive sync.
        temporary = target / "review.zip.partial"
        shutil.copy2(archive, temporary)
        temporary.replace(target / "review.zip")
        shutil.copy2(folder / "manifest.json", target / "manifest.json")
    return archive


def prepare_config(case, folder):
    config = copy.deepcopy(case["config"])
    config["simulation"].update(data_root=str(folder/"data"), monitor_root=str(folder/"monitor"))
    for stage in config["stages"]:
        stage["monitor"] = dict.fromkeys(stage["monitor"], 0)
        stage["save"].update(directory="", scalar_interval=0, nk_interval=0,
                             psi_interval=0, save_initial=False, save_final=False)
    path = folder / "configs" / f"{case['name']}.yaml"
    path.write_text(dump_yaml(config))
    return path


def make_stages(case, folder):
    from .stage_factory import make_stages as stages_from_yaml
    stages = stages_from_yaml(prepare_config(case, folder))
    for index, stage in enumerate(stages, 1):
        stage.data_dir = folder / "data" / case["name"] / f"stage_{index:02d}"
        stage.monitor_dir = folder / "monitor" / case["name"]
    return stages


def timed_steps(stage, cp, steps=100):
    """Warm up and time the actual propagator, then restore the untouched state."""
    original = stage.psi.copy()
    dynamic = stage.potential_params["dynamic"]["activate"]
    try:
        for i in range(5):
            stage.psi = stage.propagator(stage, i*stage.dt) if dynamic else stage.propagator(stage)
        cp.cuda.get_current_stream().synchronize()
        start = time.monotonic()
        for i in range(steps):
            stage.psi = stage.propagator(stage, i*stage.dt) if dynamic else stage.propagator(stage)
        cp.cuda.get_current_stream().synchronize()
        return (time.monotonic()-start)/steps
    finally:
        stage.psi = original


def evolve(stage, cp, deadline, offset, diagnostics, log):
    from .io.state_io import save_wavefunction
    from .io.study_diagnostics import record

    sample = max(1, round(10/stage.dt))
    snapshot = max(1, round(500/stage.dt))
    progress = max(1, round(100/stage.dt))
    dynamic = stage.potential_params["dynamic"]["activate"]
    if diagnostics:
        record(stage, 0, offset, images=True)
        save_wavefunction(stage.data_dir/"psi_00000000.npy", stage, 0)
    step = 0
    last_record = -1
    for step in range(1, stage.steps+1):
        stage.psi = (stage.propagator(stage, (step-1)*stage.dt)
                     if dynamic else stage.propagator(stage))
        if diagnostics and step % sample == 0:
            record(stage, step, offset, images=step % snapshot == 0)
            last_record = step
        if diagnostics and step % snapshot == 0:
            save_wavefunction(stage.data_dir/f"psi_{step:08d}.npy", stage, step)
        if step % progress == 0:
            log(f"  {stage.name}: t={offset+step*stage.dt:g}")
        if step % 100 == 0 and time.monotonic() >= deadline:
            break
    if diagnostics:
        if last_record != step:
            record(stage, step, offset, images=True)
        elif step % snapshot:
            record(stage, step, offset, images=True, scalar_row=False)
    save_wavefunction(stage.data_dir/f"psi_{step:08d}.npy", stage, step)
    cp.cuda.get_current_stream().synchronize()
    return step


def refine_snapshot(source, destination, size, cp):
    import numpy as np
    from types import SimpleNamespace
    from .core.grids import get_grids
    from .io.state_io import save_wavefunction
    from .io.study_diagnostics import refine_field

    metadata = json.loads(source.with_suffix(".json").read_text())
    original = cp.asarray(np.load(source, mmap_mode="r"), blocking=True)
    dx = metadata["simulation_range"]/len(original)
    before = float(cp.sum(cp.abs(original)**2))*dx**3
    refined = refine_field(original, size, xp=cp)
    after = float(cp.sum(cp.abs(refined)**2))*(metadata["simulation_range"]/size)**3
    error = float(cp.max(cp.abs(refined[::size//len(original), ::size//len(original), ::size//len(original)]-original)))
    if abs(after/before-1) > 1e-10 or error > 1e-10:
        raise RuntimeError("Fourier interpolation failed its norm or original-node check.")
    x, k = get_grids(size, metadata["simulation_range"])
    item = SimpleNamespace(psi=refined, N_grids=size, x_array=x, k_array=k,
                           simulation_range=metadata["simulation_range"],
                           N_atoms=metadata["N_atom"], type="real", dt=0.01,
                           config=metadata["stage"])
    save_wavefunction(destination, item, 0)
    return dict(norm_before=before, norm_after=after, maximum_original_node_error=error,
                method="Periodic complex Fourier interpolation; no renormalization")


def import_reference(template, initial, final, source_monitor, folder):
    """Bring the completed reference history into the same portable study."""
    from .io.state_io import load_wavefunction
    from .io.study_diagnostics import record

    case = dict(name="reference_a20", config=copy.deepcopy(template), time_offset=0,
                requested_duration=1000, completed_duration=1000, status="existing")
    case["config"]["stages"] = [copy.deepcopy(template["stages"][-1])]
    item = case["config"]["stages"][0]
    item["initial_state"].update(type="load", path=str(initial))
    stage = make_stages(case, folder)[0]
    stage.monitor_dir.mkdir(parents=True, exist_ok=True)
    for path in source_monitor.glob("nk_*.npz"):
        shutil.copy2(path, stage.monitor_dir/path.name)
    if (source_monitor/"scalars.csv").exists():
        shutil.copy2(source_monitor/"scalars.csv", stage.monitor_dir/"original_scalars.csv")
    stage.prepare()
    record(stage, 0, images=True)
    stage.psi = load_wavefunction(final, stage)
    record(stage, 100000, images=True)
    stage.release()
    return case


def validate_sources(template, initial, final):
    if (len(template["stages"]) != 7 or template["simulation"]["N_grids"] != 256
            or template["stages"][-1]["evolution"] != dict(type="real", steps=100000, dt=0.01)):
        raise ValueError("Use the approved seven-stage 256-cubed stronger-drive configuration.")
    for source, expected_a in ((initial, 0), (final, 20)):
        metadata = json.loads(source.with_suffix(".json").read_text())
        item = metadata["stage"]
        if (metadata["step"] != 100000 or metadata["N_grids"] != 256
                or metadata["time"] != 1000 or item["cloud"]["a"] != expected_a
                or item["evolution"]["type"] != "real"
                or metadata["simulation_range"] != template["simulation"]["simulation_range"]):
            raise ValueError(f"Unexpected source snapshot: {source}")
        for key in ("box", "absorber", "noise", "potential", "k_cut"):
            if item[key] != template["stages"][-1][key]:
                raise ValueError(f"Source {key} differs from the approved configuration: {source}")
        if item["cloud"]["N_atom"] != template["stages"][-1]["cloud"]["N_atom"]:
            raise ValueError("Reference atom numbers must match.")
    last = template["stages"][-1]
    if any(last["potential"][kind]["enabled"] for kind in ("static", "dynamic")):
        raise ValueError("This study's energy diagnostic assumes no extra potential expression.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment_drive_1p5.yaml")
    parser.add_argument("--source-run", required=True, help="timestamp of the completed seven-stage run")
    parser.add_argument("--hours", type=float, default=5.0, help="wall-time budget, including reporting")
    parser.add_argument("--output", default="study_results")
    parser.add_argument("--share-dir", help="copy review.zip here after every case, e.g. the mounted Drive repository/study_results")
    parser.add_argument("--plan-only", action="store_true", help="write all configurations without importing CuPy or running")
    args = parser.parse_args()
    if not 0 < args.hours < 100:
        parser.error("--hours must be positive and below 100")
    template = load_yaml(Path(args.config).read_text())
    def root(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT/path
    source_data = root(template["simulation"]["data_root"])/args.source_run
    source_monitor = root(template["simulation"]["monitor_root"])/args.source_run/"stage_07_relax_a20"
    initial = source_data/"stage_06_relax_continuation"/"psi_00100000.npy"
    final = source_data/"stage_07_relax_a20"/"psi_00100000.npy"
    if not args.plan_only:
        validate_sources(template, initial, final)
        if not (source_monitor/"nk_00100000.npz").exists():
            parser.error("The completed reference spectrum is missing.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = Path(args.output).expanduser().resolve()/stamp
    for name in ("configs", "source", "inputs"):
        (folder/name).mkdir(parents=True, exist_ok=True)
    cases = build_cases(template, initial, final)
    refined = folder/"inputs"/"initial_512.npy"
    cases[-1]["config"]["stages"][0]["initial_state"]["path"] = str(refined)
    for case in cases:
        prepare_config(case, folder)
    shutil.copy2(args.config, folder/"configs"/"preparation_template.yaml")
    for package in ("gpe", "config"):
        target = folder/"source"/package
        target.mkdir(parents=True)
        for path in (ROOT/package).rglob("*.py"):
            destination = target/path.relative_to(ROOT/package)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    manifest = dict(schema_version=1, created_utc=stamp, requested_hours=args.hours,
                    source_run=args.source_run, cases=cases,
                    acquisition=dict(spectra_and_scalars_time_interval=10,
                                     images_and_snapshots_time_interval=500,
                                     initial_and_final=True,
                                     note="The study runner collects diagnostics; ordinary YAML save intervals are disabled."),
                    definitions=dict(time="dimensionless time since interaction quench; long_a20 begins at 1000",
                        nk="shell-averaged density in three-dimensional k space",
                        Nk="population per discrete radial shell; sum equals wavefunction norm",
                        kxi_nominal="sqrt(beta*norm/(pi*R^2*L)); nominal cylindrical mean density",
                        energies="integrated energies; E_total excludes absorber; compare only within fixed Hamiltonian",
                        anisotropy="range of normalized lab-frame diagonal momentum second moments; zero alone does not prove isotropy",
                        log_residual="unweighted natural-log residual sum, not statistical chi-square"))
    (folder/"README.txt").write_text(
        "Power-law study review bundle\n"
        "Open analysis/index.html for figures and notes. manifest.json records settings and status.\n"
        "monitor/*/nk_*.npz contain k, nk, Nk, counts, edges, complete_shells, step and time.\n"
        "CSV files contain diagnostics and fit results; source/ preserves the exact Python code.\n"
        "Load NPZ files with numpy.load(path, allow_pickle=False). No pickle files are used.\n"
        "Full 3-D wavefunctions and raw image arrays remain in the Colab study folder, outside this ZIP.\n"
        "A copied review.zip after a partial run may not yet contain the final HTML report.\n")
    def persist():
        write_json(folder/"manifest.json", manifest)
    def log(message):
        print(message, flush=True)
        with (folder/"run.log").open("a") as file:
            file.write(message+"\n")
    persist()
    log(f"Study folder: {folder}")
    if args.plan_only:
        log("Plan only: 12 cases written. No simulation was run.")
        publish(folder, args.share_dir)
        return
    import cupy as cp
    started = time.monotonic()
    deadline = started + max(0, args.hours*3600-600)  # reserve reporting time
    manifest["sources"] = [fingerprint(initial), fingerprint(final)]
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties["name"]
    manifest["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
    manifest["cupy_version"] = cp.__version__
    manifest["status"] = "running"
    active_case = None
    try:
        reference = import_reference(template, initial, final, source_monitor, folder)
        manifest["cases"].insert(0, reference)
        persist()
        publish(folder, args.share_dir)
        for case in cases.copy():
            if case["status"] == "existing":
                continue
            if time.monotonic() >= deadline:
                case["status"] = "skipped_budget"
                continue
            log(f"Starting {case['name']}")
            active_case = case
            before = time.monotonic()
            if case["name"] == "clean_a20_512":
                manifest["interpolation"] = refine_snapshot(initial, refined, 512, cp)
            stages = make_stages(case, folder)
            previous = None
            case["status"] = "running"
            case["stages_completed"] = 0
            for index, stage in enumerate(stages):
                if time.monotonic() >= deadline:
                    case["status"] = "stopped_budget"
                    break
                stage.prepare(previous)
                previous = None
                last = index == len(stages)-1
                if last:
                    rate = timed_steps(stage, cp)
                    case["measured_ms_per_step"] = rate*1000
                    log(f"  warmed-up propagation: {rate*1000:.3f} ms/step")
                    if case["name"] == "clean_a20_512":
                        stage.steps = budget_steps(stage.steps, stage.dt, deadline-time.monotonic(), rate, reserve=120)
                        case["allocated_duration"] = stage.steps*stage.dt
                        if stage.steps == 0:
                            stage.release()
                            case["status"] = "skipped_budget"
                            break
                        stage.config["evolution"]["steps"] = stage.steps
                        case["config"]["stages"][-1]["evolution"]["steps"] = stage.steps
                        prepare_config(case, folder)
                persist()
                steps = evolve(stage, cp, deadline, case["time_offset"] if last else 0, last, log)
                previous = stage.psi
                stage.release()
                if last:
                    case["completed_duration"] = steps*stage.dt
                if steps < stage.steps:
                    case["status"] = "stopped_budget"
                    break
                case["stages_completed"] += 1
            else:
                case["status"] = ("complete_shortened_budget"
                                  if case.get("allocated_duration", case["requested_duration"]) < case["requested_duration"]
                                  else "complete")
            previous = None
            cp.get_default_memory_pool().free_all_blocks()
            cp.fft.config.get_plan_cache().clear()
            case["wall_seconds"] = time.monotonic()-before
            manifest["elapsed_seconds"] = time.monotonic()-started
            log(f"Finished {case['name']}: {case['status']} ({case['wall_seconds']/60:.1f} min)")
            persist()
            publish(folder, args.share_dir)
        manifest["status"] = "finished"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if active_case is not None and active_case["status"] == "running":
            active_case["status"] = manifest["status"]
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic()-started
        persist()
        # Preserve raw results before plotting, even if reporting later fails.
        publish(folder, args.share_dir)
        try:
            from .io.analyze_power_law import analyze
            analyze(folder)
        except Exception as exc:
            log(f"Report generation failed; raw results are safe: {exc}")
            manifest["report_error"] = str(exc)
        manifest["elapsed_seconds"] = time.monotonic()-started
        persist()
        archive = publish(folder, args.share_dir)
        log(f"Review bundle: {archive}")
        if args.share_dir:
            log(f"Drive copy: {Path(args.share_dir)/folder.name/'review.zip'}")


if __name__ == "__main__":
    main()
