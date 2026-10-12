"""Run an experiment: python -m gpe.main experiment.yaml"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the stages in a GPE experiment YAML file.")
    parser.add_argument("config", help="Path to the experiment YAML file")
    parser.add_argument("--check", action="store_true", help="Validate YAML settings without loading CuPy or running a simulation")
    args = parser.parse_args()
    from config_editor.configuration import ConfigError, load_yaml
    try:
        config = load_yaml(Path(args.config).expanduser().read_text())
    except (OSError, ConfigError) as exc:
        parser.error(str(exc))
    if args.check:
        print(f"Valid configuration: {len(config['stages'])} stages. No simulation was run.")
        return

    try:
        from .launch import launch_stage
        from .yaml_parser import parse_yaml
    except ModuleNotFoundError as exc:
        if exc.name == "cupy":
            parser.error("Simulation requires CuPy and a CUDA GPU. See gpe/run.md for setup; --check works without CuPy.")
        raise
    stages = parse_yaml(args.config)
    psi = None
    for current in stages:
        print(f"Starting {current.name} ({current.type}, {current.steps} steps)", flush=True)
        if current.initial_state["type"] != "continue":
            psi = None
        current.prepare(psi)
        psi = None  # Do not retain the first step's input throughout the whole stage.
        psi = launch_stage(current)
        current.release()
        print(f"Finished {current.name}", flush=True)


if __name__ == "__main__":
    main()
