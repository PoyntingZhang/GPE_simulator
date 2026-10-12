"""CPU tests for workflow plumbing; GPU kernels are deliberately not executed."""

import csv
import importlib
import json
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from config.yaml_parser import dump_yaml, new_config, new_stage
from gpe.core.expressions import cuda_expression


def norm(psi, x):
    return np.sum(np.abs(psi)**2) * (x[1] - x[0])**3


class ExpressionTests(unittest.TestCase):
    def test_cuda_arithmetic_preserves_powers_division_and_functions(self):
        self.assertEqual(cuda_expression("1/2*y**2"), "((1.0 / 2.0) * pow(y, 2.0))")
        self.assertEqual(cuda_expression("abs(-x) + cos(t)", True), "(fabs((-x)) + cos(t))")
        self.assertEqual(cuda_expression("pi"), repr(np.pi))
        with self.assertRaises(ValueError):
            cuda_expression("t*y")
        with self.assertRaises(ValueError):
            cuda_expression("x; return 0")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        # Scope the substitute to these tests, including when collected with GPU tests.
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        for name in list(sys.modules):
            if name == "gpe" or name.startswith("gpe."):
                del sys.modules[name]
        cupy = ModuleType("cupy")
        cupy.__getattr__ = lambda name: getattr(np, name)
        cupy.asnumpy = np.asarray
        cupy.asarray = lambda value, blocking=False, **kwargs: np.asarray(value, **kwargs)
        cupy.fft = SimpleNamespace(fftn=Mock(wraps=np.fft.fftn), fftfreq=np.fft.fftfreq,
                                  fftshift=np.fft.fftshift, ifftn=np.fft.ifftn)
        def kernel(*args, **kwargs):
            return Mock(side_effect=AssertionError("A GPU kernel cannot execute in this CPU test."))
        cupy.RawKernel = cupy.ReductionKernel = cupy.ElementwiseKernel = kernel
        sys.modules["cupy"] = cupy
        self.cp = cupy
        self.parser = importlib.import_module("gpe.stage_factory")
        self.io = importlib.import_module("gpe.io.state_io")
        self.save = importlib.import_module("gpe.io.save")
        self.launch = importlib.import_module("gpe.launch")
        self.stage_module = importlib.import_module("gpe.stage")

    def config(self):
        config = new_config()
        config["simulation"].update(N_grids=4, simulation_range=8,
                                    data_root=str(self.root / "data"),
                                    monitor_root=str(self.root / "monitor"))
        first = config["stages"][0]
        first["evolution"].update(type="real", steps=6, dt=0.125)
        first["box"]["enabled"] = False
        first["monitor"] = dict.fromkeys(first["monitor"], 0)
        first["save"].update(scalar_interval=0, save_final=False)
        return config

    def parse(self, config):
        path = self.root / "experiment.yaml"
        path.write_text(dump_yaml(config))
        return self.parser.make_stages(path)

    def ready_stage(self):
        item = self.parse(self.config())[0]
        item.psi = np.full((4, 4, 4), 0.1 + 0.2j, dtype=np.complex128)
        return item

    def test_parser_shares_axes_and_defers_large_allocations(self):
        config = self.config()
        first = config["stages"][0]
        first["initial_state"].update(type="load", path="seed.npy")
        first["potential"]["static"].update(enabled=True, expression="1/2*y**2")
        second = new_stage()
        second["potential"]["dynamic"].update(enabled=True, expression="cos(t)*x")
        second["k_cut"].update(enabled=True, value=3.5)
        config["stages"].append(second)
        stages = self.parse(config)
        self.assertEqual(len(stages), 2)
        self.assertIs(stages[0].x_array, stages[1].x_array)
        self.assertIs(stages[0].k_array, stages[1].k_array)
        for item in stages:
            self.assertIsNone(item.psi)
            self.assertIsNone(item.staticV)
            self.assertIsNone(item.propagator)
            self.assertFalse(item.data_dir.exists())
        self.assertEqual(stages[0].initial_state["path"], str(self.root / "seed.npy"))
        self.assertEqual(stages[0].potential_params["static"]["expression"],
                         "((1.0 / 2.0) * pow(y, 2.0))")
        self.assertEqual(stages[1].k_cut_params, {"activate": True, "k_cut": 3.5})
        self.assertEqual(stages[0].data_dir.parent.name, stages[1].monitor_dir.parent.name)

    def test_snapshot_roundtrip_preserves_loss_and_checks_metadata(self):
        item = self.ready_stage()
        path = item.data_dir / "psi_00000006.npy"
        self.io.save_wavefunction(path, item, 6)
        loaded = self.io.load_wavefunction(path, item)
        np.testing.assert_array_equal(loaded, item.psi)
        self.assertEqual(norm(loaded, item.x_array), norm(item.psi, item.x_array))
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual((metadata["step"], metadata["time"]), (6, 0.75))
        self.assertEqual(metadata["stage"], item.config)
        item.simulation_range += 1
        with self.assertRaisesRegex(ValueError, "simulation range"):
            self.io.load_wavefunction(path, item)
        item.simulation_range -= 1
        item.N_atoms += 1
        with self.assertRaisesRegex(ValueError, "reference N_atom"):
            self.io.load_wavefunction(path, item)
        bad = self.root / "bad.npy"
        np.save(bad, np.ones((4, 4, 4), dtype=np.complex64))
        with self.assertRaisesRegex(ValueError, "complex128"):
            self.io.load_wavefunction(bad, item)

    def test_snapshot_load_checks_legacy_grid_companion(self):
        item = self.ready_stage()
        path = self.root / "psi_final.npy"
        np.save(path, item.psi)
        grid = path.with_name("psi_final_grid.npz")
        np.savez(grid, x_array=item.x_array + 0.25, k_array=item.k_array)
        with self.assertRaisesRegex(ValueError, "grid does not match"):
            self.io.load_wavefunction(path, item)
        np.savez(grid, x_array=item.x_array, k_array=item.k_array)
        np.testing.assert_array_equal(self.io.load_wavefunction(path, item), item.psi)

    def test_snapshot_handles_multiple_slabs_and_short_last_slab(self):
        config = self.config()
        config["simulation"]["N_grids"] = 18
        item = self.parse(config)[0]
        values = np.arange(18**3).reshape(18, 18, 18)
        item.psi = np.asarray(values + 1j * values[::-1], dtype=np.complex128)
        path = self.root / "state.npy"
        self.io.save_wavefunction(path, item, 1)
        np.testing.assert_array_equal(np.load(path), item.psi)
        np.testing.assert_array_equal(self.io.load_wavefunction(path, item), item.psi)

    def test_duplicate_stage_output_folders_are_rejected(self):
        config = self.config()
        first = config["stages"][0]
        first["save"]["directory"] = "same_folder"
        second = json.loads(json.dumps(first))
        second["save"]["directory"] = "nested/../same_folder"
        config["stages"].append(second)
        with self.assertRaisesRegex(ValueError, "reuses another stage's output directory"):
            self.parse(config)

    def test_relative_output_override_cannot_escape_timestamped_run(self):
        config = self.config()
        config["stages"][0]["save"]["directory"] = "../shared"
        with self.assertRaisesRegex(ValueError, "stay inside the run folder"):
            self.parse(config)
        config["stages"][0]["save"]["directory"] = str(self.root / "absolute")
        item = self.parse(config)[0]
        self.assertEqual(item.data_dir.parent, self.root / "absolute")

    def test_save_schedules_are_independent_and_do_not_duplicate_final(self):
        item = self.ready_stage()
        item.save_params.update(psi_interval=2, scalar_interval=3, nk_interval=4,
                                save_initial=True, save_final=True)
        spectrum = dict(k=np.arange(2), nk=np.ones(2), Nk=np.ones(2))
        with patch.object(self.save, "get_N", side_effect=norm), \
             patch.object(self.save, "get_E", return_value=7), \
             patch.object(self.save, "get_L", return_value=2), \
             patch.object(self.save, "nk_extraction", return_value=spectrum) as extract, \
             patch.object(self.save, "save_wavefunction", wraps=self.io.save_wavefunction) as snapshot:
            self.save.save_outputs(item, 0, initial=True)
            for step in range(1, 7):
                self.save.save_outputs(item, step, final=(step == 6))
        self.assertEqual([call.args[2] for call in snapshot.call_args_list], [0, 2, 4, 6])
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(self.cp.fft.fftn.call_count, 3)  # Steps 3, 4, 6 only.
        with (item.monitor_dir / "scalars.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([int(row["step"]) for row in rows], [3, 6])
        self.assertAlmostEqual(float(rows[0]["N_atoms"]), item.N_atoms * norm(item.psi, item.x_array))
        with np.load(item.monitor_dir / "nk_00000004.npz") as saved:
            self.assertEqual(saved["time"], 0.5)
            np.testing.assert_array_equal(saved["nk"], spectrum["nk"])

    def test_disabled_output_does_no_fft_or_file_io(self):
        item = self.ready_stage()
        self.save.save_outputs(item, 0, initial=True)
        self.save.save_outputs(item, 6, final=True)
        self.cp.fft.fftn.assert_not_called()
        self.assertFalse(item.data_dir.exists())
        self.assertFalse(item.monitor_dir.exists())

    def test_monitor_arrays_include_axes_and_correct_projection_norm(self):
        item = self.ready_stage()
        item.psi[0, 0, 2] = 0
        item.monitor_params.update({key: 1 for key in item.monitor_params
                                    if key.startswith("show_") and key != "show_nk_interval"})
        with patch.object(self.save, "sum_abs2", side_effect=lambda z, axis: np.sum(np.abs(z)**2, axis=axis)):
            self.save.save_outputs(item, 1)
        self.assertEqual(self.cp.fft.fftn.call_count, 1)
        for name in ("TOF_xy", "TOF_yz"):
            with np.load(item.monitor_dir / f"{name}_00000001.npz") as saved:
                dk = item.k_array[1] - item.k_array[0]
                self.assertAlmostEqual(saved["values"].sum() * dk**2, norm(item.psi, item.x_array))
                self.assertEqual(saved["values"].shape, (4, 4))
        with np.load(item.monitor_dir / "density_xy_00000001.npz") as saved:
            dx = item.x_array[1] - item.x_array[0]
            self.assertAlmostEqual(saved["values"].sum() * dx**2, norm(item.psi, item.x_array))
        with np.load(item.monitor_dir / "phase_xy_00000001.npz") as saved:
            self.assertTrue(np.isnan(saved["values"][0, 0]))
            self.assertEqual(saved["z"], 0)

    def test_two_point_grid_has_positive_momentum_density(self):
        config = self.config()
        config["simulation"]["N_grids"] = 2
        item = self.parse(config)[0]
        item.psi = np.ones((2, 2, 2), dtype=np.complex128)
        item.monitor_params["show_TOF_slice_interval"] = 1
        self.save.save_outputs(item, 1)
        with np.load(item.monitor_dir / "TOF_slice_xy_00000001.npz") as saved:
            self.assertGreater(saved["values"].sum(), 0)

    def test_continued_and_loaded_stages_keep_state_and_reset_time(self):
        config = self.config()
        first = config["stages"][0]
        first["evolution"]["steps"] = 2
        second = json.loads(json.dumps(first))
        second["initial_state"]["type"] = "continue"
        second["potential"]["dynamic"].update(enabled=True, expression="t*y")
        config["stages"].append(second)
        stages = self.parse(config)
        times = []
        def make_propagator(item):
            def step(current, t=None):
                times.append(t)
                return current.psi * 0.5
            return step
        def normalize(psi, x):
            psi /= np.sqrt(norm(psi, x))
        with patch.object(self.stage_module, "make_real_propagator", side_effect=make_propagator) as factory, \
             patch("gpe.core.initial_states.normalise", side_effect=normalize):
            stages[0].prepare()
            psi = self.launch.launch_stage(stages[0])
            stages[0].release()
            self.assertAlmostEqual(norm(psi, stages[0].x_array), 1/16)
            stages[1].prepare(psi)
            self.assertIs(stages[1].psi, psi)
            self.launch.launch_stage(stages[1])
            self.assertEqual(factory.call_count, 2)
        self.assertEqual(times, [None, None, 0, 0.125])
        item = stages[1]
        path = item.data_dir / "reusable.npy"
        self.io.save_wavefunction(path, item, 2)
        expected = item.psi.copy()
        item.initial_state = dict(type="load", path=str(path))
        with patch.object(self.stage_module, "make_real_propagator", side_effect=make_propagator):
            item.prepare()
            np.testing.assert_array_equal(item.psi, expected)
            self.launch.launch_stage(item)
        self.assertEqual(times[-2:], [0, 0.125])

    def test_cli_runs_all_stages_and_saves_their_final_states(self):
        config = self.config()
        first = config["stages"][0]
        first["evolution"]["steps"] = 2
        first["save"]["save_final"] = True
        second = json.loads(json.dumps(first))
        second["initial_state"]["type"] = "continue"
        config["stages"].append(second)
        self.parse(config)
        runner = importlib.import_module("gpe.main")
        def normalize(psi, x):
            psi /= np.sqrt(norm(psi, x))
        with patch.object(sys, "argv", ["gpe.main", str(self.root / "experiment.yaml")]), \
             patch.object(self.stage_module, "make_real_propagator",
                          side_effect=lambda item: lambda current: current.psi * 0.5), \
             patch("gpe.core.initial_states.normalise", side_effect=normalize), patch("builtins.print"):
            runner.main()
        files = sorted((self.root / "data").glob("*/stage_*/psi_*.npy"))
        self.assertEqual(len(files), 2)
        np.testing.assert_allclose(np.load(files[1]), np.load(files[0]) * 0.25)

    def test_cli_check_does_not_import_gpu_modules(self):
        path = self.root / "check.yaml"
        path.write_text(dump_yaml(self.config()))
        for name in list(sys.modules):
            if name == "cupy" or name.startswith("gpe."):
                del sys.modules[name]
        # Block even an attempted CuPy import, whether or not the host has a GPU.
        sys.modules["cupy"] = None
        runner = importlib.import_module("gpe.main")
        output = io.StringIO()
        with patch.object(sys, "argv", ["gpe.main", str(path), "--check"]), \
             patch("sys.stdout", output):
            runner.main()
        self.assertIn("Valid configuration", output.getvalue())
        self.assertNotIn("gpe.stage", sys.modules)
        self.assertFalse((self.root / "data").exists())


if __name__ == "__main__":
    unittest.main()
