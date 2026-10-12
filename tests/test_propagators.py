"""Small CUDA checks against analytical solutions. Run with unittest on a GPU."""

from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError as exc:
    if exc.name != "cupy":
        raise
    cp = None


class PropagatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cp is None:
            raise unittest.SkipTest("CuPy and a CUDA GPU are required")
        try:
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise unittest.SkipTest("A CUDA GPU is required")
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise unittest.SkipTest(f"CUDA is unavailable: {exc}") from exc

    def setUp(self):
        from config_editor.configuration import dump_yaml, new_config
        from gpe.yaml_parser import parse_yaml
        self.dump_yaml, self.parse_yaml = dump_yaml, parse_yaml
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = new_config()
        self.config["simulation"].update(N_grids=8, simulation_range=2*np.pi,
                                         data_root=str(self.root / "data"),
                                         monitor_root=str(self.root / "monitor"))
        settings = self.config["stages"][0]
        settings["box"]["enabled"] = False
        settings["cloud"]["a"] = 0
        settings["evolution"].update(steps=3, dt=0.01)
        settings["save"].update(scalar_interval=0, save_final=False)
        settings["monitor"] = dict.fromkeys(settings["monitor"], 0)

    def stage(self):
        path = self.root / "test.yaml"
        path.write_text(self.dump_yaml(self.config))
        item = self.parse_yaml(path)[0]
        item.prepare()
        return item

    def plane_wave(self, item, mode=1):
        wave = np.exp(1j * mode * cp.asnumpy(item.x_array)[:, None, None])
        return np.broadcast_to(wave, (item.N_grids,) * 3).copy() / (2*np.pi)**1.5

    def test_real_phase_for_all_potential_and_cutoff_combinations(self):
        from gpe.launch import launch_stage
        from gpe.helper import get_N
        settings = self.config["stages"][0]
        settings["evolution"]["type"] = "real"
        settings["cloud"]["a"] = 10  # Constant density also gives an exact nonlinear phase.
        for static in (False, True):
            for dynamic in (False, True):
                for cutoff in (False, True):
                    with self.subTest(static=static, dynamic=dynamic, cutoff=cutoff):
                        settings["potential"]["static"].update(enabled=static, expression="1/2")
                        settings["potential"]["dynamic"].update(enabled=dynamic, expression="t/4")
                        settings["k_cut"].update(enabled=cutoff, value=1.5)
                        item = self.stage()
                        initial = self.plane_wave(item)
                        item.psi = cp.asarray(initial)
                        time = item.steps * item.dt
                        phase = (1 + item.beta / (2*np.pi)**3) * time
                        phase += (0.5*time if static else 0) + (time**2/8 if dynamic else 0)
                        result = launch_stage(item)
                        np.testing.assert_allclose(cp.asnumpy(result), initial*np.exp(-1j*phase),
                                                   rtol=1e-11, atol=1e-12)
                        self.assertAlmostEqual(float(get_N(result, item.x_array)), 1, places=11)
                        item.release()

    def test_generated_spatial_expressions_at_each_grid_coordinate(self):
        from gpe.make_kernel import make_imag_spatial_kernel, make_real_spatial_kernel
        settings = self.config["stages"][0]
        # Odd side length exercises a final CUDA block with unused threads.
        self.config["simulation"]["N_grids"] = 7
        settings["potential"]["static"].update(enabled=True, expression="x/2 + y**2/10 + abs(z)/5")
        for kind, dynamic in (("imaginary", False), ("real", False), ("real", True)):
            with self.subTest(kind=kind, dynamic=dynamic):
                settings["evolution"]["type"] = kind
                settings["potential"]["dynamic"].update(enabled=dynamic, expression="sin(t)*x + cos(pi)*z")
                item = self.stage()
                item.psi.fill(0.2 + 0.1j)
                x = cp.asnumpy(item.x_array)
                X, Y, Z = x[:, None, None], x[None, :, None], x[None, None, :]
                potential = X/2 + Y**2/10 + np.abs(Z)/5
                time = 0.37
                if dynamic:
                    potential = potential + np.sin(time)*X - Z
                kernel = make_imag_spatial_kernel(item) if kind == "imaginary" else make_real_spatial_kernel(item)
                args = (item.psi, item.staticV, item.x_array, float(item.beta), float(item.dt), int(item.N_grids))
                if dynamic:
                    args += (time,)
                kernel(((item.psi.size + 255)//256,), (256,), args)
                exponent = -potential * item.dt/2 * (1 if kind == "imaginary" else 1j)
                np.testing.assert_allclose(cp.asnumpy(item.psi), (0.2+0.1j)*np.exp(exponent),
                                           rtol=1e-12, atol=1e-12)
                item.release()

    def test_imaginary_decay_and_normalization_with_optional_static_potential(self):
        from gpe.launch import launch_stage
        settings = self.config["stages"][0]
        for static in (False, True):
            with self.subTest(static=static):
                settings["potential"]["static"].update(enabled=static, expression="1/2")
                item = self.stage()
                low, high = self.plane_wave(item, 1), self.plane_wave(item, 2)
                item.psi = cp.asarray(low + high)
                time = item.steps * item.dt
                expected = low*np.exp(-time) + high*np.exp(-4*time)
                dx = float(item.x_array[1] - item.x_array[0])
                expected /= np.sqrt(np.sum(np.abs(expected)**2)*dx**3)
                np.testing.assert_allclose(cp.asnumpy(launch_stage(item)), expected,
                                           rtol=1e-11, atol=1e-12)
                item.release()

    def test_absorber_keeps_analytical_norm_loss(self):
        from gpe.launch import launch_stage
        from gpe.helper import get_N
        self.config["stages"][0]["evolution"]["type"] = "real"
        item = self.stage()
        initial = self.plane_wave(item)
        item.psi = cp.asarray(initial)
        gamma = 0.4
        item.staticV.fill(-1j*gamma)
        time = item.steps * item.dt
        result = launch_stage(item)
        np.testing.assert_allclose(cp.asnumpy(result), initial*np.exp((-gamma-1j)*time),
                                   rtol=1e-11, atol=1e-12)
        self.assertAlmostEqual(float(get_N(result, item.x_array)), np.exp(-2*gamma*time), places=11)

    def test_cutoff_removes_high_mode_without_renormalizing(self):
        from gpe.launch import launch_stage
        from gpe.helper import get_N
        settings = self.config["stages"][0]
        settings["evolution"]["type"] = "real"
        settings["k_cut"].update(enabled=True, value=1.5)
        item = self.stage()
        low, high = self.plane_wave(item, 1), self.plane_wave(item, 2)
        item.psi = cp.asarray((low + high)/np.sqrt(2))
        result = launch_stage(item)
        expected = low*np.exp(-1j*item.steps*item.dt)/np.sqrt(2)
        np.testing.assert_allclose(cp.asnumpy(result), expected, rtol=1e-11, atol=1e-12)
        self.assertAlmostEqual(float(get_N(result, item.x_array)), 0.5, places=11)

    def test_snapshot_gpu_roundtrip_across_slab_boundary(self):
        from gpe.state_io import save_wavefunction, load_wavefunction
        self.config["simulation"]["N_grids"] = 18
        item = self.stage()
        path = self.root / "state.npy"
        original = cp.asnumpy(item.psi)
        save_wavefunction(path, item, 0)
        np.testing.assert_array_equal(cp.asnumpy(load_wavefunction(path, item)), original)

    def test_yaml_run_with_two_stages_and_all_saved_diagnostics(self):
        from gpe.main import main
        first = self.config["stages"][0]
        first["evolution"]["steps"] = 2
        first["save"].update(save_final=True, scalar_interval=1, nk_interval=1)
        first["monitor"].update({name: 1 for name in first["monitor"] if name.startswith("show_")})
        second = json.loads(json.dumps(first))
        second["evolution"]["type"] = "real"
        second["initial_state"]["type"] = "continue"
        second["box"].update(enabled=True, R=1, L=2, wall_thickness=0.5, trap_depth=0.3)
        second["noise"]["enabled"] = True
        second["absorber"].update(enabled=True, activation_distance=0.5, strength=0.2)
        second["potential"]["static"].update(enabled=True, expression="x**2/20")
        second["potential"]["dynamic"].update(enabled=True, expression="cos(t)*y/10")
        second["k_cut"].update(enabled=True, value=2)
        self.config["stages"].append(second)
        path = self.root / "stages.yaml"
        path.write_text(self.dump_yaml(self.config))
        with patch.object(sys, "argv", ["gpe.main", str(path)]), patch("builtins.print"):
            main()
        states = sorted((self.root / "data").glob("*/stage_*/psi_*.npy"))
        self.assertEqual(len(states), 2)
        for state in states:
            saved = np.load(state)
            self.assertEqual(saved.dtype, np.complex128)
            self.assertTrue(np.isfinite(saved).all())
        scalar_files = list((self.root / "monitor").glob("*/stage_*/scalars.csv"))
        self.assertEqual(len(scalar_files), 2)
        for scalar in scalar_files:
            with scalar.open() as file:
                self.assertEqual([row["step"] for row in csv.DictReader(file)], ["1", "2"])
            self.assertEqual(len(list(scalar.parent.glob("*_00000002.npz"))), 8)


if __name__ == "__main__":
    unittest.main()
