"""Study controls and analytic numerical checks; GPU checks skip without CUDA."""

import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import zipfile

import numpy as np

from config.yaml_parser import new_config, new_stage, normalize_config
from gpe.io.analyze_power_law import local_slopes, power_fit
from gpe.power_law_study import budget_steps, build_cases, evolve, prepare_config, publish, timed_steps
from gpe.io.study_diagnostics import physical_scalars, refine_field

ROOT = Path(__file__).resolve().parents[1]


class StudyTests(unittest.TestCase):
    def test_controls_start_from_the_same_snapshot_and_change_one_factor(self):
        template = new_config()
        template['simulation']['N_grids'] = 256
        template['stages'] = [new_stage(imaginary=index == 0) for index in range(7)]
        for index, stage in enumerate(template['stages']):
            stage['cloud'].update(N_atom=300000, a=20 if index == 6 else 0)
            stage['noise'].update(enabled=True, seed=42)
            stage['evolution']['steps'] = [2000,25000,25000,25000,1000,100000,100000][index]
        original = copy.deepcopy(template)
        cases = build_cases(template, Path('/initial.npy'), Path('/final.npy'))
        self.assertEqual(template, original)
        self.assertEqual(len(cases), 12)
        by_name = {c['name']: c for c in cases}
        for case in cases:
            normalize_config(case['config'])
            if case['name'].startswith('seed_'):
                seed = int(case['name'].split('_')[-1])
                self.assertEqual({s['noise']['seed'] for s in case['config']['stages']}, {seed})
                self.assertEqual(case['config']['stages'][0]['initial_state']['type'], 'gaussian')
                self.assertTrue(all(s['initial_state']['type'] == 'continue' for s in case['config']['stages'][1:]))
            else:
                stage = case['config']['stages'][0]
                self.assertEqual(stage['initial_state']['type'], 'load')
                self.assertEqual(stage['initial_state']['path'], '/final.npy' if case['name']=='long_a20' else '/initial.npy')
                self.assertEqual(stage['potential'], template['stages'][-1]['potential'])
                self.assertEqual(stage['box'], template['stages'][-1]['box'])
                self.assertEqual(stage['cloud']['N_atom'], 300000)
        self.assertEqual(by_name['dt_quarter']['config']['stages'][0]['evolution']['steps'], 400000)
        self.assertEqual(by_name['long_a20']['time_offset'], 1000)
        self.assertEqual(by_name['a10']['config']['stages'][0]['evolution']['steps'], 400000)
        self.assertFalse(by_name['clean_a20_512']['config']['stages'][0]['noise']['enabled'])

    def test_complex_refinement_preserves_nodes_norm_and_signed_modes(self):
        n, m, length = 8, 16, 7.0
        j = np.arange(n)
        # Includes a negative Nyquist mode: no real-signal Nyquist splitting.
        psi = (0.12*np.exp(2j*np.pi*(-4*j[:, None, None]+2*j[None, :, None]-j[None, None, :])/n)
               + 0.06*np.exp(2j*np.pi*j[None, :, None]/n))*np.ones((n,n,n))
        refined = refine_field(psi, m)
        np.testing.assert_allclose(refined[::2,::2,::2], psi, atol=1e-14)
        self.assertAlmostEqual(float(np.sum(abs(psi)**2)*(length/n)**3),
                               float(np.sum(abs(refined)**2)*(length/m)**3), places=12)
        fine_j = np.arange(m)
        expected = (0.12*np.exp(2j*np.pi*(-4*fine_j[:, None, None]+2*fine_j[None, :, None]-fine_j[None, None, :])/m)
                    + 0.06*np.exp(2j*np.pi*fine_j[None, :, None]/m))*np.ones((m,m,m))
        np.testing.assert_allclose(refined, expected, atol=1e-14)

    def test_energies_against_exact_plane_wave(self):
        n, length, beta, v, amplitude = 12, 8, 7, 2.5, 0.03
        j = np.arange(n)
        psi = amplitude*np.exp(2j*np.pi*(j[:,None,None]+2*j[None,:,None])/n)*np.ones((n,n,n))
        k = 2*np.pi*np.fft.fftfreq(n, length/n)
        values = physical_scalars(psi, np.fft.fftn(psi), np.full(psi.shape,v-0.1j), beta, length/n, k)
        norm = amplitude**2*length**3
        self.assertAlmostEqual(values['norm'], norm)
        self.assertAlmostEqual(values['E_kin'], norm*5*(2*np.pi/length)**2)
        self.assertAlmostEqual(values['E_potential'], norm*v)
        self.assertAlmostEqual(values['E_interaction'], beta/2*amplitude**4*length**3)
        self.assertAlmostEqual(values['kx2_fraction'], 0.2)
        self.assertAlmostEqual(values['ky2_fraction'], 0.8)
        self.assertAlmostEqual(values['kz2_fraction'], 0)
        self.assertAlmostEqual(values['E_total'], sum(values[k] for k in ('E_kin','E_potential','E_interaction')))

    def test_budget_never_extends_requested_run_and_keeps_sample_boundaries(self):
        self.assertEqual(budget_steps(200000,.01,10000,.0324),200000)
        self.assertEqual(budget_steps(200000,.01,500,.0324),0)
        steps = budget_steps(200000,.01,4000,.0324)
        self.assertEqual(steps%1000,0)
        self.assertLessEqual(steps*.0324+600,4000)

    def test_fits_distinguish_power_law_and_log_curvature(self):
        k = np.linspace(.09,5,70)
        nk = .03*k**-3.2
        fit = power_fit(k,nk)
        self.assertAlmostEqual(fit['alpha'],3.2)
        self.assertLess(fit['log_rms'],1e-12)
        np.testing.assert_allclose(local_slopes(k,nk)[:,1],3.2,atol=1e-12)
        curved = power_fit(k,nk*np.exp(.4*np.log(k)**2))
        self.assertAlmostEqual(curved['log_curvature'],.4)
        self.assertGreater(curved['log_rms'],.02)
        self.assertTrue(np.isnan(power_fit(k,nk,.3,.4)['alpha']))

    def test_review_archive_excludes_large_states_and_preserves_raw_spectra(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'study'; folder.mkdir()
            for name in ('configs','analysis','monitor/case','data/case'):
                (folder/name).mkdir(parents=True)
            (folder/'manifest.json').write_text('{}')
            (folder/'README.txt').write_text('review')
            np.savez(folder/'monitor/case/nk_00000000.npz', k=[0,1],nk=[2,3])
            np.save(folder/'data/case/psi.npy',np.ones(2))
            np.savez(folder/'monitor/case/TOF_xy_00000000.npz',values=np.ones((2,2)))
            archive = publish(folder,Path(temp)/'share')
            with zipfile.ZipFile(archive) as z:
                self.assertIn('monitor/case/nk_00000000.npz',z.namelist())
                self.assertFalse(any('psi.npy' in p or 'TOF_' in p for p in z.namelist()))
            self.assertEqual(archive.read_bytes(),(Path(temp)/'share/study/review.zip').read_bytes())

    def test_timing_restores_input_even_for_in_place_propagators(self):
        original = np.array([1+2j,3-4j])
        stage = SimpleNamespace(psi=original.copy(), dt=.01,
                                potential_params={'dynamic':{'activate':False}})
        def propagate(item):
            item.psi *= .9
            return item.psi
        stage.propagator = propagate
        device = SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None)))
        self.assertGreater(timed_steps(stage,device,steps=3),0)
        np.testing.assert_array_equal(stage.psi,original)

    def test_deadline_keeps_last_spectrum_and_avoids_duplicate_scalar_rows(self):
        stage = SimpleNamespace(steps=150,dt=.1,name='test',psi=np.array([1.0]),
                                data_dir=Path('/unused'),potential_params={'dynamic':{'activate':False}})
        stage.propagator = lambda item:item.psi
        device = SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None)))
        save = Mock()
        with patch.dict('sys.modules', {'gpe.io.state_io':SimpleNamespace(save_wavefunction=save)}), \
             patch('gpe.io.study_diagnostics.record') as record:
            steps = evolve(stage,device,0,1000,True,lambda message:None)
        self.assertEqual(steps,100)
        self.assertEqual([call.args[1] for call in record.call_args_list],[0,100,100])
        self.assertFalse(record.call_args_list[-1].kwargs['scalar_row'])
        self.assertEqual(save.call_args.args[2],100)


try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False


@unittest.skipUnless(GPU_AVAILABLE, 'requires CuPy and a CUDA GPU')
class StudyGpuTests(unittest.TestCase):
    def test_interpolation_on_gpu_matches_cpu(self):
        rng = np.random.default_rng(4)
        psi = (rng.normal(size=(8,8,8))+1j*rng.normal(size=(8,8,8)))*0.01
        np.testing.assert_allclose(cp.asnumpy(refine_field(cp.asarray(psi),16,cp)), refine_field(psi,16),atol=1e-13)

    def test_record_and_runner_save_t0_exact_final_and_offset(self):
        from gpe.stage_factory import make_stages
        from gpe.power_law_study import timed_steps
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'configs').mkdir()
            config = new_config()
            config['simulation'].update(N_grids=16,simulation_range=12)
            item = config['stages'][0]
            item['evolution'].update(type='real',steps=4,dt=.01)
            item['absorber']['enabled'] = False
            item['box']['enabled'] = False
            item['cloud']['a'] = 0
            case = dict(name='tiny',config=config)
            stages = make_stages(prepare_config(case,folder))
            stage = stages[0]
            stage.prepare()
            before = stage.psi.copy()
            self.assertGreater(timed_steps(stage,cp,steps=2),0)
            np.testing.assert_array_equal(cp.asnumpy(stage.psi),cp.asnumpy(before))
            count = evolve(stage,cp,time.monotonic()+60,1000,True,lambda message:None)
            self.assertEqual(count,4)
            with np.load(stage.monitor_dir/'nk_00000000.npz') as initial:
                self.assertEqual(float(initial['time']),1000)
                self.assertAlmostEqual(float(initial['Nk'].sum()),1,places=10)
            with np.load(stage.monitor_dir/'nk_00000004.npz') as final:
                self.assertEqual(float(final['time']),1000.04)
            rows = np.genfromtxt(stage.monitor_dir/'scalars.csv',names=True,delimiter=',')
            self.assertEqual(len(rows),2)
            self.assertTrue((stage.monitor_dir/'TOF_xy_00000004.npz').exists())
            self.assertTrue((stage.data_dir/'psi_00000004.npy').exists())
            self.assertAlmostEqual(rows['E_total'][0],rows['E_total'][-1],places=10)
            stage.release()


if __name__ == '__main__':
    unittest.main()
