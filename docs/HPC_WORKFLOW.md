# From local YAML to a GPU job

This is the command-line workflow that the future GUI will drive. The folder
builder exists now; authentication, upload, submission, and live viewing are not
yet connected to GUI buttons. The commands below are examples, not actions already
performed. Replace the account, username, remote directory, and environment choices
with your own values.

## What the two launch scripts do

`run.sh` sources this run's `environment.sh`, enters the simulation folder, and
runs `python -u -m gpe.main config/run.yaml`, saving terminal output to `logs/run.log`.
`-u` makes Python print promptly; `-m` runs a Python module. `2>&1` combines stderr
and stdout. `| tee -a logs/run.log` displays and appends that output to the log.
`set -o pipefail` ensures a Python failure remains a failed command even when `tee`
succeeds. It does not make a job survive a disconnected terminal.

`submit.sh` describes the resources for Slurm and invokes `run.sh` on the allocated
GPU node. `sbatch submit.sh` submits it to the scheduler and returns a job ID.
Submission does not mean execution has started: the job can wait in the queue.
After successful batch submission, closing the SSH connection does not stop the
job. See [CSD3 batch jobs](https://docs.hpc.cam.ac.uk/hpc/user-guide/batch.html).

## Typical CSD3 upload and launch

Locally, construct the YAML and prepare a run:

```bash
python -m gui_periphery.editor.app
python -m gui_periphery.prepare_run config/my_run.yaml \
  --name simulation_rotation_test \
  --notes-file description.txt
```

Connect using ordinary SSH, responding to its password/key and one-time-code
prompts. Keep normal host-key verification:

```bash
ssh USER@login-icelake.hpc.cam.ac.uk
```

On CSD3, choose your project storage location and prepare a parent directory:

```bash
mkdir -p /rds/PROJECT/YOUR_DIRECTORY/simulations
exit
```

Back on the local computer, upload only the prepared folder:

```bash
rsync -av runs/simulation_rotation_test \
  USER@login-icelake.hpc.cam.ac.uk:/rds/PROJECT/YOUR_DIRECTORY/simulations/
ssh USER@login-icelake.hpc.cam.ac.uk
```

On CSD3, configure the per-run environment and submission template:

```bash
cd /rds/PROJECT/YOUR_DIRECTORY/simulations/simulation_rotation_test
less /usr/local/Cluster-Docs/SLURM/slurm_submit.wilkes3
module avail
nano environment.sh
nano requirements-gpu.txt
nano submit.sh
```

Use the current site template to choose the Python/CUDA modules. Set your GPU
account and requested wall time in `submit.sh`. The supplied template requests
one task, one GPU and three CPU cores on `ampere`. This matches the documented
Wilkes3 resource form; confirm it against the template installed on CSD3 before
your first submission. The solver currently uses one GPU; allocating additional
GPUs will not distribute the calculation automatically.
[CSD3 template guidance](https://docs.hpc.cam.ac.uk/hpc/user-guide/quickstart.html)
and [GPU resource directives](https://docs.hpc.cam.ac.uk/hpc/user-guide/batch.html).

Set up the run's environment where permitted by the cluster, then validate and
submit. Environment setup installs dependencies but does not execute the solver:

```bash
bash setup_environment.sh
bash run.sh --check
sbatch --parsable submit.sh > logs/job_id.txt
cat logs/job_id.txt
squeue -u "$USER"
```

Dependency installation requires access to package sources; where unavailable,
use site-provided packages or a prepared wheel cache. Choose one CuPy build that
matches the selected CUDA environment; see the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).
`requirements-gpu.txt` can also pin the version for that launch. The setup script
records installed Python package versions in `logs/environment.txt`.

Once the job starts:

```bash
tail -f logs/run.log
```

Slurm errors appear in `logs/slurm-JOBID.err`; scheduler stdout is in the matching
`.out` file. A submitted job does not require the local GUI or SSH session to remain
open. Snapshots are reusable states; interrupted stages do not yet resume at their
saved step automatically.

## Authentication and small-file monitoring

CSD3 documents connection sharing with this local `~/.ssh/config` entry:

```sshconfig
Host login*.hpc.cam.ac.uk
    ControlMaster auto
    ControlPath ~/.ssh/control-%C
    ControlPersist 900
```

The initial connection still needs MFA. Later connections to the same login node
can reuse it. The planned GUI should use the normal SSH authentication prompts and
known-host checks, retaining neither passwords nor one-time codes in run folders
or logs. [CSD3 MFA and connection sharing](https://docs.hpc.cam.ac.uk/hpc/user-guide/mfa.html).

For a manual preview transfer, run on your local machine:

```bash
mkdir -p runs/simulation_rotation_test/monitor_data
rsync -av \
  USER@login-icelake.hpc.cam.ac.uk:/rds/PROJECT/YOUR_DIRECTORY/simulations/simulation_rotation_test/monitor_data/ \
  runs/simulation_rotation_test/monitor_data/
```

This selects monitor data only. Full 3D wavefunctions remain under remote
`saved_data/`. Before enabling continuous GUI refresh, publish diagnostic files
atomically or make the reader retry incomplete files; today's writers were kept
unchanged by the refactor. GUI refresh rate must be independent of simulation
sampling intervals. The GUI should show the last completed sample and its time,
plus job state, rather than treating an old sample as current progress.

## Colab mock run, after reviewing the refactor

The mock checks the same runtime bundle, YAML, raw monitor outputs, and eventual
local viewer. It cannot test Slurm, cluster SSH/MFA, or disconnect persistence.
First run the GPU tests from a copy of this branch on Colab:

```bash
cd /content/GPE_simulator
python -m pip install -r requirements.txt
# Choose the CuPy wheel matching this runtime; CUDA 12 example:
python -m pip install cupy-cuda12x
python -m unittest discover -s tests -p test_propagators.py -v
```

Transfer the prepared run folder to Drive, mount Drive in Colab, and copy the folder
to `/content`. For example:

```bash
cp -a /content/drive/MyDrive/GPE_simulator/runs/simulation_rotation_test /content/
cd /content/simulation_rotation_test
python -m gpe.main config/run.yaml --check
bash run.sh
```

The bundle's requirements must already be installed in the active Colab environment
as above. Its `environment.sh` leaves that environment active when no local `.venv`
exists. Use a small YAML first, chosen through the editor; preparation never silently
reduces your grid size or number of steps. Copy completed results to persistent
storage before the runtime expires. No mock or production simulation was launched
as part of this reorganization.
