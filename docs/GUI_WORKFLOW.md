# One interface for the complete workflow

Start by double-clicking `Open GPE.command` on macOS, or run
`python -m gui_periphery.app` from a Python environment containing
`gui_periphery/requirements.txt`. This launches a local Python service and opens
its browser interface. CuPy is needed only on the machine running the simulation.

The launcher opens a private URL containing a fresh session token. Use that opened
window; a bare localhost address intentionally cannot control an authenticated SSH
session. The token is not a password and changes whenever the GUI service restarts.

## Experiment

Create the stage sequence using the existing editor, or open a schema-2 YAML file.
Validation and potential-expression behavior are unchanged. You can save YAML,
but it is not a prerequisite for the next step: **Prepare run** takes the current
draft directly and writes the YAML for you.

## Launch

Choose a run name and write the purpose/description in **Purpose & notes**.
Choose **CSD3 / SSH + Slurm** or **Machine running this GUI**. The second choice
means the machine running the GUI's Python service, not necessarily the device
displaying the browser. Running the GUI on your Mac does not give it Colab's GPU.
On macOS you can prepare and download a run, but local CUDA setup and launch are
disabled. For a Colab test, open the GUI inside the GPU runtime using the cell below.

For CSD3, enter your GPU account, partition, wall time, and site Python/CUDA module
names. These are environment settings, separate from the physics YAML. The GUI
generates `environment.sh` and `submit.sh`; you do not have to edit shell files.
CuPy for CUDA 12/13 selects the wheel installed by environment setup. The Python
field selects the interpreter used to create a new virtual environment.

Click **Prepare run**. This creates a distinct folder, records notes and source
hashes, and copies the stage settings. Launch always uses this prepared copy.
Changing the draft or launch settings afterwards requires preparing a new run.
Output roots are set to `saved_data` and `monitor_data` inside the run folder.
Loaded wavefunctions use absolute paths on the simulation machine; preparation
and upload do not copy existing wavefunctions.

For HPC:

1. Enter the hostname, SSH username and port. Click **Connect**.
2. Respond to OpenSSH's prompts in the GUI. These may ask for a key passphrase,
   password, one-time code or confirmation of a new host fingerprint. Verify new
   fingerprints against [CSD3's published keys](https://docs.hpc.cam.ac.uk/hpc/user-guide/hostkeys.html).
3. Enter the absolute remote parent directory and click **Upload run**. A new named
   directory is created there; existing directories are not overwritten.
4. Click **Set up environment** if dependencies need installing. This creates a
   per-run virtual environment remotely, using the chosen modules and CuPy wheel.
   Its output appears in Activity. Skip this when the chosen environment is already
   ready. The cluster must permit package installation from that node/network.
5. Click **Submit GPU job**. The interface records the Slurm job ID. Submission is
   distinct from running: the job may queue. Refresh shows the scheduler state.

OpenSSH retains its normal known-host checks and uses a private connection-sharing
socket. Passwords and codes travel through OpenSSH's askpass mechanism via an
authenticated local prompt bridge. Responses are cleared after delivery and are
not written into profiles, browser storage, job files or logs. GUI profiles retain
only non-secret fields such as host, username, directories and environment choices.
[CSD3 MFA and connection sharing](https://docs.hpc.cam.ac.uk/hpc/user-guide/mfa.html).

The GUI requests one GPU and one process; adding GPU requests would not make this
solver distribute its calculation. Confirm the actual module names and account
for your CSD3 allocation before its first live job. Local tests use fake SSH and
Slurm responses; successful CSD3 login/submission has not yet been demonstrated.

For a GPU on the same machine as the Python GUI service, click **Launch simulation**.
The GUI checks CuPy/device availability and configuration before starting the run.
You can use the existing environment, especially in Colab, or create a per-run
environment with the setup button.

## Monitor

Open **Monitor**, click **Refresh now**, or enable **Auto refresh**. Refresh intervals
are measured in wall-clock seconds and do not change any YAML sampling intervals.

- Momentum spectrum: full saved k range by default, optional maximum k, log–log
  toggle and selection of a previously cached/saved spectrum. Incomplete shells
  are grey. Zero/non-positive values are omitted only in the logarithmic display.
- Images: density, phase or TOF as available in the raw files. TOF defaults to ±5;
  set its limit to zero for the full available range. Images are sampled to at most
  256 pixels per axis for display. The raw files are unchanged.
- Scalars: latest saved time, norm, kinetic energy and angular momentum, plus a
  selectable history plot. Long histories are thinned for display only.
- Logs: the most recent tail of the simulation output.

On HPC, refresh fetches changed latest files for each diagnostic/stage and scalar
CSV files. It retains fetched samples locally. It does not download full 3D
wavefunctions or automatically backfill all earlier spectrum files. A separate
HPC transfer is needed if you want the complete historical dataset for analysis.
Incomplete NPZ files and partially appended CSV rows are retried/skipped; a readable
earlier sample remains available. Captions show the saved time of each plot, which
can differ between diagnostics with different sampling cadences.

**Reopen a prepared run** restores its saved job details. Reconnect to the same
HPC host/user before refreshing it. Slurm jobs keep running when the GUI or SSH
connection closes. Locally launched processes also detach from the GUI, but a
Colab runtime reset still destroys its processes and `/content` files. After a
GUI service restart, local jobs can be viewed and their completion record read;
the GUI does not cancel a local PID it no longer owns. Slurm jobs can be cancelled
after reconnecting. **Cancel job** asks for confirmation before stopping a job.

You can also enter an existing local monitor folder in the Monitor tab to inspect
previous results. This needs neither SSH nor a GPU.

## Colab bootstrap

Choose a GPU runtime. With the current software synced to
`MyDrive/GPE_simulator`, this cell mounts Drive, copies the software into the
runtime, installs the GUI dependencies and opens the interface. It does not copy
Mac virtual environments or past runs, and leaves the Drive source unchanged.

```python
from google.colab import drive
from pathlib import Path
import shutil

drive.mount("/content/drive")
source = Path("/content/drive/MyDrive/GPE_simulator")
destination = Path("/content/GPE_simulator")
if not (source / "gui_periphery/colab.py").is_file():
    raise FileNotFoundError(f"Updated software not found in {source}; check Drive sync and the folder name.")
destination.mkdir(parents=True, exist_ok=True)
for folder in ("gpe", "config", "gui_periphery"):
    shutil.copytree(source / folder, destination / folder, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc", ".DS_Store"))
shutil.copy2(source / "requirements.txt", destination / "requirements.txt")

%cd /content/GPE_simulator
%pip install -q -r gui_periphery/requirements.txt
from gui_periphery.colab import show
show()
```

This displays the same interface through Colab's authenticated port proxy with a
per-session access token. In **Launch**, choose **Machine running this GUI**. If
the runtime already has the matching CuPy installation, you can prepare and launch
without the environment-setup step. Otherwise use the setup button after selecting
the appropriate CUDA wheel. Begin with a small experiment and check its monitor
outputs before a long launch. Nothing silently changes your grid or stage steps.

The proxy entry point uses Colab's
[`serve_kernel_port_as_iframe`](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py).
On 12 October 2026, the user confirmed that simulation launch, configuration editing
and monitoring worked in Colab. Notebook disconnection/runtime lifetime limitations
remain unchanged; store important outputs on mounted Drive or copy them there.

## Implementation and verification

All additions are in `gui_periphery/`, with tests and documentation. The solver,
physics kernels, stage loop, diagnostics calculations and output writer were not
modified. The existing CLI remains available for scripting, but the normal GUI
workflow needs no manually entered upload, environment, plotting or submit command.

The regression suite covers preparation, archive contents, command quoting,
submission errors/job IDs, changed-file transfer, partial diagnostics, authentication
answer cleanup, HTTP origin/session protection and a complete askpass round trip.
An isolated browser test covered preparing/downloading a run and plotting synthetic
monitor data with the time/image/scalar controls. These tests do not substitute for
an actual GPU/SSH/Slurm integration run.

Local verification for this integration: **76 tests passed, 10 GPU-related skips**.
The browser test used synthetic monitor fixtures and did not launch the solver or
authenticate to CSD3. The core and shared configuration files have no changes from
the preceding reorganization commit.

The subsequent macOS setup guard passed 22 workflow/bundle tests and a browser
check. The Colab result above is user-reported live validation. The user also
confirmed CSD3 authentication, upload, environment setup and submission on
12 October 2026. GPU execution and live remote monitoring on CSD3 remain unverified
while the submitted job waits in the queue.
