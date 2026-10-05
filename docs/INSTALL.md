# Install from nothing

You need exactly two things: **Python 3.11+** and **this repo**. No Docker, no CUDA,
no compiler, no GPU. Everything is prebuilt wheels. Total download about 52 MB.

---

## 1. Install Python

Check first — you may already have it:

```
python --version          # Windows
python3 --version         # macOS / Linux
```

If that prints `3.11` or higher, skip to step 2.

### Windows

Download the installer from <https://www.python.org/downloads/> (3.11 or newer) and run it.

**Tick "Add python.exe to PATH" on the first screen.** This is the one step people miss,
and without it `python` won't be found in a new terminal.

No admin rights? Choose "Customize installation" → "Install for me only". That works in
a locked-down corporate account.

If the Microsoft Store opens when you type `python`, Windows is showing a stub. Either
install from python.org as above, or disable the stub under
Settings → Apps → Advanced app settings → App execution aliases.

### macOS

**Requirements, both hard:** Apple Silicon (M1 or later) and macOS 14 (Sonoma) or newer.
Check with `uname -m` (must print `arm64`) and `sw_vers -productVersion`.

This is not a preference. `mujoco` 3.12.0 publishes only `macosx_11_0_arm64` wheels and
`onnxruntime` 1.30.0 only `macosx_14_0_arm64` — there is **no Intel macOS wheel for either**,
so an Intel Mac cannot install this at all, and macOS 13 or older fails on onnxruntime.

**Recommended — `uv` installs Python for you, no Homebrew and no admin rights:**

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then skip to step 2 and use `uv sync`. On a managed laptop this is
the path of least resistance: uv keeps its own Python under your home directory and never
touches the system one.

**With Homebrew, if you already have it:**

```
brew install python@3.11
```

**Neither available** (e.g. `curl | sh` blocked by policy): download
[python-3.11.9-macos11.pkg](https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg)
and run it. 3.11.9 is the newest 3.11 with a macOS installer — later 3.11.x are
source-only security releases.

### Linux

```
sudo apt install python3.11 python3.11-venv python3-pip     # Debian / Ubuntu
sudo dnf install python3.11                                  # Fedora / RHEL
```

The `-venv` package matters on Debian/Ubuntu — without it, step 2 fails.

---

## 2. Install the simulator

### With uv (one command does everything)

```
git clone https://github.com/Johannes4044/unitree-g1-mujoco-sim.git
cd unitree-g1-mujoco-sim
uv sync
```

That installs Python 3.11 if it is missing, creates `.venv`, and installs all 15
packages. Prefix commands with `uv run` and you never have to activate anything:

```
uv run python scripts/check_scenarios.py
```

### With pip

```
git clone https://github.com/Johannes4044/unitree-g1-mujoco-sim.git
cd unitree-g1-mujoco-sim

python -m venv .venv                 # on macOS/Linux: python3 -m venv .venv
.venv\Scripts\activate               # Windows
source .venv/bin/activate            # macOS / Linux

pip install -r requirements.txt
pip install -e .
```

The virtual environment keeps these 15 packages out of your system Python. You must
activate it in each new terminal — you'll know it worked because the prompt is
prefixed with `(.venv)`.

`pip install -e .` installs *this repo* as the `g1` package, so `import g1.robot`
works from any directory.

### Check it worked

```
python scripts/check_scenarios.py
```

Expect `5/5 scenarios OK` after about a minute. Then:

```
python workspace/01_hello_sim.py
```

which writes a PNG into `workspace/out/`. If both work, you're done — go to
[`workspace/README.md`](../workspace/README.md).

---

## 3. Offline install (no internet on the target machine)

For an air-gapped or restricted laptop. Do the first part on any machine **with**
internet, running the **same OS and Python version** as the target.

**On the networked machine:**

```
pip download -r requirements-lock.txt -d wheels
pip download hatchling editables -d wheels
```

The second line is easy to forget and the reason an offline install usually fails:
`hatchling` and `editables` are needed to *build* this repo into an installable
package, and pip only fetches them at build time. Without them the last step below
dies with `No matching distribution found for editables~=0.3`.

That fills `wheels/` with 21 `.whl` files, about 52 MB. Copy the **whole repo
folder including `wheels/`** to the target machine — a USB stick, a share, whatever
your transfer path is.

**On the target machine:**

```
python -m venv .venv
.venv\Scripts\activate                       # or: source .venv/bin/activate
pip install --no-index --find-links wheels -r requirements-lock.txt
pip install --no-index --find-links wheels -e .
```

`--no-index` forbids pip from reaching the network, so this fails loudly rather
than silently phoning home. Use `requirements-lock.txt` here, not
`requirements.txt` — the lock pins every transitive dependency, so the target gets
exactly what you downloaded.

If the target OS differs from the download machine, add the platform to the
download step, e.g.:

```
pip download -r requirements-lock.txt -d wheels \
  --platform win_amd64 --python-version 3.11 --only-binary :all:
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `python: command not found` | Not on PATH. Reinstall ticking "Add to PATH", or use `py -3.11` on Windows. |
| `No module named venv` | Debian/Ubuntu: `sudo apt install python3.11-venv`. |
| `No module named g1` | `pip install -e .` not run, or the venv isn't activated. |
| `onnxruntime` has no matching wheel | Python older than 3.11, macOS older than 14, or an **Intel** Mac (arm64 wheels only). |
| `ARB_clip_control unavailable...` on stderr | Harmless. A depth-precision notice from the renderer, not an error. |
| Rendering fails on a headless Linux box | Needs an OpenGL context. Try `MUJOCO_GL=egl`, or `osmesa` for pure software. |
| `pip` is very slow or blocked | Corporate proxy. Use the offline path in step 3. |
| Offline: `No matching distribution found for editables` | You skipped `pip download hatchling editables -d wheels` on the networked machine. |

---

## What you are installing

| Package | Licence | Why |
|---|---|---|
| `mujoco` | Apache-2.0 | physics and rendering |
| `numpy` | BSD-3-Clause | arrays |
| `pillow` | MIT-CMU | writes PNGs and WebPs |
| `imageio` | BSD-2-Clause | image IO used by the render scripts |
| `onnxruntime` | MIT | CPU inference for the Holosoma walking policy |

Plus 10 transitive dependencies, all permissive. No copyleft, no proprietary
components, nothing that needs a licence decision before you ship it. See
[`THIRD-PARTY-NOTICES.md`](../THIRD-PARTY-NOTICES.md).
