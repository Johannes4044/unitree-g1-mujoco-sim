# Third-party notices

This repository is licensed under Apache-2.0; see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). This file lists the third-party material that is
**redistributed** with it — the vendored robot model and meshes, the vendored
walking policy — and the licences of the handful of Python packages it installs
at run time.

**Licence policy: permissive only** (Apache-2.0 / MIT / BSD / Zlib). There is no
copyleft component in this repository, and the one weak-copyleft item in the
dependency tree (Eigen, MPL-2.0, statically linked into two wheels) is disclosed
in §4. Nothing here is modified from upstream except as §2 and
[`models/ASSETS.md`](models/ASSETS.md) record.

There is deliberately no ffmpeg anywhere: `scripts/webp_clip.py` writes rollout
videos as animated WebP through Pillow (libwebp, BSD-3-Clause), because every
redistributable ffmpeg binary we could reach — Debian's `ffmpeg`, the
`imageio-ffmpeg` wheel — is a GPL build. `imageio-ffmpeg` is in no dependency
list.

## 1. What is redistributed in this repository

| Path | Contents | Licence |
|---|---|---|
| `models/g1_pick_place.xml`, `models/assets/` (122 STL meshes, ~20 MB) | the compiled G1 model and its meshes, **vendored** — committed, not fetched at setup, so one `git clone` or one zip is enough on an offline machine | BSD-3-Clause (Unitree) + Apache-2.0 (BrainCo) + Apache-2.0 (this project); per-file in [`models/ASSETS.md`](models/ASSETS.md) |
| `models/objects/`, `models/scenarios/`, `scenarios/` | scenario parts and composed scenes authored here; they reference `models/assets/` through `meshdir="../assets"` | Apache-2.0 (this project) |
| `src/g1/wbc/models/holosoma/fastsac_g1_29dof.onnx` (0.9 MB) | Amazon Holosoma's pretrained G1 locomotion policy, with its upstream `LICENSE` and `NOTICE` next to it | Apache-2.0 (§3) |
| `src/`, `scripts/` | this project's code | Apache-2.0 |

## 2. Robot model assets

Per source and per file: [`models/ASSETS.md`](models/ASSETS.md). In summary, the
vendored model is built from

- **MuJoCo Menagerie `unitree_g1`** at `8161bba264d7fa7c99ca301e91e7fb44737676ad`
  — Menagerie's model of Unitree Robotics' G1 description. BSD-3-Clause,
  Copyright (c) 2016-2023 Unitree Robotics. Provides the body links, joints,
  limits, inertials, body actuators, the `stand` keyframe and 49 body meshes.
- **BrainCo `revo2_description` 1.0.0** at `0ea3db7d92f91565ea6a9105ec49f1205a39f3f9`
  — Apache-2.0, BrainCo; ships no NOTICE file. Provides the Revo 2 five-finger
  hand links, joint origins/axes/limits/effort, inertials, `<mimic>` coupling
  ratios and 71 meshes.
- **Unitree `unitree_ros`** `robots/g1_description/meshes/` at
  `7d6075f7f58588b189b940130e3edab3c839b2df` — BSD-3-Clause, Copyright (c)
  2016-2022 Unitree Robotics. Provides the visual-only backpack and RealSense
  D455 meshes, and (as numeric constants in `scripts/build_scene.py`) the
  D435i / Mid-360 / head-module / D455 sensor poses.

The pin on `revo2_description` **1.0.0** rather than 1.1.x is deliberate: the
1.1.0 / 1.1.1 tags move the thumb metacarpal zero by 3 deg, add +9 / +6 deg
origin offsets on the thumb MCP/PIP, change the joint limits and change the mimic
ratios, which changes what a recorded hand joint angle *means*.
`build_scene.py` asserts the 1.0.0 contract. See the "Not used, must not ship"
section of [`models/ASSETS.md`](models/ASSETS.md), which also records why
`BrainCoTech/brainco-description` is **not** used (it carries no licence file and
is not cleared for redistribution).

Changes made to these third-party files — mesh decimation, URDF-to-MJCF
compilation, the root-link rename, the removal of the Dex3 hands — are listed in
[`models/ASSETS.md`](models/ASSETS.md).

### BSD-3-Clause (Unitree Robotics)

Applies to the Menagerie `unitree_g1` model (copyright years 2016-2023) and to
the `unitree_ros` meshes (2016-2022). Reproduced in full as the licence requires:

    Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The same text, with the 2016-2022 copyright line, applies to the `unitree_ros`
meshes.

### Apache-2.0 (BrainCo `revo2_description`)

Licensed under the Apache License, Version 2.0. The full text is this
repository's top-level [`LICENSE`](LICENSE) file, which must accompany any
redistribution of these assets. `revo2_description` ships no NOTICE file, so
there is no additional attribution notice to carry.

## 3. Walking policy — Amazon Holosoma (Apache-2.0, code and weights)

`src/g1/wbc/models/holosoma/fastsac_g1_29dof.onnx` is Amazon's Holosoma G1
locomotion policy from [`amazon-far/holosoma`](https://github.com/amazon-far/holosoma),
vendored byte-for-byte from pinned commit
`bccd4d7451640a2800ddc77e469d911a84f91994` (0.9 MB, sha256
`8346fd90778439395922a8c7256f24125ae84b8dea949128bac9e23c02bc7717`, recorded in
the `src/g1/wbc/holosoma.py` docstring).

Code **and** weights are Apache-2.0: the weights ship inside the same repository
with no separate model licence (added upstream in `9c238cf80f`). Upstream's
`LICENSE` (the Apache-2.0 text) and `NOTICE` ("Copyright Amazon.com, Inc. or its
affiliates. All Rights Reserved.") are vendored next to the weights, as the
licence requires. The policy is loaded and run by `g1.wbc.holosoma` through
`onnxruntime` (MIT). Nothing about it needs network access: the weights are
committed, so the controller works on a machine that has never been online.

## 4. Python dependencies

Everything in `pyproject.toml` is permissive. Licences read from each wheel's own
metadata and licence files:

| Package | Licence | Notes |
|---|---|---|
| `mujoco` 3.12.0 (pinned exactly) | Apache-2.0 | statically links **Eigen (MPL-2.0)**, built with `EIGEN_MPL2_ONLY` so Eigen's LGPL-2.1 parts are excluded — see the wheel's `LICENSES_THIRD_PARTY.md`, and §5 below |
| `numpy` | BSD-3-Clause (with 0BSD / MIT / Zlib / CC0 parts) | |
| `pillow` | MIT-CMU (the HPND-family PIL licence) | bundles libwebp (BSD-3-Clause, Google), libjpeg, libpng and freetype (FTL chosen over GPL-2). libwebp is what writes the rollout animations |
| `imageio` | BSD-2-Clause | depends only on numpy + Pillow and bundles **no** ffmpeg; its ffmpeg plugin is lazy and `imageio-ffmpeg` is not installed |
| `onnxruntime` | MIT | statically links **Eigen (MPL-2.0)** — see its `ThirdPartyNotices.txt`, and §5 below |

Transitively: mujoco pulls `glfw` (Zlib), `pyopengl` (BSD-3-Clause),
`absl-py` (Apache-2.0), `etils`, `fsspec`, `typing-extensions`, `zipp`;
onnxruntime pulls `flatbuffers` (Apache-2.0), `protobuf` (BSD-3-Clause) and
`packaging` (Apache-2.0 / BSD-2-Clause). All permissive. Worth a note:
`pyopengl` (BSD-3-Clause upstream) declares no licence in its wheel metadata and
ships no licence file, and `etils` / `flatbuffers` are Apache-2.0 by classifier
rather than by a bundled licence file.

## 5. Weak copyleft (MPL-2.0): Eigen inside two wheels

MPL-2.0 is *file-level* copyleft: distributing the component unmodified, and
merely linking or compiling it in, imposes only the obligation to make the source
of those files available and to keep the notices.

**Eigen** (https://gitlab.com/libeigen/eigen) is statically compiled into the
`mujoco` wheel (under `EIGEN_MPL2_ONLY`, which excludes Eigen's LGPL-2.1 parts)
and into the `onnxruntime` wheel. This repository does not modify Eigen and does
not redistribute either wheel — pip or `uv` fetches them from PyPI — but the
disclosure belongs here anyway, because an offline install (a wheelhouse copied
onto an air-gapped machine) does redistribute them. If a recipient asks for the
corresponding source, it is the unmodified upstream Eigen release named by each
wheel's own third-party notices.

No other copyleft component appears anywhere in this repository or its dependency
closure.
