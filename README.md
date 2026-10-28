# One photo to a 3D model, locally on a base Mac mini

The scripts from my video **"I Was Wrong About Speculative Decoding (3x the Tokens)"**. In it I turn single photos into
textured 3D models with **TRELLIS.2** on a base **Mac mini M6 (32 GB)**, completely offline, then into game-ready
low-poly models. I also run the tests you asked for in the comments on my previous video.

- [`image-to-3d/`](image-to-3d): the image-to-3D workflow, its fixes, and the low-poly bake
- [`comment-tests/`](comment-tests): the speculative decoding (MTP) tests and the long coding-session memory test

---

## image-to-3d

These scripts sit on top of [trellis-mac](https://github.com/shivampkumar/trellis-mac), a community port of
Microsoft's [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) that swaps the NVIDIA-only parts for versions that run on
Apple's Metal. You also need [Blender](https://www.blender.org/) (I used 5.2 LTS) on your `PATH`.

### Setup

1. Clone trellis-mac and run its `setup.sh`.
2. **Check the build actually worked.** Mine said "done" while five Metal components had failed to build, and it still
   runs without them, just slower and with worse textures. Look for errors in the build logs. If you find them, make
   these two fixes:
   - PyTorch 2.14 headers need C++20. Change `-std=c++17` to `-std=c++20` in `deps/{mtlbvh,mtldiffrast,mtlmesh,mtlgemm}/setup.py`
     and `deps/trellis2-apple/o-voxel/setup.py`.
   - The Eigen library submodule was missing. Fetch it with `git -C deps/trellis2-apple submodule update --init --depth 1`.

   Then rebuild each component:
   ```sh
   MACOSX_DEPLOYMENT_TARGET=12.0 uv pip install --python .venv/bin/python --no-build-isolation deps/<name>
   .venv/bin/python patches/mps_compat.py
   ```
3. Copy everything in `image-to-3d/` into the trellis-mac folder.
4. Optional but recommended: `git apply patches/generate.py.diff`. It adds two things:
   - **MIT-licensed background removal.** It uses `--rembg ZhengPeng7/BiRefNet` instead of TRELLIS's default
     briaai/RMBG-2.0, which is non-commercial. It needs `einops`.
   - **Multi-view input.** `--extra-views` takes photos of the hidden side. It's experimental: views that disagree
     get averaged into mush.
5. For the low-poly UV step, any Python with `xatlas` and `numpy` will do (`pip install xatlas numpy`). Point
   `XATLAS_PYTHON` at it if that's not the default `python3`.

### Use

```sh
./generate.sh photo.jpg best          # best = 1024 cascade (~13 min), fast = 512 (~2 min)
./generate.sh photo.jpg best 10000 0.32   # low-poly triangle budget, longest side in metres
```

This writes:
- `outputs/<name>-<mode>.glb`: the high-poly model
- `game/<name>-<mode>/`: the low-poly FBX and GLB, plus baked BaseColor, Normal and MaskMap textures in Unity HDRP layout
- `renders/`: 4-view previews

Peak memory is about 26 GB, so close heavy apps first and don't run an LLM at the same time. The weights take about 16 GB
in `~/.cache/huggingface/hub`.

| Script | What it does |
|---|---|
| `generate.sh` | Photo → high-poly GLB → low-poly → previews, in one command |
| `lowpoly.py` | Blender: thicken TRELLIS's "triangle soup", voxel re-mesh into one shell, decimate, xatlas UVs, bake normal/AO/colour/roughness/metal maps. About 40 s |
| `strip_far_side.py` | Removes metal parts (e.g. buckles) that TRELLIS mirrored onto the hidden side of the object |
| `turntable.py` | Studio-lit 1080p turntable video of any GLB (`--wire` shows the triangles) |
| `render_views.py` | 2×2 preview grid of a GLB |
| `xatlas_uv.py` | UV unwrap helper called by `lowpoly.py` |

Every Blender script prints its usage at the top of the file.

### Licences to check before you ship anything

| Component | Licence |
|---|---|
| TRELLIS.2 and trellis-mac | MIT |
| DINOv3 (required image encoder) | Meta's DINOv3 licence, which has conditions; read it |
| briaai/RMBG-2.0 (TRELLIS's default background remover) | Non-commercial; the patch above swaps it out |

---

## comment-tests

These need the local model setup from [mac-mini-local-llm](https://github.com/suhasaraos/mac-mini-local-llm) (a
`.venv` and `models/`). Copy these files into that folder, then run `mkdir -p results`.

### Speculative decoding: is it accurate, and is it faster?

All tests use the same 10-question quiz at temperature 0.

```sh
# mlx-vlm with Qwen's MTP drafter (pip install mlx-vlm==0.7.10)
.venv/bin/python spec_quiz.py --model models/qwen3.8-27b-dwq3.5 --tag off
.venv/bin/python spec_quiz.py --model models/qwen3.8-27b-dwq3.5 --tag mtp --draft mlx-community/Qwen3.8-27B-MTP-bf16

# The same quiz against any OpenAI-compatible server
.venv/bin/python api_quiz.py --url http://127.0.0.1:8000/v1 --model NAME --tag NAME

# MTPLX and oMLX, each with MTP off and on. Install each tool in its own venv and download its model first.
MTPLX_MODEL=/path/to/Qwen3.8-27B-MTPLX-Optimized-Speed OMLX_MODEL=/path/to/Qwen3.8-27B-oQ4e-mtp ./run_variants.sh
```

My results, in tokens per second:

| Tool | MTP off | MTP on | Speed-up | Score (off → on) |
|---|---|---|---|---|
| mlx-vlm (3.5-bit DWQ) | 11.8 | 11.0 at best | 0.9× | 9/10 → 9/10 |
| **oMLX** 0.7.3rc1 (Jundot oQ4e) | 9.5 | **28.4** | **3.0×** | 9/10 → 9/10 |
| MTPLX 2.12.2 (Optimized Speed) | 7.6 | 23.6 | 3.1× | 9/10 → 8/10 |

Each tool uses its own 3.5 to 4-bit build of the model, so compare each row with itself.

### Memory during a long coding session

```sh
TRELLIS_MAC=~/trellis-mac .venv/bin/python coding_session.py --tag default
TRELLIS_MAC=~/trellis-mac .venv/bin/python coding_session.py --tag fixed --server-args "--prompt-cache-size 1"
```

The script feeds 10 real source files, one coding task each, until the context passes 56K tokens. It logs memory, swap
and memory pressure after every turn to `results/mem_<tag>.csv`. Point `TRELLIS_MAC` at any folder with the same
file layout, or edit `FILES` to use your own code.

- **Default settings:** the server crashed on turn 6.
- **With `--prompt-cache-size 1`:** all 10 turns completed, peaking at 30.1 of 32 GB with no extra swap.

## License

MIT
