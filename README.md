# ComfyUI MiniMax H3 DMAD X2

[日本語](README_ja.md) | English

Generate MiniMax H3 first-frame video and audio using the **FL2VA base**, a fused **DMAD four-step** model, **SLA keep 5%**, and **X2 VAE INT8 streaming**. The two examples generate at **544 × 960** and save **1088 × 1920**, **124 frames at 24 fps**.

This package includes installation, GPU model fusion and ready-to-load workflows. Keep the repository outside `ComfyUI/custom_nodes`. Setup installs the separate [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) dependency into the selected ComfyUI. Its configuration and directory layout follow [ComfyUI-MiniMax-H3-PDMD-X2](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-PDMD-X2).

**Scope:** this package prepares only the FL2VA-base fused model and includes two first-frame I2VA examples. Both-endpoint FL2VA has not been tested. Upstream DMAD was trained for text-to-audio-video, so this conditioned use is experimental. The average-rank-39 conversion, INT8 fusion, lower sampling resolution, X2 and SLA combination has its own validation; upstream rank-128 quality claims do not automatically apply.

## Included files

- `setup.bat` and `scripts/`: dependency installation, sample inputs, configured CUDA fusion and verified model reuse.
- `config.example.json`: paths, GPU and optional shared models directories; the FL2VA base is fixed.
- `model_recipe.json`: accepted source sizes/hashes, native implementation hashes and sampling settings.
- `workflows/` and `workflows/api/`: two GUI/API pairs using the FL2VA fused model and a first-frame image.
- `examples/assets/`: the small input images used by the examples.
- `docs/validation.md`: fresh-install generation results, historical timing context and checks of this package.
- `LICENSE` and `NOTICE.md`: code license, attribution and model terms.

## Requirements

- Windows and ComfyUI **0.37.0**, commit `1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7`. Preparation verifies the native fusion and sampling source hashes. A different core is rejected; this package does not update or patch it.
- The Python belonging to that ComfyUI's venv or Windows Portable installation. Tested: Python 3.13.13, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, comfy-kitchen 0.2.35, comfy-aimdo 0.5.5, PyAV 18.1.0, safetensors 0.8.0 and triton-windows 3.8.0.post28. Complete pins are in `constraints-windows.txt`.
- An NVIDIA CUDA GPU and H.264 NVENC support. Generation and fusion were tested on RTX 4070 12 GB with 48 GiB system RAM. This RAM capacity is a tested configuration, not a proven minimum.
- Git for installing X2 Stream at `eb148a3d0f91a273668bbe027bfcbafb85f2c5e8`. Existing different or edited X2 installations are preserved and rejected.
- **30 GB free** for a new fused checkpoint, in addition to the source models. Setup creates one FL2VA checkpoint, approximately **20.97 GB / 19.53 GiB**. Reusing a verified existing file needs no additional checkpoint. For a complete new environment with all six source weights downloaded, budget about **90 GB free**, including sources, both prepared models and dependencies/caches.

Fusion uses the configured CUDA GPU, one layer at a time. The prior observed PyTorch allocated fusion peak was about 1.30 GiB; this is not total device memory. **Fusion is not CPU-only**, unlike the PDMD package. Stop generation on the selected GPU before fusion.

For an isolated installation, prepare the tested core and CUDA environment first:

```powershell
git clone https://github.com/Comfy-Org/ComfyUI.git D:\ComfyUI-DMAD
git -C D:\ComfyUI-DMAD checkout --detach 1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7
py -3.13 -m venv D:\ComfyUI-DMAD\.venv
& D:\ComfyUI-DMAD\.venv\Scripts\python.exe -m pip install --no-cache-dir torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

If Python 3.13 is not registered with `py` and you already have `uv`, use this isolated alternative after cloning/checking out the core. It keeps the base Python inside the ComfyUI directory without installing user-wide executable shims or registry entries:

```powershell
uv python install 3.13.13 --install-dir D:\ComfyUI-DMAD\python --no-bin --no-registry --no-cache
uv venv --seed --python D:\ComfyUI-DMAD\python\cpython-3.13.13-windows-x86_64-none\python.exe D:\ComfyUI-DMAD\.venv
uv pip install --python D:\ComfyUI-DMAD\.venv\Scripts\python.exe --no-cache torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Setup prefers `uv` when available and otherwise uses this selected Python's `pip`. It protects installed torch/torchvision/torchaudio versions and applies the recorded dependency pins. System/global Python is rejected. It does not download model weights, start ComfyUI or generate a video.

## Models

Provide the weights under their respective terms. No weights are bundled.

| File | ComfyUI folder | Role |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models/` | Unfused FL2VA base; [Comfy-Org H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| `minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors` | `loras/` | Accepted ~367 MB [Kijai conversion](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/loras), strength 1.0 |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders/` | [Comfy-Org H3 text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `vae/` | Ordinary H3 VAE for image conditioning |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/` | Audio decode; [Comfy-Org VAEs](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `MiniMax-H3-X2-Detail-v1.safetensors` | `vae/` | Original [X2 Detail VAE](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE) for decoder preparation |
| `MiniMax-H3-X2-Detail-v1-decoder-int8-convrot.safetensors` | `vae/h3_x2_stream/` | Prepared X2 INT8 decoder, ~2.83 GB |

An already converted X2 decoder can be shared by path. Otherwise, after setup installs X2 Stream, use its `examples/00_prepare_int8.json` workflow to prepare the decoder once. The supplied generation workflows load that prepared decoder directly. **Keep the ordinary H3 VAE on image-conditioning branches and X2 on the final video decode branch.**

The upstream project now supplies [official ComfyUI nodes and rank-128 LoRAs](https://github.com/Yzmblog/DMAD/blob/main/comfyui/ComfyUI-DMAD/README.md). This package uses the previously validated average-rank-39 file and native ComfyUI LCM. Its fusion utility deliberately rejects other LoRAs; the official rank-128 files are not interchangeable inputs.

## Prepare the DMAD model

Clone this package outside `ComfyUI/custom_nodes` and run the following setup commands from its directory:

```powershell
git clone https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-DMAD-X2.git D:\H3-DMAD-X2
Set-Location D:\H3-DMAD-X2
```

1. Copy `config.example.json` to `config.local.json`.
2. Set `comfyui_root`, `base_model` and `dmad_lora`. The base must be the documented unfused FL2VA file. Choose `output_directory` and `gpu` (physical GPU index, default 0). Paths relative to the configuration resolve from its directory. Optional `model_roots` entries are **models directories** containing `vae`, `text_encoders`, etc.
3. Run using that ComfyUI's Python:

```bat
setup.bat "D:\ComfyUI-DMAD\.venv\Scripts\python.exe" "config.local.json"
setup.bat "D:\ComfyUI-DMAD\.venv\Scripts\python.exe" "config.local.json" --check
```

Normal setup installs dependencies, then reuses a model only if its manifest, metadata and full-file hash pass verification. Otherwise it creates `minimax_h3_fl2va_DMAD_full_avg39_bf16merge_int8_convrot.safetensors`. It applies the LoRA to 200 INT8 ConvRot and eight BF16 weights, uses the native module-path rounding seeds, verifies untouched tensor bytes and checks six representative layer reloads. Source models are preserved.

The default manifest is `<output model filename>.fusion.json` beside the model. **To reuse a previously fused FL2VA model**, set `output_directory` to its current directory and `fusion_manifest` to its existing `merge_result.json`; original DMAD FL2VA merge reports are accepted. Use the documented FL2VA output filename. This avoids copying or rebuilding the ~21 GB file. Unrelated outputs, incomplete files and missing/mismatched manifests cause an error.

There is no model-partition selector in the example config. Legacy configs specifying `model_kind: fl2va` are accepted; other partition values are rejected before installation or fusion. A wrong base file is also rejected by its size/hash.

Setup writes `extra_model_paths.local.yaml` beside the config and copies inputs into `ComfyUI/input/` with the `dmad_` prefix. Existing different settings or inputs are preserved and rejected. Normal setup reports any missing generation files; prepare them before running `--check`. The check is read-only and requires the model, dependencies, inputs and generated paths config to be ready.

For direct fusion or inspection:

```powershell
& 'D:\ComfyUI-DMAD\.venv\Scripts\python.exe' -B -X utf8 scripts/fuse_dmad.py --config config.local.json
& 'D:\ComfyUI-DMAD\.venv\Scripts\python.exe' -B -X utf8 scripts/fuse_dmad.py --config config.local.json --inspect
& 'D:\ComfyUI-DMAD\.venv\Scripts\python.exe' -B -X utf8 scripts/fuse_dmad.py --config config.local.json --verify
```

`--inspect` checks the source files and native implementation without GPU execution or writes. `--verify` additionally verifies an existing fused model and manifest. `--check` on **the fusion script** is a different operation: it runs six CUDA layer checks and writes `<fusion manifest>.check.json`, without building a full checkpoint. Reports are never overwritten.

## Generate in ComfyUI

Start through your usual launcher with `--extra-model-paths-config` pointing to the generated YAML, or use this foreground helper:

```powershell
& 'D:\ComfyUI-DMAD\.venv\Scripts\python.exe' -B -X utf8 scripts/launch.py --config config.local.json --port 8198
```

The helper sets the configured GPU for this process, binds localhost, uses `--disable-pinned-memory`, refuses an occupied port and stops with Ctrl+C.

| GUI workflow | Mode |
|---|---|
| `workflows/fl2va_case01_sla5.json` | Classroom dialogue, first-frame I2VA with the FL2VA fused model, seed 43 |
| `workflows/fl2va_case02_sla5.json` | Sword scene, first-frame I2VA with the same fused model, seed 43 |

Drag a workflow onto ComfyUI and check model selectors. API graphs are under `workflows/api/`. When copying images manually, prefix their filenames with `dmad_` and place them directly under `ComfyUI/input/`. Outputs use `DMAD/` under the configured ComfyUI output directory.

Preset: **LCM**, the four-step manual sigmas recorded in `model_recipe.json`, **video/audio shifts 12/2**, BasicGuider, SLA keep 5% and FFN chunks 8. The native LCM update re-noises the denoised prediction; H3 maps audio's separate schedule. This matches the DMAD update rule, but does not reproduce upstream's exact noise-generator ordering or rank-128 samples. **Do not apply DMAD again to a fused DMAD model, or fuse it into another already distilled checkpoint such as MATLOW.**

## Validation and timing

The current `setup.bat` passed a fresh-install check on 2026-10-06/07: new ComfyUI, dedicated venv, all six source weights downloaded again, one newly fused FL2VA model and a newly prepared X2 decoder. Both unchanged API examples generated videos and passed full video/audio decoding, in **64.078 sec (1m 04.078s)** and **58.175 sec (0m 58.175s)**. These consecutive functional runs are not cold benchmark repeats. See [validation details](docs/validation.md) and [fresh-install evidence](docs/fresh-install.json).

The latest classroom whole-execution medians were **60.667 seconds (1m 00.667s)** for MATLOW + SLA 5% and **61.210 seconds (1m 01.210s)** for DMAD + SLA 5%. Each median uses three model/cache-unloaded runs, with OS and compilation caches retained; server startup is excluded. These came from different measurement sets on ComfyUI 0.38.0. The small difference does not establish stable speed superiority and is not a timing result for both example scenes. Picture and sound quality remain for human review.

## Troubleshooting

- Wrong source hash or core implementation: select the documented unfused base, average-rank-39 LoRA and tested core. Preparation does not silently substitute another checkpoint or patch the core.
- Existing fused model without a manifest: locate its original merge report and set `fusion_manifest`. Do not create a second large model just to rename it.
- Missing X2 decoder: prepare it once through X2 Stream's preparation workflow, or add its existing models directory to `model_roots`.
- Missing model selector: load the generated paths YAML when starting ComfyUI and restart after changing paths.
- Insufficient disk space: one new fused checkpoint needs 30 GB free; verified reuse avoids this allocation.
- Input/config conflict or interrupted fusion: inspect the named file before retrying. Existing different files and reports are not overwritten.
- Audio or visual quality: review the generated clips. Technical decode checks and upstream claims do not establish this combination's perceptual quality.

## License

GPL-3.0 code and workflow adaptations. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Model weights remain subject to their respective terms.
