# Validation

## Fresh new-user installation — 2026-10-06 to 2026-10-07 JST

The current `setup.bat` package passed an end-to-end check in a newly cloned ComfyUI, with a dedicated venv and private Python 3.13.13. All six source checkpoints were downloaded again from immutable Hugging Face revisions and checked against their distribution SHA-256 hashes. No existing weights, fused model, converted X2 decoder, shared model root or hard-linked source weight was used.

The documented CUDA PyTorch/torchvision bootstrap was followed, then the actual package `setup.bat` installed the remaining dependencies and the pinned X2 Stream node and created one new FL2VA fused checkpoint. Setup returned 0 and `existing_model_reused: false`. The total setup command took 440.244 sec including input/output hashing, dependency installation and fusion. Its GPU fusion stage took 320.822 sec, produced 20,970,380,008 bytes and reproduced the recorded SHA-256 `1992b560ef87fccd9fc46b8df6875ee2262e3cfc81a18c9004f5a204d92f54af`. All untouched tensor bytes and six representative layer reload checks passed.

The dependency's `examples/00_prepare_int8.json` graph then prepared a new X2 INT8 decoder from the freshly downloaded original VAE, producing 2,834,835,436 bytes in 9.625 sec. `setup.bat --check` returned 0 with `generation_ready: true` and no missing files. Its `existing_model_reused: true` refers to checking the checkpoint just created in this cycle.

Exactly one video per shipped API sample was generated, with the supplied graph unchanged and its GUI graph embedded as workflow metadata. There were no extra warmup videos.

| Shipped sample | Whole execution sec | Minutes and seconds | Full video/audio decode |
|---|---:|---|---|
| fl2va_case01_sla5 | 64.078 | 1m 04.078s | Passed |
| fl2va_case02_sla5 | 58.175 | 0m 58.175s | Passed |

Both outputs are H.264 MP4, 1088×1920, 124 frames at 24 fps, with 32 kHz stereo audio. Whole-execution time is history `execution_start` to `execution_success`; server startup, model download/fusion and media inspection are excluded. These are consecutive functional runs without model/cache unloading, not a repeated cold/warm performance comparison.

The case01 GUI file was opened in this new server through Ctrl+O, with no missing-node/model error shown. Both samples were executed through their unchanged API files. Case02 GUI import and GUI-export/API equality were not rechecked in this cycle. The historical GUI/API checks below remain separate evidence.

The package's tested executable files and workflow/assets matched the candidate exactly. ComfyUI and X2 tracked sources were not edited. The owned server was stopped after both videos and the queue was empty. Human picture/audio acceptance remains pending; passing technical checks establishes usability in this tested configuration.

A complete download-from-scratch setup retained 45.69 GB of source weights, a 20.97 GB fused checkpoint and a 2.83 GB X2 decoder, plus Python/dependencies/caches. Allow about **90 GB free** for this route; the 30 GB fusion guard is additional to already downloaded sources and installed dependencies. [Machine-readable fresh-install evidence](fresh-install.json).

## Completed generation and fusion checks — 2026-10-04 JST

A fresh ComfyUI 0.37.0 clone and Python 3.13.13 venv were used on Windows 11 / RTX 4070 12 GB / 48 GiB RAM. Existing weights and the converted X2 decoder were shared by path. The FL2VA-base results retained below used the prior DMAD package; this package retains their workflow contents and fusion functions, with a configuration-based command interface fixed to FL2VA.

Exactly one video per case was generated, without extra warmups. The first run includes first-generation loading. These two results are functional validation measurements, not steady-state warm benchmarks. Both use a first-frame image with the FL2VA base; both-endpoint conditioning was not tested.

| Case | Whole execution sec | Sampler node sec including loading | GPU observed GiB | Whole-PC RAM observed GiB |
|---|---:|---:|---:|---:|
| First-frame I2VA case01 | 67.478 | 40.956 | 11.22 | 20.27 |
| First-frame I2VA case02 | 60.576 | 39.087 | 11.19 | 20.25 |

Both videos passed full FFmpeg video/audio decoding: 124 frames, 24 fps, 32 kHz stereo and 1088×1920. Four model evaluations and sparse attention execution were logged. GPU/RAM values are maxima observed at roughly 30-second intervals, not exact allocation peaks or minimum requirements.

The fresh-environment FL2VA fusion matched the prior output by full-file size and SHA-256:

- FL2VA: `1992b560ef87fccd9fc46b8df6875ee2262e3cfc81a18c9004f5a204d92f54af`, 20,970,380,008 bytes; recorded conversion 766.2 sec excluding initial hashing.

Fusion retained native BF16 LoRA arithmetic, INT8 ConvRot layout and per-module rounding seeds. Six representative layer checks and all unchanged tensor bytes were verified. Prior CUDA allocation peaks were approximately 1.30 GiB. Other GPUs are not guaranteed byte-identical. [Machine-readable historical summary](validation.json).

The two FL2VA-base GUI/API pairs were included in the prior registered-node schema checks. A representative I2VA graph was opened in ComfyUI; its GUI-exported API inputs matched the distributed API graph.

## Latest classroom timing — 2026-10-05 JST

| Method / set | Whole-execution runs, sec | Median sec | Minutes and seconds |
|---|---|---:|---|
| MATLOW + SLA 5%, initial paired set | 65.105 / 62.935 / 62.502 | 62.935 | 1m 02.935s |
| DMAD + SLA 5%, same paired set | 61.210 / 60.782 / 63.260 | 61.210 | 1m 01.210s |
| MATLOW + SLA 5%, later remeasurement | 62.377 / 60.667 / 60.633 | 60.667 | 1m 00.667s |

Each run unloaded models and execution caches, used `--cache-none` and executed conditioning, sampling and saving. OS file caches and CUDA/Triton compilation caches remained; these are not measurements immediately after a PC reboot. Total time is ComfyUI history `execution_start` to `execution_success`, including input/loading, generation, decode and save, excluding server startup and media inspection.

These classroom runs used ComfyUI 0.38.0 and the standard environment. The package remains pinned to the independently fresh-install-validated 0.37.0 fusion implementation. The latest 60.667 / 61.210 sec values belong to different repetition sets; rankings changed between sets. They establish neither stable speed superiority nor cold timings for both supplied scenes.

## This package — 2026-10-06 JST

The configuration/setup interface and repository layout follow the PDMD X2 package. The scope is now fixed to the FL2VA base, one fused checkpoint and two GUI/API pairs. Unsupported legacy partition settings are rejected before preparation. Historical generation results above are not new runs of this revised setup interface. Checks for this revision are recorded in [package-check.json](package-check.json).

The initial structural/configuration checks preceded the fresh-install cycle reported at the top of this document. That new cycle used the actual revised setup interface, created its own weights and completed both shipped API examples.

Picture and sound quality require human review. Passing media decode, source hashes, structural checks and deterministic fusion checks does not establish perceptual superiority or official rank-128 reproduction.
