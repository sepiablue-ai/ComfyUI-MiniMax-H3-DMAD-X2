# Attribution and scope

- DMAD: https://github.com/Yzmblog/DMAD . The upstream student was trained for T2VA. This package targets the FL2VA base only, with experimental first-frame I2VA examples. Official ComfyUI nodes and rank-128 converted LoRAs now exist upstream; this package uses the previously tested average-rank-39 conversion and native LCM schedule.
- ComfyUI: https://github.com/Comfy-Org/ComfyUI (GPL-3.0). Fusion uses native LoRA, ConvRot, mixed-precision and module-path rounding APIs. Sampling uses native LCM and H3 audio/video schedules. The core is not patched.
- Kijai's average-rank-39 ComfyUI LoRA: https://huggingface.co/Kijai/MiniMax-H3-experimental . Ranks vary across modules. No extra QKV/FFN permutation is applied during fusion.
- ComfyUI-H3-X2-Stream: https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream (GPL-3.0), including its KJNodes FFN attribution. It is installed separately at the documented revision, not vendored.
- MiniMax H3 model files: https://huggingface.co/Comfy-Org/MiniMax-H3 . X2 Detail VAE: https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE . Original, converted and fused weights remain subject to their respective terms; the code license does not relicense them.
- The synthetic Sayaka images and local case prompts are supplied to reproduce these examples. They are not official DMAD benchmark data or additional training data.
- Configuration and package layout follow https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-PDMD-X2 . DMAD's CUDA/BF16 fusion implementation is retained from the locally validated DMAD package; PDMD's CPU fusion was not substituted.

Code and workflow adaptations are GPL-3.0; see LICENSE. No model binaries, generated videos, ComfyUI checkout or Python environment are distributed.
