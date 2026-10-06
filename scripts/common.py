"""Shared configuration and verification, without importing the GPU runtime."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "minimax_h3_fl2va_DMAD_full_avg39_bf16merge_int8_convrot.safetensors"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def recipe():
    return read_json(ROOT / "model_recipe.json")


def model_name():
    return MODEL_NAME


def configured_path(value, parent):
    require(isinstance(value, str) and value.strip(), "A nonempty path string is required")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else parent / candidate).resolve()


def load_config(path):
    path = Path(path).resolve(strict=True)
    data = read_json(path)
    require(isinstance(data, dict), "Configuration must be a JSON object")
    fields = {"comfyui_root", "model_kind", "base_model", "dmad_lora", "output_directory",
              "fusion_manifest", "model_roots", "gpu"}
    require(not (set(data) - fields), "Unknown configuration fields: " + ", ".join(sorted(set(data) - fields)))
    # Accept old FL2VA configs, but reject every other partition before any I/O.
    require(data.pop("model_kind", "fl2va") == "fl2va", "This repository only supports the FL2VA base")
    for field in ("comfyui_root", "base_model", "dmad_lora", "output_directory"):
        require(field in data, "Set " + field + " in the configuration")
        data[field] = configured_path(data[field], path.parent)
    roots = data.get("model_roots", [])
    require(isinstance(roots, list), "model_roots must be an array of models directories")
    data["model_roots"] = [configured_path(value, path.parent) for value in roots]
    gpu = data.get("gpu", 0)
    require(type(gpu) is int and gpu >= 0, "gpu must be a nonnegative physical GPU index")
    data.update(gpu=gpu, config_path=path)
    data["model_path"] = data["output_directory"] / model_name()
    manifest = data.get("fusion_manifest")
    require(manifest is None or isinstance(manifest, str), "fusion_manifest must be a path or null")
    data["fusion_manifest"] = (configured_path(manifest, path.parent) if manifest is not None
                               else data["output_directory"] / (model_name() + ".fusion.json"))
    protected = {data["base_model"], data["dmad_lora"]}
    require(data["model_path"] not in protected, "Fused output must differ from both source files")
    partial = data["model_path"].with_suffix(data["model_path"].suffix + ".incomplete")
    require(data["fusion_manifest"] not in protected | {data["model_path"], partial, path},
            "Fusion manifest must differ from the source, output, incomplete model and config files")
    return data


def require_comfy_python(config):
    core = config["comfyui_root"]
    require((core / "main.py").is_file() and (core / "comfy/lora.py").is_file(),
            "comfyui_root must contain main.py and comfy/")
    executable = Path(sys.executable).resolve()
    candidates = [folder / ".venv/Scripts/python.exe" for folder in (core, core.parent)]
    candidates += [core.parent / name / "python.exe" for name in ("python_embeded", "python_embedded")]
    embedded = executable.parent.name.lower() in ("python_embeded", "python_embedded")
    require(sys.prefix != sys.base_prefix or embedded, "Global/system Python is not allowed")
    require(any(candidate.is_file() and executable == candidate.resolve() for candidate in candidates),
            "Select the Python belonging to the configured ComfyUI venv or Windows Portable")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_header(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Incomplete safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        require(2 <= size <= 100_000_000, "Invalid safetensors header size")
        encoded = stream.read(size)
        require(len(encoded) == size, "Incomplete safetensors header")
    return json.loads(encoded), 8 + size


def free_bytes(path):
    ancestor = Path(path)
    while not ancestor.exists():
        parent = ancestor.parent
        require(parent != ancestor, "Destination volume is unavailable")
        ancestor = parent
    return shutil.disk_usage(ancestor).free


def verify_sources(config, full_hash=True, kernel=True):
    reference = recipe()
    sources = (("base_model", reference["base"]),
               ("dmad_lora", reference["lora"]))
    for field, expected in sources:
        source = config[field]
        require(source.is_file() and source.stat().st_size == expected["bytes"],
                "Missing or wrong-size " + field + ": " + str(source))
        if full_hash:
            require(sha256(source) == expected["sha256"],
                    "Wrong " + field + " SHA-256; select the documented unfused base and average-rank-39 LoRA")
    for relative, expected in reference["comfyui_source_sha256"].items():
        source = config["comfyui_root"] / relative
        require(source.is_file() and sha256(source) == expected,
                "Unsupported ComfyUI implementation: " + relative + "; use the documented revision")
    if kernel:
        spec = importlib.util.find_spec("comfy_kitchen")
        source = Path(spec.origin).parent / "tensor/int8.py" if spec and spec.origin else None
        require(source is not None and source.is_file() and sha256(source) == reference["int8_source_sha256"],
                "Comfy Kitchen INT8 implementation differs from the tested version")


def verify_model(config):
    model, manifest = config["model_path"], config["fusion_manifest"]
    require(model.is_file() and manifest.is_file(), "Both fused model and fusion manifest are required for reuse")
    data = read_json(manifest)
    reference = recipe()
    expected_base = reference["base"]["sha256"]
    require(data.get("mode") == "merge" and data.get("unchanged_tensor_bytes_verified") is True,
            "Manifest does not record a completed, verified DMAD fusion")
    require(data.get("base_sha256") == expected_base and data.get("lora_sha256") == reference["lora"]["sha256"],
            "Fusion manifest has the wrong base or LoRA provenance")
    require(data.get("strength") == 1.0 and data.get("merge_dtype") == "bfloat16"
            and data.get("changed_weight_count") == 208 and data.get("changed_scale_count") == 200,
            "Unsupported fusion method, strength or module count")
    require(data.get("output_bytes") == model.stat().st_size == reference["reference_output"]["bytes"],
            "Fused model size differs from its manifest or recorded layout")
    header, _ = load_header(model)
    meta = header.get("__metadata__", {})
    require(meta.get("dmad_premerged") == "true" and meta.get("dmad_base_sha256") == expected_base
            and meta.get("dmad_lora_sha256") == reference["lora"]["sha256"]
            and meta.get("dmad_strength") == "1.0" and meta.get("dmad_target_count") == "208"
            and meta.get("dmad_merge_dtype") == "bfloat16", "Fused model metadata differs from the recipe")
    require(sha256(model) == data.get("output_sha256"), "Fused model SHA-256 differs from its manifest")
    return data


def preflight_output(config):
    model, manifest = config["model_path"], config["fusion_manifest"]
    require(not os.path.lexists(model.with_suffix(model.suffix + ".incomplete")),
            "An incomplete fusion output exists; inspect it before retrying")
    if os.path.lexists(model) or os.path.lexists(manifest):
        verify_model(config)
        return True
    require(free_bytes(config["output_directory"]) >= 30 * 10**9,
            "Allow at least 30 GB free for one new fused checkpoint")
    return False


def model_paths(config):
    categories = ("diffusion_models", "loras", "text_encoders", "vae")
    data = {"dmad_x2_fused": {"diffusion_models": config["output_directory"].as_posix()},
            "dmad_x2_base": {"diffusion_models": config["base_model"].parent.as_posix()},
            "dmad_x2_lora": {"loras": config["dmad_lora"].parent.as_posix()}}
    for index, root in enumerate(config["model_roots"]):
        require(root.is_dir(), "Shared models directory is missing: " + str(root))
        data[f"dmad_x2_shared_{index}"] = {"base_path": root.as_posix(),
                                             **{category: category + "/" for category in categories}}
    return data


def compatible_json(path, value, write=False):
    path = Path(path)
    if os.path.lexists(path):
        require(path.is_file() and read_json(path) == value,
                "Existing configuration differs; refusing overwrite: " + str(path))
    elif write:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def generation_missing(config, include_inputs=True):
    files = {"text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
             "vae": ["minimax_h3_audio_vae_fp32.safetensors",
                     "h3_x2_stream/MiniMax-H3-X2-Detail-v1-decoder-int8-convrot.safetensors"]}
    files["vae"].append("minimax_h3_video_vae_int8_convrot.safetensors")
    roots = [config["comfyui_root"] / "models", *config["model_roots"]]
    missing = [f"{category}/{name}" for category, names in files.items() for name in names
               if not any((root / category / name).is_file() for root in roots)]
    if include_inputs:
        missing += ["input/dmad_" + source.name for source in (ROOT / "examples/assets").iterdir()
                    if source.is_file() and not (config["comfyui_root"] / "input" / ("dmad_" + source.name)).is_file()]
    return missing
