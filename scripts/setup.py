"""Install the selected ComfyUI environment and prepare or reuse its DMAD model."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import (ROOT, compatible_json, generation_missing, load_config, model_paths,
                    preflight_output, recipe, require, require_comfy_python, sha256,
                    verify_model, verify_sources)


def git_output(folder, *arguments):
    return subprocess.check_output(["git", "-C", str(folder), *arguments], text=True,
                                   encoding="utf-8", errors="strict").strip()


def check_x2(path):
    require(path.is_dir() and (path / "__init__.py").is_file() and (path / "requirements.txt").is_file(),
            "Missing or incomplete ComfyUI-H3-X2-Stream installation")
    require(git_output(path, "rev-parse", "HEAD") == recipe()["x2_stream_commit"],
            "X2 revision differs; keep existing files and select the documented installation")
    require(not git_output(path, "status", "--porcelain", "--untracked-files=no"),
            "Existing X2 files have local edits; refusing to replace them")


def install(config):
    protected = []
    for package in ("torch", "torchvision", "torchaudio"):
        try:
            protected.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            require(package == "torchaudio", "Install the documented CUDA PyTorch and torchvision first")
    require(metadata.version("torch") == "2.14.0+cu130", "Tested CUDA PyTorch is 2.14.0+cu130")
    require(metadata.version("torchvision") == "0.29.0+cu130", "Tested torchvision is 0.29.0+cu130")
    x2 = config["comfyui_root"] / "custom_nodes/ComfyUI-H3-X2-Stream"
    if not os.path.lexists(x2):
        require(shutil.which("git"), "Git is required to install X2 Stream")
        subprocess.run(["git", "clone", "--no-checkout", recipe()["sources"]["x2_stream"], str(x2)], check=True)
        subprocess.run(["git", "-C", str(x2), "checkout", "--detach", recipe()["x2_stream_commit"]], check=True)
    check_x2(x2)
    # Both the checked-in pins and installed CUDA package versions constrain the resolver.
    import tempfile
    scratch = ROOT / "tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="setup-", dir=scratch) as temporary:
        pins = Path(temporary) / "cuda-constraints.txt"
        pins.write_text("\n".join(protected) + "\n", encoding="utf-8")
        uv = shutil.which("uv")
        command = ([uv, "pip", "install", "--python", sys.executable] if uv else
                   [sys.executable, "-B", "-m", "pip", "install", "--no-cache-dir"])
        command += ["-c", str(ROOT / "constraints-windows.txt"), "-c", str(pins)]
        for requirements in (config["comfyui_root"] / "requirements.txt",
                             ROOT / "requirements-windows.txt", x2 / "requirements.txt"):
            command += ["-r", str(requirements)]
        subprocess.run(command, check=True)


def check_dependencies(config):
    for package, expected in (("torch", "2.14.0+cu130"), ("torchvision", "0.29.0+cu130"),
                              ("comfy-kitchen", "0.2.35"), ("comfy-aimdo", "0.5.5"),
                              ("av", "18.1.0"), ("safetensors", "0.8.0"), ("triton-windows", "3.8.0.post28")):
        require(metadata.version(package) == expected, "Unexpected " + package + " version; see tested environment")
    import av
    require("h264_nvenc" in av.codecs_available, "PyAV must support H.264 NVENC")
    check_x2(config["comfyui_root"] / "custom_nodes/ComfyUI-H3-X2-Stream")


def inputs(config, write=False):
    destination = config["comfyui_root"] / "input"
    for source in sorted((ROOT / "examples/assets").iterdir()):
        if not source.is_file():
            continue
        target = destination / ("dmad_" + source.name)
        if os.path.lexists(target):
            require(target.is_file() and sha256(target) == sha256(source),
                    "Different input contents already exist; refusing overwrite: " + str(target))
        elif write:
            destination.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true", help="Read-only verification; no installs, fusion or writes")
    args = parser.parse_args()
    config = load_config(args.config)
    require_comfy_python(config)
    verify_sources(config, full_hash=False, kernel=False)
    paths = model_paths(config)
    yaml = config["config_path"].parent / "extra_model_paths.local.yaml"
    compatible_json(yaml, paths)
    inputs(config)
    # Inspect full provenance and existing outputs before installation or a large write.
    verify_sources(config, kernel=False)
    reused = preflight_output(config)
    require(not args.check or reused, "Fused model is not prepared; run setup without --check")
    if not args.check:
        install(config)
    check_dependencies(config)
    verify_sources(config, full_hash=False)
    if not args.check and not reused:
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "scripts/fuse_dmad.py"),
                        "--config", str(config["config_path"])], check=True)
        verify_model(config)
    if not args.check:
        inputs(config, write=True)
        compatible_json(yaml, paths, write=True)
    missing = generation_missing(config)
    if args.check and not yaml.is_file():
        missing.append("extra_model_paths.local.yaml")
    print(json.dumps({"status": "check_passed" if args.check and not missing else "model_ready",
                      "existing_model_reused": reused, "model": str(config["model_path"]),
                      "fusion_manifest": str(config["fusion_manifest"]), "read_only": args.check,
                      "generation_started": False, "comfyui_started": False,
                      "generation_ready": not missing, "missing_generation_files": missing}, ensure_ascii=False, indent=2))
    require(not args.check or not missing, "Generation prerequisites are missing; see the reported files and README")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
