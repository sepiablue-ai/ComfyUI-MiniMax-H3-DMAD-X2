"""Run the configured ComfyUI in the foreground, with process-local GPU settings."""
import argparse
import os
import socket
import subprocess
import sys
from common import compatible_json, load_config, model_paths, require, require_comfy_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8198)
    args = parser.parse_args()
    require(1 <= args.port <= 65535, "port must be between 1 and 65535")
    config = load_config(args.config)
    require_comfy_python(config)
    yaml = config["config_path"].parent / "extra_model_paths.local.yaml"
    require(yaml.is_file(), "Run setup to prepare extra_model_paths.local.yaml first")
    compatible_json(yaml, model_paths(config))
    with socket.socket() as connection:
        require(connection.connect_ex(("127.0.0.1", args.port)) != 0,
                "Port is already in use; inspect or reuse that server")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(config["gpu"]), PYTHONIOENCODING="utf-8",
                       PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_TELEMETRY="1")
    command = [sys.executable, "-B", "-s", "-X", "utf8", str(config["comfyui_root"] / "main.py"),
               "--listen", "127.0.0.1", "--port", str(args.port), "--disable-pinned-memory",
               "--disable-auto-launch", "--extra-model-paths-config", str(yaml)]
    print(f"ComfyUI: http://127.0.0.1:{args.port}", flush=True)
    try:
        return subprocess.call(command, cwd=config["comfyui_root"], env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
