"""Fuse DMAD into the configured INT8 ConvRot base, one layer at a time.

--inspect validates input files without loading torch or writing reports.
--verify validates an existing output and manifest without GPU execution.
--check checks six representative layers on the configured CUDA GPU and
writes a separate check report, without creating a full model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from common import (load_config, preflight_output, require_comfy_python,
                    verify_model, verify_sources)


ROOT = Path(__file__).resolve().parents[1]
CORE = None
GPU_INDEX = 0
LORA_SHA = "ddfdd94f4c9a0f32bf3649bb8e98ca20db572e4db6bcd9cca546a5b33ea32d11"
CHUNK = 8 * 1024 * 1024
REPRESENTATIVES = (
    "blocks.0.attn.out_proj.weight",
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.mlp.fc1.weight",
    "blocks.0.mlp.fc2.weight",
    "token_refiner.blocks.0.attn.qkv_proj.weight",
    "token_refiner.blocks.0.mlp.fc1.weight",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_header(stream):
    stream.seek(0)
    prefix = stream.read(8)
    require(len(prefix) == 8, "Missing safetensors length")
    size = struct.unpack("<Q", prefix)[0]
    require(2 <= size <= 100_000_000, "Invalid safetensors header size")
    encoded = stream.read(size)
    require(len(encoded) == size, "Incomplete safetensors header")
    return json.loads(encoded), size + 8, prefix + encoded


def tensor_bytes(stream, header, start, key):
    lo, hi = header[key]["data_offsets"]
    stream.seek(start + lo)
    data = bytearray(hi - lo)
    require(stream.readinto(data) == len(data), f"Incomplete tensor: {key}")
    return data


def ordered_tensors(header):
    return sorted((k for k in header if k != "__metadata__"),
                  key=lambda k: header[k]["data_offsets"][0])


def validate_offsets(header, file_size, start):
    offset = 0
    for key in ordered_tensors(header):
        lo, hi = header[key]["data_offsets"]
        require(lo == offset and hi >= lo, f"Non-contiguous data offsets: {key}")
        offset = hi
    require(start + offset == file_size, "Header/payload size mismatch")


def inspect_sources(base, lora):
    with base.open("rb") as bf, lora.open("rb") as lf:
        bh, bstart, _ = read_header(bf)
        lh, lstart, _ = read_header(lf)
        validate_offsets(bh, base.stat().st_size, bstart)
        validate_offsets(lh, lora.stat().st_size, lstart)
        require(not any("dmad" in k.lower() or "premerged" in k.lower()
                        for k in bh.get("__metadata__", {})), "Base is already marked premerged")
        require(len(lh) - int("__metadata__" in lh) == 624, "Expected 624 DMAD tensors")
        require(lh.get("__metadata__", {}).get("converted_layout") == "comfyui_minimax_h3",
                "LoRA is not the converted ComfyUI layout")
        targets = {}
        consumed = set()
        quant_count = 0
        for key in lh:
            if not key.endswith(".lora_A.weight"):
                continue
            stem = key[:-len(".lora_A.weight")]
            require(stem.startswith("diffusion_model."), f"Unexpected LoRA prefix: {stem}")
            target = stem.removeprefix("diffusion_model.") + ".weight"
            bkey, akey = stem + ".lora_B.weight", stem + ".alpha"
            require(target in bh and bkey in lh and akey in lh, f"Missing target/triplet: {stem}")
            require(all(lh[k]["dtype"] == "BF16" for k in (key, bkey, akey)), "Unexpected LoRA dtype")
            rank, input_dim = lh[key]["shape"]
            output_dim, up_rank = lh[bkey]["shape"]
            require(rank == up_rank and [output_dim, input_dim] == bh[target]["shape"],
                    f"LoRA/base shape mismatch: {target}")
            require(lh[akey]["shape"] == [], f"Non-scalar alpha: {akey}")
            alpha = struct.unpack("<f", b"\x00\x00" + tensor_bytes(lf, lh, lstart, akey))[0]
            require(alpha / rank == 1.0, f"Unexpected converted alpha/rank: {stem}")
            qkey = target[:-len("weight")] + "comfy_quant"
            skey = target[:-len("weight")] + "weight_scale"
            if target.startswith("blocks."):
                require(bh[target]["dtype"] == "I8", f"Expected INT8 main weight: {target}")
                conf = json.loads(tensor_bytes(bf, bh, bstart, qkey))
                require(conf == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
                        f"Unexpected quantization: {target}")
                require(bh[skey]["dtype"] == "F32" and bh[skey]["shape"] == [output_dim, 1],
                        f"Unexpected per-row scale: {target}")
                quant_count += 1
            else:
                require(target.startswith("token_refiner.blocks.") and bh[target]["dtype"] == "BF16"
                        and qkey not in bh and skey not in bh, f"Unexpected refiner weight: {target}")
            targets[target] = (stem, key, bkey, akey)
            consumed.update((key, bkey, akey))
        require(len(targets) == 208 and quant_count == 200, "Expected 200 INT8 + 8 BF16 targets")
        require(consumed == set(lh) - {"__metadata__"}, "Unconsumed DMAD tensors")
    with lora.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(digest == LORA_SHA, "DMAD SHA256 mismatch")
    return bh, lh, targets, {
        "base": str(base), "base_bytes": base.stat().st_size,
        "lora": str(lora), "lora_sha256": digest, "targets": 208,
        "int8_convrot_targets": 200, "bf16_refiner_targets": 8,
        "all_alpha_div_rank": 1.0,
        "merge_dtype": "bfloat16", "strength": 1.0,
        "seed_policy": "comfy.utils.string_to_seed(full diffusion_model module path, without .weight)",
        "qkv_fc1_conversion": "already baked in Kijai LoRA; no further permutation",
    }


def prepare_runtime():
    # Keep process-local compiler/cache writes inside this repository.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(GPU_INDEX), PYTHONDONTWRITEBYTECODE="1",
                      PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1")
    for name, relative in {
        "HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
        "TRITON_CACHE_DIR": "cache/triton", "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
        "CUDA_CACHE_PATH": "cache/cuda", "XDG_CACHE_HOME": "cache",
        "TORCH_HOME": "cache/torch", "TEMP": "tmp", "TMP": "tmp",
    }.items():
        path = ROOT / ".cache" / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(CORE))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--disable-pinned-memory", "--disable-comfy-compiler"]
    import torch
    import comfy.ops
    import comfy.lora
    import comfy.utils
    from comfy.model_patcher import LowVramPatch
    from comfy.quant_ops import QuantizedTensor
    require(torch.cuda.device_count() == 1,
            "A single selected CUDA GPU is required")
    torch.cuda.reset_peak_memory_stats()
    return torch, comfy.ops, comfy.lora, comfy.utils, LowVramPatch, QuantizedTensor


def merge_layer(runtime, bf, bh, bs, lf, lh, ls, key, triplet, check):
    torch, ops, lora_api, utils, LowVramPatch, QuantizedTensor = runtime
    dtype_map = {"I8": torch.int8, "BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}

    def load(stream, header, start, name):
        return torch.frombuffer(tensor_bytes(stream, header, start, name),
                                dtype=dtype_map[header[name]["dtype"]]).reshape(header[name]["shape"])

    prefix = key[:-len("weight")]
    names = [key]
    quantized = bh[key]["dtype"] == "I8"
    if quantized:
        names.extend((prefix + "weight_scale", prefix + "comfy_quant"))
    state = {name[len(prefix):]: load(bf, bh, bs, name) for name in names}
    out_dim, in_dim = bh[key]["shape"]
    linear_cls = ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear
    layer = linear_cls(in_dim, out_dim, bias=False, device="cpu", dtype=torch.bfloat16)
    layer.load_state_dict(state, strict=True)
    del state
    require(isinstance(layer.weight, QuantizedTensor) == quantized, f"Layout load failed: {key}")
    original = layer.weight.to(device="cuda:0", dtype=torch.bfloat16)
    logical = layer.convert_weight(original)
    if not quantized:
        logical = logical.clone()
    stem, down_key, up_key, alpha_key = triplet
    lora_state = {k: load(lf, lh, ls, k) for k in (down_key, up_key, alpha_key)}
    full_key = "diffusion_model." + key
    adapters = lora_api.load_lora(lora_state, {stem: full_key}, log_missing=True)
    require(set(adapters) == {full_key}, f"Canonical LoRA loader did not match {key}")
    patches = {full_key: [(1.0, adapters[full_key], 1.0, None, None)]}
    merged = LowVramPatch(full_key, patches)(logical)
    require(bool(torch.isfinite(merged).all()), f"Non-finite merged weight: {key}")
    # DynamicVRAM uses the full module name as seed_key, not the weight name.
    seed = utils.string_to_seed(full_key.rsplit(".", 1)[0])
    result = layer.set_weight(merged, seed=seed, return_weight=True)
    require(isinstance(result, QuantizedTensor) == quantized, f"Quantization lost: {key}")
    if quantized:
        require(result._layout_cls == "TensorWiseINT8Layout" and result._params.convrot
                and result._params.convrot_groupsize == 256, f"ConvRot layout changed: {key}")
        require(bool(torch.isfinite(result._params.scale).all()) and bool((result._params.scale > 0).all()),
                f"Invalid output scales: {key}")
        payload = result.state_dict(key)
    else:
        require(result.dtype == torch.bfloat16, f"Refiner dtype changed: {key}")
        payload = {key: result}
    expected = {key, prefix + "weight_scale"} if quantized else {key}
    require(set(payload) == expected, f"Unexpected serialized keys: {key}")
    check_result = None
    if check:
        again = layer.set_weight(merged, seed=seed, return_weight=True)
        again_sd = again.state_dict(key) if quantized else {key: again}
        require(all(torch.equal(payload[k], again_sd[k]) for k in payload),
                f"Requantization was not reproducible: {key}")
        restored = linear_cls(in_dim, out_dim, bias=False, device="cpu", dtype=torch.bfloat16)
        reload_state = {k[len(prefix):]: v.to("cpu") for k, v in payload.items()}
        if quantized:
            reload_state["comfy_quant"] = load(bf, bh, bs, prefix + "comfy_quant")
        restored.load_state_dict(reload_state, strict=True)
        saved = restored.state_dict()
        require(all(torch.equal(saved[k[len(prefix):]], v.to("cpu")) for k, v in payload.items()),
                f"ComfyUI layer reload changed tensors: {key}")
        check_result = {"key": key, "requant_repeat_exact": True, "native_layout_reload_exact": True}
        del again, again_sd, restored, reload_state, saved
    cpu_payload = {k: v.detach().to("cpu").contiguous() for k, v in payload.items()}
    for name, value in cpu_payload.items():
        require(list(value.shape) == bh[name]["shape"] and value.dtype == dtype_map[bh[name]["dtype"]],
                f"Serialized dtype/shape changed: {name}")
        require(value.numel() * value.element_size() == bh[name]["data_offsets"][1] - bh[name]["data_offsets"][0],
                f"Serialized byte count changed: {name}")
    del original, logical, merged, result, payload, layer, adapters, patches, lora_state
    return cpu_payload, check_result


def encoded_header(header):
    value = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    value += b" " * (-len(value) % 8)
    return struct.pack("<Q", len(value)) + value


def copy_payload(source, destination, header, changed, base_hash):
    untouched = hashlib.sha256()
    for key in ordered_tensors(header):
        lo, hi = header[key]["data_offsets"]
        remaining = hi - lo
        while remaining:
            data = source.read(min(CHUNK, remaining))
            require(data, f"Unexpected end of base: {key}")
            destination.write(data)
            base_hash.update(data)
            if key not in changed:
                untouched.update(data)
            remaining -= len(data)
    return untouched.hexdigest()


def verify_output(path, base_header, targets, changed, untouched_expected, runtime):
    torch = runtime[0]
    digest, untouched = hashlib.sha256(), hashlib.sha256()
    with path.open("rb") as stream:
        header, start, raw = read_header(stream)
        require({k: v for k, v in header.items() if k != "__metadata__"} ==
                {k: v for k, v in base_header.items() if k != "__metadata__"},
                "Output tensor headers differ from the base")
        validate_offsets(header, path.stat().st_size, start)
        digest.update(raw)
        for key in ordered_tensors(header):
            lo, hi = header[key]["data_offsets"]
            remaining = hi - lo
            while remaining:
                data = stream.read(min(CHUNK, remaining))
                require(data, f"Unexpected end of output: {key}")
                digest.update(data)
                if key not in changed:
                    untouched.update(data)
                remaining -= len(data)
        require(untouched.hexdigest() == untouched_expected, "An untouched tensor changed")
        for key in targets:
            if header[key]["dtype"] == "I8":
                skey = key[:-len("weight")] + "weight_scale"
                scale = torch.frombuffer(tensor_bytes(stream, header, start, skey), dtype=torch.float32)
                require(bool(torch.isfinite(scale).all()) and bool((scale > 0).all()), f"Invalid stored scale: {skey}")
    return digest.hexdigest()


def write_report(path, report):
    # Exclusive creation also prevents overwriting a report created during the run.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")


def main():
    global CORE, GPU_INDEX
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    require_comfy_python(config)
    CORE, GPU_INDEX = config["comfyui_root"], config["gpu"]
    base, lora, output = config["base_model"], config["dmad_lora"], config["model_path"]
    print("Verifying input hashes and the native fusion implementation...", flush=True)
    verify_sources(config, kernel=not (args.inspect or args.verify))
    if args.verify:
        report = verify_model(config)
        print(json.dumps({"status": "verified", "output_sha256": report["output_sha256"],
                          "model_created": False, "gpu_used": False}, indent=2), flush=True)
        return
    bh, lh, targets, report = inspect_sources(base, lora)
    if args.inspect:
        print(json.dumps(report, indent=2), flush=True)
        return
    partial = output.with_suffix(output.suffix + ".incomplete")
    if not args.check and preflight_output(config):
        print(json.dumps({"status": "reused", "model_created": False, "output": str(output)}, indent=2), flush=True)
        return
    report_path = config["fusion_manifest"]
    if args.check:
        report_path = report_path.with_name(report_path.name + ".check.json")
    require(not os.path.lexists(report_path), "Report already exists; refusing overwrite: " + str(report_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = prepare_runtime()
    torch = runtime[0]
    started = time.perf_counter()
    checks = []
    with torch.inference_mode(), base.open("rb") as bf, lora.open("rb") as lf:
        _, bs, base_raw_header = read_header(bf)
        _, ls, _ = read_header(lf)
        if args.check:
            for key in REPRESENTATIVES:
                payload, checked = merge_layer(runtime, bf, bh, bs, lf, lh, ls, key, targets[key], True)
                checks.append(checked)
                del payload
                torch.cuda.empty_cache()
            report.update(mode="check", checks=checks, elapsed_sec=time.perf_counter() - started,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            write_report(report_path, report)
            print(json.dumps(report, indent=2), flush=True)
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        changed = set(targets)
        changed.update(key[:-len("weight")] + "weight_scale" for key in targets if bh[key]["dtype"] == "I8")
        oh = dict(bh)
        oh["__metadata__"] = dict(bh.get("__metadata__", {}),
            dmad_premerged="true", dmad_lora_sha256=LORA_SHA, dmad_strength="1.0",
            dmad_base_sha256="0" * 64, dmad_merge_dtype="bfloat16",
            dmad_seed_policy="DynamicVRAM full module path without .weight",
            dmad_target_count="208", dmad_no_additional_lora="true")
        out_header = encoded_header(oh)
        created_partial = False
        try:
            with partial.open("xb+") as destination:
                created_partial = True
                destination.write(out_header)
                bf.seek(bs)
                source_hash = hashlib.sha256(base_raw_header)
                untouched = copy_payload(bf, destination, bh, changed, source_hash)
                report["base_sha256"] = source_hash.hexdigest()
                oh["__metadata__"]["dmad_base_sha256"] = report["base_sha256"]
                final_header = encoded_header(oh)
                require(len(final_header) == len(out_header), "Metadata changed reserved header length")
                destination.seek(0)
                destination.write(final_header)
                for number, (key, triplet) in enumerate(targets.items(), 1):
                    payload, checked = merge_layer(runtime, bf, bh, bs, lf, lh, ls, key, triplet, key in REPRESENTATIVES)
                    if checked:
                        checks.append(checked)
                    for name, value in payload.items():
                        destination.seek(len(out_header) + bh[name]["data_offsets"][0])
                        raw = memoryview(value.view(torch.uint8).numpy()).cast("B")
                        require(destination.write(raw) == len(raw), f"Short write: {name}")
                        del raw
                    del payload, value
                    if number % 16 == 0 or number == len(targets):
                        print(f"Merged {number}/208", flush=True)
                    torch.cuda.empty_cache()
                destination.flush()
                os.fsync(destination.fileno())
            report["output_sha256"] = verify_output(partial, bh, targets, changed, untouched, runtime)
            require(not output.exists(), "Output appeared during conversion; refusing overwrite")
            partial.rename(output)
        except BaseException:
            if created_partial and partial.exists():
                partial.unlink()
            raise
    report.update(mode="merge", output=str(output), output_bytes=output.stat().st_size,
                  changed_weight_count=208, changed_scale_count=200, unchanged_tensor_bytes_verified=True,
                  checks=checks, elapsed_sec=time.perf_counter() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    write_report(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
