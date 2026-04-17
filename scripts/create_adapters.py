#!/usr/bin/env python3
"""
Generate mock PEFT LoRA adapter checkpoints under adapters/ (same layout as setup.sh).

Requires: pip install torch transformers peft huggingface_hub

Examples:
  python3 scripts/create_adapters.py
  python3 scripts/create_adapters.py --out-dir ./adapters --model-dir ~/model_cache/Qwen2.5-0.5B-Instruct
  python3 scripts/create_adapters.py --download-model   # fetch base model if missing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Same names as scripts/setup.sh (keep in sync when you change the list)
ADAPTER_NAMES = [
    "crop_corn_disease", "crop_wheat_disease", "crop_soy_disease",
    "crop_tomato_disease", "crop_cotton_disease", "crop_general_health",
    "crop_rice_disease", "crop_barley_disease", "crop_oat_disease",
    "crop_potato_disease", "crop_berry_disease", "crop_grape_disease",
    "crop_citrus_disease", "crop_apple_disease", "crop_peach_disease",
    "crop_canola_disease", "crop_sunflower_disease", "crop_alfalfa_health",
    "crop_pasture_health", "crop_yield_pred_north", "crop_yield_pred_south",
    "crop_stress_heat", "crop_stress_drought", "crop_ndvi_zones",
    "crop_growth_stage", "crop_harvest_window",
    "pest_aphid", "pest_rootworm", "pest_spider_mite", "pest_caterpillar", "pest_general",
    "pest_grasshopper", "pest_weevil", "pest_thrips", "pest_whitefly", "pest_cutworm",
    "pest_borer", "pest_leafhopper", "pest_slug", "pest_snail", "pest_ant",
    "pest_bee_health", "pest_beneficial_count",
    "soil_nitrogen", "soil_phosphorus", "soil_moisture", "soil_ph",
    "soil_potassium", "soil_organic_matter", "soil_compaction",
    "soil_salinity", "soil_erosion_risk", "soil_temp_root",
    "soil_n_source", "soil_microbiome", "soil_carbon_estimate",
    "irrigation_zone_a", "irrigation_zone_b", "irrigation_zone_c", "irrigation_zone_d",
    "irrigation_zone_e", "irrigation_zone_f", "irrigation_sched_block1",
    "irrigation_sched_block2", "irrigation_drip_health", "irrigation_sprinkler_uniform",
    "irrigation_water_quality", "irrigation_pressure", "irrigation_flow_meter",
    "irrigation_leak_detect",
    "weather_forecast", "weather_frost_alert", "weather_humidity",
    "weather_wind_alert", "weather_hail_risk", "weather_precipitation",
    "weather_heat_index", "weather_dew_point", "weather_soil_temp",
    "weather_evapotranspiration",
    "equip_tractor", "equip_drone", "equip_sensor",
    "equip_harvester", "equip_planter", "equip_sprayer",
    "equip_baler", "equip_spreader", "equip_gps_guidance",
    "equip_yield_monitor", "equip_fuel_telemetry",
    "livestock_cattle_health", "livestock_poultry", "livestock_pasture_rotation",
    "dairy_milk_quality", "dairy_feed_ration", "grain_storage_temp",
    "grain_moisture_bin", "carbon_footprint_field", "nutrient_runoff_risk",
]


def _default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parent.parent
    out = repo_root / "adapters"
    home = Path.home()
    model = home / "model_cache" / "Qwen2.5-0.5B-Instruct"
    return out, model


def download_model(model_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_dir=str(model_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax*", "tf_*"],
    )


def create_adapters(
    out_dir: Path,
    model_dir: Path,
    names: list[str],
    *,
    force: bool,
) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM

    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for n in names:
        p = out_dir / n
        if p.is_dir() and not force:
            continue
        todo.append(n)

    if not todo:
        print(f"All {len(names)} adapters already exist under {out_dir} (use --force to regenerate).")
        return

    print(f"Creating {len(todo)} adapter(s) under {out_dir} ...")

    # One fresh base load per adapter avoids stacking LoRA wraps on one model object.
    for i, name in enumerate(todo):
        path = out_dir / name
        if path.exists():
            import shutil
            shutil.rmtree(path)

        print(f"  [{i + 1}/{len(todo)}] {name}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        peft_model = get_peft_model(model, cfg)
        peft_model.save_pretrained(str(path))
        del peft_model
        del model

    print("Done.")


def main() -> int:
    default_out, default_model = _default_paths()

    p = argparse.ArgumentParser(description="Create mock LoRA adapters (PEFT) for memLoRA-edge.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=f"Output directory (default: {default_out})",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=default_model,
        help=f"Qwen2.5 base model directory (default: {default_model})",
    )
    p.add_argument(
        "--download-model",
        action="store_true",
        help="Download Qwen2.5-0.5B-Instruct into --model-dir if missing.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing adapter directories.",
    )
    args = p.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not model_dir.is_dir():
        if args.download_model:
            print(f"Downloading base model to {model_dir} ...")
            download_model(model_dir)
        else:
            print(
                f"Model not found: {model_dir}\n"
                "  Run with --download-model, or download Qwen2.5-0.5B-Instruct there first.",
                file=sys.stderr,
            )
            return 1

    create_adapters(out_dir, model_dir, ADAPTER_NAMES, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
