"""CAMulator model, transform, forcing, and state initialization."""

from __future__ import annotations

import os
from typing import Optional

import torch
import xarray as xr
import yaml

from vercor.jax_logging import LoggerLike, get_default_logger
from vercor.setups.external import camulator_imports
from vercor.setups.external.camulator_stepper import CAMulatorStepper
from vercor.setups.external.camulator_tensors import _prepare_static_forcing_tensor


def add_init_noise(
    state: torch.Tensor,
    noise_std: float = 0.05,
    logger: LoggerLike | None = None,
) -> torch.Tensor:
    """Return CAMulator initial conditions with Gaussian perturbation."""

    log = logger if logger is not None else get_default_logger()
    log.info(f"Adding initial condition noise (std={noise_std})")
    noise = torch.randn_like(state) * noise_std
    return state + noise


def initialize_camulator(
    config_path: str,
    model_name: Optional[str] = None,
    device: str = "cuda",
    logger: LoggerLike | None = None,
) -> dict:
    """Initialize CAMulator model state and supporting runtime objects."""

    camulator_imports._load_credit_modules()

    log = logger if logger is not None else get_default_logger()
    log.info(f"Initializing CAMulator from config: {config_path}")

    with open(config_path) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = camulator_imports.credit_main_parser(
        conf,
        parse_training=False,
        parse_predict=True,
        print_summary=False,
    )
    conf["predict"]["mode"] = None

    current_device = torch.device(device)
    log.info(f"Using device: {current_device}")

    log.info("Loading transforms...")
    camulator_imports.load_transforms(conf)

    if conf["data"]["scaler_type"] == "std_new":
        state_transformer = camulator_imports.Normalize_ERA5_and_Forcing(conf)
    else:
        raise ValueError(f"Unsupported scaler_type: {conf['data']['scaler_type']}")

    log.info(
        f"Loading model: {model_name if model_name else 'checkpoint.pt (default)'}"
    )
    if model_name:
        model = camulator_imports.load_model_name(
            conf,
            model_name,
            load_weights=True,
        ).to(current_device)
    else:
        model = camulator_imports.load_model(conf, load_weights=True).to(current_device)

    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:
        log.info(f"Setting up distributed mode: {conf['predict']['mode']}")
        model = camulator_imports.distributed_model_wrapper(
            conf,
            model,
            current_device,
        )
        if conf["predict"]["mode"] == "fsdp":
            model = camulator_imports.load_model_state(conf, model, current_device)

    model.eval()
    log.info("Model loaded and set to eval mode")

    log.info("Loading initial conditions...")
    ic_path = conf["predict"]["init_cond_fast_climate"]
    if not os.path.exists(ic_path):
        raise FileNotFoundError(
            f"Initial condition file not found: {ic_path}\nPlease run Make_Climate_Initial_Conditions.py first."
        )

    initial_state = torch.load(ic_path, map_location=current_device).to(current_device)
    log.info(f"Initial state shape: {initial_state.shape}")

    log.info("Loading forcing data...")
    forcing_file = conf["predict"]["forcing_file"]
    if not os.path.exists(forcing_file):
        raise FileNotFoundError(f"Forcing file not found: {forcing_file}")

    chunk_size = conf["data"].get("forcing_chunk_size", 32)
    forcing_ds = xr.open_dataset(forcing_file, chunks={"time": chunk_size})

    log.info("Normalizing forcing data...")
    forcing_ds_norm = state_transformer.transform_dataset(forcing_ds)
    forcing_ds_norm = forcing_ds_norm.chunk({"time": chunk_size})

    log.info("Loading static forcing...")
    sf_vars = conf["data"]["static_variables"]
    static_forcing = _prepare_static_forcing_tensor(forcing_ds, sf_vars, current_device)
    log.info(f"Static forcing shape: {static_forcing.shape}")

    log.info("Loading metadata and coordinates...")
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])
    metadata = camulator_imports.load_metadata(conf)

    log.info("Creating CAMulatorStepper with conservation fixers...")
    stepper = CAMulatorStepper(model, conf, current_device)

    log.info("=" * 70)
    log.info("Initialization complete!")
    log.info(f"Model device: {current_device}")
    log.info(f"State shape: {initial_state.shape}")
    log.info(f"Static forcing: {len(sf_vars)} variables")
    log.info(
        f"Conservation fixers: Mass={stepper.flag_mass}, Water={stepper.flag_water}, Energy={stepper.flag_energy}"
    )
    log.info(f"Wind filtering: {stepper.enable_wind_filtering}")
    log.info("=" * 70)

    return {
        "model": model,
        "stepper": stepper,
        "conf": conf,
        "state_transformer": state_transformer,
        "forcing_dataset": forcing_ds_norm,
        "forcing_dataset_raw": forcing_ds.chunk({"time": chunk_size}),
        "static_forcing": static_forcing,
        "initial_state": initial_state,
        "latlons": latlons,
        "metadata": metadata,
        "device": current_device,
    }


__all__ = ["add_init_noise", "initialize_camulator"]
