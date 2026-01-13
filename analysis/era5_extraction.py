#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract selected ERA5-Land bands from multi-band GeoTIFFs and compute derived indicators (GDAL).

Input:
  - A folder containing multiple multi-band GeoTIFFs.
  - Each GeoTIFF stores ERA5 variables in the EXACT order given by the user (see ERA5_BAND_ORDER).
  - Example filename: ERA5_complement_2000_01_26.tif

Output:
  - One GeoTIFF per input file, containing:
      (1) selected raw bands (KEEP_BANDS)
      (2) derived band(s) (DERIVED) - here: vpd_kpa only
  - Output preserves geotransform/projection/nodata and uses block-wise processing.

Notes:
  - VPD is computed in kPa from T2m and Td2m using Tetens formula:
      VPD = es(T) - es(Td)
      es(T) = 0.6108 * exp(17.27*T / (T + 237.3))
  - Temperature unit auto-detection: if median > 100, assume Kelvin and convert to Celsius.
  - This script is aligned with your land-cover-change study plan:
      Keep moisture availability, atmospheric dryness, energy constraint, and temperature.
      Wind-related variables are not used.
"""

import os
import argparse
from typing import Dict, List, Optional

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


# ---------------------------------------------------------------------
# 1) Band order mapping (1-based indices) - EXACTLY as user provided
# ---------------------------------------------------------------------
ERA5_BAND_ORDER = [
    'temperature_2m',
    'u_component_of_wind_10m',
    'v_component_of_wind_10m',
    'snow_cover',
    'total_precipitation_sum',
    'surface_latent_heat_flux_sum',
    'dewpoint_temperature_2m',
    'surface_pressure',
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4',
    'temperature_2m_max',
    'skin_temperature_max',
    'potential_evaporation_sum',
    'total_evaporation_sum',
    'skin_reservoir_content',
    'surface_net_solar_radiation_sum',
]
BAND_INDEX: Dict[str, int] = {name: i + 1 for i, name in enumerate(ERA5_BAND_ORDER)}  # 1-based


# ---------------------------------------------------------------------
# 2) Defaults aligned with your plan:
#    moisture availability + atmospheric dryness + energy constraint + temperature
# ---------------------------------------------------------------------
KEEP_BANDS_DEFAULT = [
    # Temperature state / extremes
    "temperature_2m",
    "temperature_2m_max",

    # Moisture availability
    "total_precipitation_sum",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",

    # Energy constraint (choose one primary; here we keep potential evaporation)
    "potential_evaporation_sum",

    # Optional but often useful for high-latitude / mountain ecozones
    "snow_cover",
]

DERIVED_DEFAULT = [
    "vpd_kpa",  # only derived indicator required here
]


# ---------------------------------------------------------------------
# 3) Helpers
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def list_tifs(input_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(input_dir):
        if fn.lower().endswith((".tif", ".tiff")):
            files.append(os.path.join(input_dir, fn))
    files.sort()
    return files


def parse_list_arg(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def guess_temp_unit_to_celsius(arr: np.ndarray) -> np.ndarray:
    """
    ERA5 temperatures are usually Kelvin. Heuristic conversion to Celsius:
      if median > 100 => Kelvin
    """
    out = arr.astype(np.float32, copy=False)
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return out
    if np.median(finite) > 100.0:
        return (out - 273.15).astype(np.float32)
    return out.astype(np.float32)


def saturation_vapor_pressure_kpa(temp_c: np.ndarray) -> np.ndarray:
    """
    Tetens formula (kPa):
      es = 0.6108 * exp(17.27*T / (T + 237.3))
    """
    temp_c = temp_c.astype(np.float32, copy=False)
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def compute_vpd_kpa(t2m: np.ndarray, td2m: np.ndarray) -> np.ndarray:
    """
    VPD (kPa) = es(T) - es(Td); clamp to >= 0.
    """
    t_c = guess_temp_unit_to_celsius(t2m)
    td_c = guess_temp_unit_to_celsius(td2m)
    es = saturation_vapor_pressure_kpa(t_c)
    ea = saturation_vapor_pressure_kpa(td_c)
    vpd = es - ea
    vpd = np.maximum(vpd, 0.0).astype(np.float32)
    return vpd


def get_nodata_value(ds: gdal.Dataset) -> Optional[float]:
    b1 = ds.GetRasterBand(1)
    return b1.GetNoDataValue()


def read_block(ds: gdal.Dataset, band_idx: int, xoff: int, yoff: int, xsize: int, ysize: int) -> np.ndarray:
    band = ds.GetRasterBand(band_idx)
    return band.ReadAsArray(xoff, yoff, xsize, ysize)


def write_block(out_ds: gdal.Dataset, out_band_idx: int, arr: np.ndarray, xoff: int, yoff: int) -> None:
    out_band = out_ds.GetRasterBand(out_band_idx)
    out_band.WriteArray(arr, xoff, yoff)


def create_out_dataset(
    ref_ds: gdal.Dataset,
    out_path: str,
    band_names: List[str],
    dtype=gdal.GDT_Float32,
    nodata: Optional[float] = None,
    compress: str = "LZW",
) -> gdal.Dataset:
    driver = gdal.GetDriverByName("GTiff")
    xsize = ref_ds.RasterXSize
    ysize = ref_ds.RasterYSize
    nbands = len(band_names)

    creation_options = [
        f"COMPRESS={compress}",
        "TILED=YES",
        "BIGTIFF=IF_SAFER",
        "PREDICTOR=2",
    ]

    out_ds = driver.Create(out_path, xsize, ysize, nbands, dtype, options=creation_options)
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())

    for i, name in enumerate(band_names, start=1):
        b = out_ds.GetRasterBand(i)
        if nodata is not None:
            b.SetNoDataValue(nodata)
        b.SetDescription(name)

    out_ds.FlushCache()
    return out_ds


# ---------------------------------------------------------------------
# 4) Processing per file (block-wise)
# ---------------------------------------------------------------------
def process_one_file(
    in_path: str,
    out_path: str,
    keep_bands: List[str],
    derived: List[str],
) -> None:
    ds = gdal.Open(in_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open: {in_path}")

    # validate keep bands
    for b in keep_bands:
        if b not in BAND_INDEX:
            raise ValueError(f"Unknown band name in KEEP_BANDS: {b}")

    # validate derived
    for d in derived:
        if d not in ("vpd_kpa",):
            raise ValueError(f"Unknown derived variable: {d}. Supported: vpd_kpa")

    # Determine needed source bands for this run
    needed = set(keep_bands)
    if "vpd_kpa" in derived:
        needed.update(["temperature_2m", "dewpoint_temperature_2m"])

    # Output band order: kept first, then derived
    out_band_names = list(keep_bands) + list(derived)
    out_index = {name: i + 1 for i, name in enumerate(out_band_names)}

    nodata = get_nodata_value(ds)
    out_ds = create_out_dataset(
        ref_ds=ds,
        out_path=out_path,
        band_names=out_band_names,
        dtype=gdal.GDT_Float32,
        nodata=nodata,
        compress="LZW",
    )

    # Determine block size
    band1 = ds.GetRasterBand(1)
    blk_x, blk_y = band1.GetBlockSize()
    if blk_x <= 0 or blk_y <= 0:
        blk_x, blk_y = 512, 512

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    for yoff in range(0, ysize, blk_y):
        ywin = min(blk_y, ysize - yoff)
        for xoff in range(0, xsize, blk_x):
            xwin = min(blk_x, xsize - xoff)

            # Read required inputs
            src: Dict[str, np.ndarray] = {}
            for name in needed:
                src[name] = read_block(ds, BAND_INDEX[name], xoff, yoff, xwin, ywin)

            # Nodata mask (propagate)
            if nodata is not None:
                # Use one stable band for mask; temperature_2m usually exists, else first available
                key_for_mask = "temperature_2m" if "temperature_2m" in src else next(iter(src.keys()))
                mask = (src[key_for_mask] == nodata)
            else:
                mask = None

            # Write kept bands
            for name in keep_bands:
                arr = src[name].astype(np.float32, copy=False)
                if mask is not None:
                    arr = arr.copy()
                    arr[mask] = nodata
                write_block(out_ds, out_index[name], arr, xoff, yoff)

            # Derived: VPD
            if "vpd_kpa" in derived:
                vpd = compute_vpd_kpa(src["temperature_2m"], src["dewpoint_temperature_2m"])
                if mask is not None:
                    vpd = vpd.copy()
                    vpd[mask] = nodata
                write_block(out_ds, out_index["vpd_kpa"], vpd, xoff, yoff)

    out_ds.FlushCache()
    out_ds = None
    ds = None


# ---------------------------------------------------------------------
# 5) CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract selected ERA5 bands and compute derived indicators (GDAL, block-wise)."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder containing ERA5 multi-band GeoTIFFs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output folder for processed GeoTIFFs")

    # IMPORTANT: defaults must be comma-joined (not empty-joined)
    parser.add_argument(
        "--keep_bands",
        type=str,
        default="temperature_2m,temperature_2m_max,total_precipitation_sum,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2,volumetric_soil_water_layer_3, volumetric_soil_water_layer_4, potential_evaporation_sum,snow_cover".join(KEEP_BANDS_DEFAULT),
        help=(
            "Comma-separated raw ERA5 bands to keep in output. "
            "Aligned with land-cover-change plan: moisture availability, atmospheric dryness, energy, temperature."
        ),
    )
    parser.add_argument(
        "--derived",
        type=str,
        default="vpd_kpa".join(DERIVED_DEFAULT),
        help="Comma-separated derived variables to compute. Supported: vpd_kpa",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_lc_core_vpd",
        help="Suffix appended to output filename before .tif",
    )
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir
    ensure_dir(out_dir)

    keep_bands = parse_list_arg(args.keep_bands)
    derived = parse_list_arg(args.derived)

    tifs = list_tifs(in_dir)
    if not tifs:
        raise RuntimeError(f"No GeoTIFF files found in: {in_dir}")

    # Quick consistency checks
    if "vpd_kpa" in derived:
        # Ensure required raw bands exist (even if not kept)
        if "temperature_2m" not in BAND_INDEX or "dewpoint_temperature_2m" not in BAND_INDEX:
            raise RuntimeError("Band mapping missing temperature_2m or dewpoint_temperature_2m for VPD.")

    for in_path in tifs:
        base = os.path.basename(in_path)
        root, _ = os.path.splitext(base)
        out_path = os.path.join(out_dir, f"{root}{args.suffix}.tif")
        print(f"[INFO] Processing: {base} -> {os.path.basename(out_path)}")
        process_one_file(in_path, out_path, keep_bands, derived)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
