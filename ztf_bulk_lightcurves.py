"""
ztf_bulk_lightcurves.py
-----------------------
Bulk-download ZTF DR24 r-band light curves for a list of targets using LSDB + AWS S3.

This script bypasses the IRSA web API entirely and reads static Parquet files directly
from the IPAC S3 bucket. Each target gets a cone search; clean epochs (catflags==0)
are written to individual CSV files in the output directory.

Designed to be run locally or on a Colab instance as a standalone preprocessing step,
independent of the main Jupyter pipeline.

Requirements
------------
    pip install lsdb hats astropy pandas

Usage
-----
1. Set your target list in the CONFIGURATION section below (CSV, FITS, or hardcoded).
2. Adjust SEARCH_RADIUS and OUTPUT_DIR as needed.
3. Run:  python ztf_bulk_lightcurves.py

Output
------
One CSV per (objectid, filterid) combination found within the search cone:
    <name>_oid<objectid>_f<filterid>.csv

Each CSV contains columns: hmjd, mag, magerr, clrcoeff, catflags, objectid, filterid, objra, objdec
"""

import os
import pandas as pd
import numpy as np
import lsdb
from astropy.table import Table


# ============================================================
# CONFIGURATION -- edit this section
# ============================================================

# Cone search radius around each target (arcseconds)
SEARCH_RADIUS = 0.3

# Output directory for saved light curve CSVs
OUTPUT_DIR = "/Volumes/naysha/massestimator/data"

# ZTF DR24 HATS catalog on AWS (read-only, no credentials required)
ZTF_DR24_URL = "s3://ipac-irsa-ztf/contributed/dr24/lc/hats"

# ── Option A: Load targets from a CSV file ────────────────────────────────────
# Expects columns named 'ra' and 'dec' (decimal degrees).
# An optional 'name' column is used to label output files.
#
# TARGET_FILE = "my_targets.csv"
# targets = pd.read_csv(TARGET_FILE)

# ── Option B: Load targets from a FITS table ─────────────────────────────────
#
# TARGET_FILE = "my_targets.fits"
# targets = Table.read(TARGET_FILE).to_pandas()

# ── Option C: Hardcode your targets ──────────────────────────────────────────
targets = pd.DataFrame({
    "name": ["target1", "target2", "target3"],
    "ra":   [0.0498315,  210.5,     120.3],
    "dec":  [0.0403923,   -5.2,      45.8],
})

# ============================================================
# END CONFIGURATION
# ============================================================

# Column name aliases -- adjust if your table uses different names
RA_COL   = "ra"
DEC_COL  = "dec"
NAME_COL = "name" if "name" in targets.columns else None

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_target(name, ra, dec):
    """Fetch, clean, and save ZTF light curves for a single sky position.

    For each source found within SEARCH_RADIUS arcsec of (ra, dec), one CSV
    is written per (objectid, filterid) combination.

    Parameters
    ----------
    name : str   -- label used in output filenames
    ra   : float -- right ascension in decimal degrees
    dec  : float -- declination in decimal degrees
    """
    print(f"  Cone searching ZTF DR24 within {SEARCH_RADIUS} arcsec ...")

    # Open the catalog with a spatial filter. Only partitions that overlap
    # this cone are downloaded -- the full dataset is never read into memory.
    cat = lsdb.open_catalog(
        ZTF_DR24_URL,
        search_filter=lsdb.ConeSearch(ra, dec, radius_arcsec=SEARCH_RADIUS),
    )

    df = cat.compute()   # triggers the actual S3 reads

    if len(df) == 0:
        print(f"  No ZTF sources found within {SEARCH_RADIUS} arcsec")
        return

    print(f"  Found {len(df)} source(s)")

    for _, src in df.iterrows():
        oid      = src["objectid"]
        filterid = src["filterid"]
        obj_ra   = src["objra"]
        obj_dec  = src["objdec"]

        # In DR24 HATS format the light curve is stored as a nested DataFrame
        # with columns: hmjd, mag, magerr, clrcoeff, catflags
        lc = src["lightcurve"]

        if lc is None or len(lc) == 0:
            print(f"  objectid={oid} filterid={filterid}: no epochs, skipping")
            continue

        # Keep only clean epochs (catflags==0 means no known artefacts)
        if "catflags" in lc.columns:
            lc = lc[lc["catflags"] == 0]

        if len(lc) == 0:
            print(f"  objectid={oid} filterid={filterid}: all epochs flagged, skipping")
            continue

        # Attach source metadata so each CSV is self-contained
        lc = lc.copy()
        lc["objectid"] = oid
        lc["filterid"] = filterid
        lc["objra"]    = obj_ra
        lc["objdec"]   = obj_dec

        fname   = f"{name}_oid{oid}_f{filterid}.csv"
        outpath = os.path.join(OUTPUT_DIR, fname)
        lc.to_csv(outpath, index=False)
        print(f"  Saved {outpath}  ({len(lc)} clean epochs, filterid={filterid})")


# ── Main loop ─────────────────────────────────────────────────────────────────

print("Opening ZTF DR24 catalog from AWS ...")
print("(First call reads only metadata -- no full dataset download)")

for i, row in targets.iterrows():
    ra   = row[RA_COL]
    dec  = row[DEC_COL]
    name = row[NAME_COL] if NAME_COL else f"target_{i:04d}"

    print(f"\n--- [{i+1}/{len(targets)}] {name}  (RA={ra:.5f}, Dec={dec:.5f}) ---")

    try:
        process_target(name, ra, dec)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print(f"\nDone. Light curves saved to: {OUTPUT_DIR}/")
