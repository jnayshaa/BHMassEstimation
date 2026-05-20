# Quasar Black Hole Mass Estimation via DRW Variability

**Undergraduate Honors Thesis Pipeline**

This repository contains the full pipeline for estimating supermassive black hole (BH) masses of quasars from their optical variability, using ZTF light curves and machine learning. The project is structured as a sequence of numbered Jupyter notebooks that can be followed step by step.

---

## Project Overview

Quasars are accreting supermassive black holes whose optical brightness fluctuates stochastically over timescales of days to years. This variability is well-described by a **Damped Random Walk (DRW)** process, and DRW parameters — particularly the characteristic timescale τ and amplitude σ — have been shown to correlate with black hole mass (Burke et al. 2021; Helias et al. 2026).

This pipeline:
1. Builds a clean quasar sample from SDSS DR16Q with known virial BH masses
2. Fetches ZTF r-band light curves for each quasar via cloud-hosted Parquet files
3. Fits a DRW model to each light curve using Gaussian Process regression + MCMC
4. Trains gradient-boosted tree models to predict BH mass from the DRW parameters
5. Validates the physical DRW-mass correlations against published literature

---

## Pipeline Structure

```
01_build_training_dataset.ipynb      -- Build and filter the quasar catalog
02_healpix_coverage_maps.ipynb       -- Visualize sky coverage (diagnostic)
03_drw_parameter_estimation.ipynb    -- Fit DRW models via MCMC for all quasars
ztf_bulk_lightcurves.py              -- Utility: bulk download ZTF light curves locally
04_xgboost_bhmass.ipynb             -- Train XGBoost to predict BH mass
05_model_comparison_diagnostics.ipynb -- Full model comparison, ablation, physical diagnostics
```

### Data flow

```
SDSS DR16Q catalog (FITS)
        |
        v
01 -- build_training_dataset
        |
        +--> data/DR16Q_final_stripe82.fits
                    |
                    v
             03 -- drw_parameter_estimation  <--  ZTF DR23 (AWS S3 via lsdb)
                    |
                    +--> data/DRW_results.csv
                                |
                                v
                    04 -- xgboost_bhmass
                    05 -- model_comparison_diagnostics
```

---

## Notebooks

### `01_build_training_dataset.ipynb`
Loads the SDSS DR16Q quasar properties catalog (Wu & Shen 2022), filters for redshift z ≤ 2 and valid virial BH mass, cross-matches with the original DR16Q photometry catalog to recover SDSS PSF magnitudes, and applies a brightness cut of r < 20 for ZTF detectability. Extracts the Stripe 82 subset for variability analysis.

**Key output:** `data/DR16Q_final_stripe82.fits` (~1,000–5,000 quasars depending on magnitude cut)

### `02_healpix_coverage_maps.ipynb`
Diagnostic notebook. Projects ZTF DR24 epoch-count maps, the SDSS DR16Q quasar footprint, and the Stripe 82 region onto a HEALPix grid and plots them in a combined Mollweide projection. Useful for confirming that all three datasets overlap as expected before running the main pipeline.

### `03_drw_parameter_estimation.ipynb`
The core computational step. For each quasar in the Stripe 82 catalog:
- Fetches ZTF DR23 r-band light curves via **lsdb** (cloud-hosted, spatially partitioned Parquet on AWS S3)
- Selects the closest counterpart within 2 arcsec using angular separation
- Applies quality cuts: catflags == 0, magerr < 0.2 mag
- Fits a DRW model using `celerite` (GP kernel) + `emcee` (MCMC sampler, 32 walkers, 2000 steps, 500 burn-in)
- Extracts τ, σ, μ from the posterior median

**Key output:** `data/DRW_results.csv`

### `ztf_bulk_lightcurves.py`
Standalone script for bulk-downloading ZTF DR24 light curves outside of the Colab/notebook environment. Takes a target list (CSV, FITS, or hardcoded) and saves individual CSV files per (objectid, filterid). Useful for local preprocessing or extending the pipeline to a larger target list.

### `04_xgboost_bhmass.ipynb`
Trains an XGBoost gradient-boosted tree to predict `LOGMBH` from five features: log τ, log σ, μ, PSFMAG_r, and Z_FIT. Includes proper preprocessing (log-transform, inf/nan removal), early stopping, 5-fold cross-validation, and diagnostic plots (predicted vs true, feature importance, residuals).

### `05_model_comparison_diagnostics.ipynb`
Comprehensive evaluation notebook:
- **Model comparison:** Ridge, Random Forest, XGBoost, LightGBM, MLP — all with 5-fold CV
- **Ablation study:** quantifies the contribution of redshift (Z_FIT) as a feature
- **Physical diagnostics:** power-law fits to the rest-frame τ-MBH and σ-MBH relations, compared against Helias et al. (2026), Burke et al. (2021), and MacLeod et al. (2010)

---

## Data

The pipeline uses publicly available datasets:

| Dataset | Description | Source |
|---------|-------------|--------|
| DR16Q (Wu & Shen 2022) | SDSS quasar properties + virial BH masses | http://quasar.astro.illinois.edu/paper_data/DR16Q/ |
| DR16Q_v4 | Original SDSS DR16Q catalog (photometry) | https://data.sdss.org/datamodel/files/BOSS_QSO/DR16Q/ |
| ZTF DR23 | Photometric light curves (lsdb/AWS) | `s3://ipac-irsa-ztf/contributed/dr23/lc/hats` |
| ZTF DR24 | Epoch-count coverage map (for notebook 02) | `s3://ipac-irsa-ztf/contributed/dr24/lc/hats` |

The raw FITS files (`dr16q_prop_May01_2024.fits`, `DR16Q_v4.fits`) should be placed in `data/raw/`. ZTF data is fetched on-the-fly and never stored in full.

---

## Requirements

```bash
pip install astropy astroquery pandas numpy "numpy<2.0" pyarrow tqdm
pip install "lsdb==0.9.0" hats s3fs
pip install emcee celerite
pip install xgboost lightgbm scikit-learn matplotlib seaborn
pip install healpy   # notebook 02 only
```

All notebooks were originally developed on Google Colab (Python 3.10). The `drive.mount` cells in each notebook can be removed if running locally.

<!-- ---

## Key Results

The pipeline recovers DRW parameters for a large sample of Stripe 82 quasars and demonstrates that:
- The rest-frame DRW timescale τ correlates with BH mass as a power law consistent with published slopes (Burke+21, Helias+26)
- XGBoost and LightGBM outperform linear regression and MLP for this task
- Including photometric redshift (Z_FIT) as a feature meaningfully improves prediction accuracy (ablation study)

---

## References

- Wu & Shen 2022, ApJS -- DR16Q quasar properties: https://iopscience.iop.org/article/10.3847/1538-4365/ac9ead
- Lyke et al. 2020, ApJS -- SDSS DR16Q catalog: https://iopscience.iop.org/article/10.3847/1538-4365/aba623
- Burke et al. 2021 -- DRW timescale as a BH mass proxy
- Helias et al. 2026 -- τ-MBH power-law calibration with 127 AGN
- MacLeod et al. 2010 -- DRW variability amplitude and BH mass -->
