# OilTrace
AI-Driven Oil Spill Detection and Vessel Attribution System

**Leveraging satellite imagery to determine oil spills at sea, and correlating AIS data to identify the vessel most likely associated with the spill.**

A working end-to-end prototype: it detects an oil slick in a SAR-like image, reconstructs where and when the oil was released by running the drift backwards, searches historical AIS around that point in space and time, and ranks candidate vessels with a transparent, explainable score.

```
DETECT  ->  TRACE  ->  ATTRIBUTE  ->  VISUALISE
```

> **This is a decision-support prototype.** It reports *investigation priority*, never proof of responsibility. The top result is labelled "highest correlated candidate", not "confirmed responsible vessel".

---

## 1. How to run it

```bash
# 1. Get the code
git clone <your-repo-url>
cd oiltrace-ai

# 2. Create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the whole pipeline headlessly (recommended before a demo)
python run_demo.py

# 5. Launch the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`. Leave **Presentation demo** on, press **Run OILTRACE analysis**, and the complete pipeline finishes in a few seconds.

There is nothing else to download. On first run the app generates its own demo scene, environment file and AIS extract. No internet connection is needed after `pip install`.

**Regenerate every scene's demo data from scratch:**

```bash
python run_demo.py --force
```

**If the port is busy:**

```bash
streamlit run app.py --server.port 8502
```

---

## 2. What the pipeline does

```
Satellite image
      |
  Preprocessing  ................  median despeckle, Gaussian smoothing
      |
  Segmentation  .................  U-Net / reference mask / image processing
      |
  Spill mask
      |
  Spill geometry  ...............  polygon, centroid, area, extent, orientation
      |
  Synthetic georeference  .......  pixel -> lat/lon via configurable bbox
      |
  Environmental data  ...........  currents, wind, waves
      |
  Backward drift (hindcast)  ....  probable origin + estimated release time
      |
  Historical AIS
      |
  Spatial filter  ...............  within candidate radius of the origin
      |
  Temporal filter  ..............  within the release window
      |
  Trajectory + behaviour  .......  alignment, speed, heading, gaps
      |
  Attribution scoring  ..........  interpretable weighted model
      |
  Vessel ranking -> GIS dashboard
```

The satellite image is **never** joined directly to AIS records. The link runs through the drift model — that is the central logic of the system.

---

## 3. What each module does

| File | Responsibility |
|---|---|
| `app.py` | Streamlit dashboard. Presentation only — contains no analysis logic. |
| `config.yaml` | Every tunable value: bounding box, weights, radii, drift coefficients. |
| `run_demo.py` | Headless end-to-end test. Prints every stage and checks the demo story. |
| `modules/geo.py` | `pixel_to_geo` / `geo_to_pixel`, haversine distance, bearings, offsets. |
| `modules/dataset_adapter.py` | Walks a local dataset folder and reports its **actual** structure. Pairs images with masks by filename. |
| `modules/detection.py` | Three-mode segmentation plus a confidence score computed from the pixels. |
| `modules/spill_analysis.py` | Builds the "Spill Object": polygon, centroid, area, perimeter, length, width, elongation, bearing. |
| `modules/drift.py` | Vector drift model. Forward projection and backward hindcast to the probable origin. |
| `modules/ais_filter.py` | Groups AIS into per-vessel tracks, measures closest approach, applies the spatial then temporal filters. |
| `modules/trajectory.py` | Proximity, temporal and cosine-alignment scores against the reconstructed drift. |
| `modules/behaviour.py` | Five behavioural indicators plus AIS continuity. |
| `modules/attribution.py` | Weighted scoring, ranking, confidence bands, plain-language reasons. |
| `modules/visualization.py` | Folium map layers and the raster overlay panels. |
| `modules/report.py` | Self-contained HTML investigation report. |
| `modules/pipeline.py` | Orchestrates every stage and records status for the progress checklist. |
| `utils/scene_builder.py` | Deterministic synthetic SAR scene + ground-truth mask. |
| `utils/demo_data.py` | Deterministic environment CSV and 10-vessel AIS extract. |

---

## 4. The attribution model

```
score = 30%  proximity to probable origin
      + 25%  temporal correlation with the release window
      + 20%  trajectory alignment with reconstructed drift
      + 15%  behavioural anomaly
      + 10%  AIS track continuity
```

Deliberately **not** a trained ranker. Every point traces back to a measured quantity, which is what an investigator actually needs to act on. Weights live in `config.yaml` and can be changed without touching code.

Behavioural indicators — speed change, heading change, loitering, route deviation, AIS gaps — describe *motion patterns only*. Vessels slow, turn and lose AIS coverage for many lawful reasons. The score is named "behavioural anomaly score" and carries only 15% weight for exactly this reason.

---

## 5. Detection modes

The application **never crashes because a model is missing.** It tries three modes in priority order:

| Mode | When it is used | Label shown in the UI |
|---|---|---|
| 1. U-Net | `models/unet_oilspill.onnx` exists and `onnxruntime` is installed | `U-Net Model` |
| 2. Reference mask | The dataset supplies a matching ground-truth mask | `Demo Mode - Reference Ground Truth` |
| 3. Image processing | Always available | `Prototype Segmentation - Image Processing` |

Mode 2 is honestly labelled. The app states that the mask was **not** generated by a model, and it does **not** compute IoU/Dice in that mode — scoring a mask against itself would be a meaningless perfect score.

The fallback (mode 3) is a local-background dark-spot detector, not a global threshold. Oil damps capillary waves, so a slick returns less energy than the surrounding sea; comparing each pixel against a heavily blurred copy of the scene makes this robust to wind streaks and to the slick occupying under 1% of the frame. On the built-in demo scene it reaches **IoU 0.937 / Dice 0.968** against the ground truth.

**Detection confidence is computed, not assigned** — a weighted combination of three measurements:

- **contrast** — how much darker the slick interior is than the surrounding annulus
- **homogeneity** — a slick's return is smoother than wind-roughened water
- **elongation** — slicks are elongated; compact blobs are usually look-alikes

---

## 6. Using your own Kaggle oil-spill dataset

Extract it into `data/kaggle_oilspill/`:

```
data/kaggle_oilspill/
    images/
        img_001.png
        ...
    masks/
        img_001_mask.png
        ...
```

**The layout above is only an example.** Nothing is assumed. `modules/dataset_adapter.py` walks whatever you put there and reports what it genuinely finds — formats, pixel modes, dimensions, folder labels, and how many image/mask pairs it could match. Open the **Dataset & provenance** tab to see the inspection result.

Pairing works via directory hints (`images/`, `masks/`, `labels/`, `gt/`, …) and filename suffixes (`_mask`, `_gt`, `_label`, `_seg`, …). If no pairs match, detection falls through to the image-processing mode.

Then turn **Presentation demo** off in the sidebar and pick an image from the dropdown, or upload one directly.

### Why the geographic coordinates are synthetic

Kaggle oil-spill images are not georeferenced. Rather than discard them, the prototype uses them for the **segmentation stage** — which is where they are genuinely useful — and applies a **configurable demonstration bounding box** *after* detection so the downstream geospatial pipeline can be exercised:

```yaml
demo_scene:
  north: 18.60
  south: 18.20
  west: 72.50
  east: 73.00
```

Editable live in the sidebar. The UI labels this "Synthetic Demo Bounding Box" throughout and never presents these as real coordinates from the dataset. Image dimensions are read dynamically — nothing is hard-coded to 2048 × 2048.

---

## 7. Replacing the simulated inputs

### Real AIS instead of synthetic AIS

Replace `data/ais/vessels.csv`. Required columns:

```
mmsi, timestamp, latitude, longitude, speed, heading, vessel_name, vessel_type
```

Timestamps ISO-8601 UTC, speed in knots, heading in degrees from north. Supply **full trajectories** (many rows per vessel), not one position each — trajectory and behaviour analysis need the sequence. Nothing else changes; `ais_filter.load_tracks` groups by MMSI automatically.

For a live system, replace the `pd.read_csv` call in `modules/pipeline.py` with a spatio-temporal query against your AIS database, bounded by the probable origin and release window.

### A trained U-Net instead of demo segmentation

1. Train on SAR imagery (VV + VH polarisation) with ground-truth masks, `0 = background`, `1 = oil`.
2. Export to ONNX as `models/unet_oilspill.onnx`.
3. `pip install onnxruntime`.

Mode 1 activates automatically on the next run. Input tensor shape is read from the model, and the output is resized back to the source resolution. To use PyTorch or TensorFlow directly instead, replace `_try_unet()` in `modules/detection.py` — it is the only place inference happens.

### Real environmental data instead of synthetic

Replace `data/environment/environment.csv`. Required columns:

```
timestamp, latitude, longitude, current_u, current_v, wind_u, wind_v,
wave_direction, wave_speed
```

Velocities in m/s (`u` = eastward, `v` = northward), wave direction in degrees toward which waves travel. Source from Copernicus Marine, HYCOM, ERA5 or an equivalent reanalysis.

`EnvironmentField` in `modules/drift.py` uses nearest-neighbour sampling — that class is the seam where an xarray/NetCDF gridded reader drops in.

### Real georeferenced Sentinel-1

Read the GeoTIFF affine transform with `rasterio` and replace `pixel_to_geo` in `modules/geo.py`. Everything downstream already speaks latitude/longitude, so no other module changes.

---

## 8. What is prototype, and what would production look like

| Stage | Prototype (this repo) | Production |
|---|---|---|
| Satellite imagery | Local image dataset / synthetic scene | Georeferenced Sentinel-1 SAR ingestion |
| Georeferencing | Synthetic demo bounding box | Satellite geospatial metadata (GeoTIFF affine) |
| Segmentation | Reference mask / deterministic image processing | Trained U-Net on VV + VH polarisation |
| Environmental | Synthetic metocean CSV | Operational current, wind and wave reanalysis |
| AIS | Synthetic 10-vessel extract | Historical AIS database (terrestrial + satellite) |
| Drift | Simplified vector advection | Lagrangian particle model with weathering and an uncertainty envelope |
| Attribution | Interpretable weighted scoring | ML ranking model trained on confirmed cases |
| GIS | Folium in-browser map | PostGIS + tile services + role-based access |

### Known limitations, stated plainly

- Geographic coordinates come from a synthetic bounding box, **not** satellite georeferencing. Areas are labelled "estimated prototype area" and are not survey grade.
- The drift model has no weathering, no turbulent diffusion and no uncertainty ellipse — it returns a point estimate where a real system returns a probability field.
- The hindcast duration is derived from slick elongation via a single spreading-rate constant. This is a deliberately simple parameterisation.
- Attribution weights are chosen for interpretability, not fitted to confirmed cases.
- Behavioural anomalies are **not** evidence of illegal discharge.

### Production improvements worth naming

Multi-sensor fusion (Sentinel-1 + optical + thermal), look-alike discrimination (low-wind areas, biogenic films, algal blooms all appear dark in SAR), ensemble drift with uncertainty envelopes, AIS spoofing and dark-vessel detection via satellite RF, oil-type fingerprinting against bunker samples, and a chain-of-custody audit trail for evidentiary use.

---

## 9. Showing multiple images to judges

There are three ways, in increasing order of "this is a real system".

### a) Switch scenario (best for a live pitch)

The **Scenario** dropdown in the sidebar holds three fully independent cases. They share no data — each has its own imagery, metocean field, AIS extract and guilty vessel:

| Scenario | Character | Hindcast | Top candidate |
|---|---|---|---|
| Arabian Sea — offshore Maharashtra | 12.6 km² slick drifting northeast | 3.50 h | MARINA STAR, 94.4 |
| Gulf of Kutch — Kandla approach | Smaller 8.2 km² slick drifting **southwest** | 2.50 h | KUTCH VOYAGER, 93.2 |
| Bay of Bengal — Chennai approach | Large 21.4 km² weathered slick | 5.00 h | CORMORANT, 94.8 |

Different locations, different drift directions, different slick sizes, different fleets. Flipping between them mid-demo is the strongest answer to *"does this only work on one picture?"* — the drift bearing swings from 057° to 235° to 045° and the system tracks it each time.

Add your own scenario by appending to `scenes:` in `config.yaml`. No code changes: give it a bounding box, an observation time, an intended centroid/origin/release, a slick size and ten vessel names. The generator builds coherent imagery, metocean data and AIS from that, and `python run_demo.py` verifies it.

### b) Load images from a dataset folder

Turn **Presentation demo** off and pick any image from `data/kaggle_oilspill/` (or upload one). Detection, geometry and georeferencing all run on the real image.

Because an arbitrary image has no environmental or AIS data attached, the prototype synthesises both **around the spill it actually detected** — see `adaptive_context` in `config.yaml`. The dashboard shows a banner saying exactly that. This demonstrates the full mechanism end to end without pretending there is real AIS for an arbitrary picture.

### c) Verify everything at once, on stage

```bash
python run_demo.py                      # every scene, with pass/fail checks
python run_demo.py --scene bay_of_bengal
python verify.py                        # 57-check specification audit
```

Running `verify.py` in a terminal beside the dashboard is a strong move: it proves the numbers are recomputed, not memorised.

### An honest note on what this does and does not show

Switching scenarios proves the **pipeline generalises** — the geometry, drift solver, filtering and scoring all work at any location, on any heading, over any time window. It does **not** prove the detector generalises to real Sentinel-1 imagery, because all three scenes use synthetic SAR. If a judge asks, that is the right answer to give, along with the U-Net path in §7 that is already wired and waiting for a trained model.

---

## 10. The demo scenario

Deterministic by design — a fixed seed means a live presentation produces the same result every time.

```
Synthetic SAR scene  ->  slick detected, 12.6 km², 90.6% confidence
                     ->  centroid  18.4335 N, 72.8246 E  at 10:30 UTC
                     ->  backward drift 3.50 h at 0.673 m/s, bearing 057°
                     ->  probable origin  18.3904 N, 72.7540 E
                     ->  estimated release  07:00 UTC
                     ->  10 vessels in the AIS extract
                     ->  6 within the 20 km radius
                     ->  4 within the ±2 h release window
                     ->  MARINA STAR ranked #1 at 94.4/100
```

**None of these values are hard-coded.** The scenario is engineered at the *input* end: the ocean currents in `environment.csv` are solved so that `current = required_drift − wind_drag − wave_drift`, and the scene builder bisects the slick's minor axis until the rasterised footprint matches the designed area. The pipeline then rediscovers the origin, the release time and the ranking independently. Change the bounding box or the search radius in the sidebar and every number moves accordingly.

The ten vessels are built with deliberately different relationships to the spill so the ranking is visibly meaningful:

| Vessel | Relationship | Outcome |
|---|---|---|
| MARINA STAR | 0.7 km from origin, on time, aligned course, slows to 1.8 kn and turns 52°, continuous AIS | **#1 — 94.4** |
| OCEAN PIONEER | Moderately close, right time, moderate alignment | #2 — 73.1 |
| SEA FALCON | Close and on time, but sailing *against* the drift | #3 — 58.5 |
| EASTERN PEARL | Distant, early, has an AIS gap | #4 — 45.4 |
| BLUE HORIZON | Close but 4.5 h early | Excluded — temporal |
| NORDIC WAVE | Within radius but 3.2 h early | Excluded — temporal |
| ANCHOR BAY, GULF MERIDIAN, CORAL TRADER, PACIFIC DAWN | Beyond the 20 km radius | Excluded — spatial |

---

## 11. Publishing to GitHub

```bash
cd oiltrace-ai
git init
git add .
git commit -m "OILTRACE AI - satellite oil spill detection and vessel attribution prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/oiltrace-ai.git
git push -u origin main
```

`.gitignore` already excludes the virtual environment, generated demo assets, large datasets and model weights. Anyone who clones the repo gets a working demo, because the demo data regenerates itself on first run.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: streamlit_folium` | `pip install -r requirements.txt` inside the activated venv |
| Map area is blank | Base tiles need internet; every other panel works offline. Uncheck "SAR scene overlay" if the raster is slow. |
| `Port 8501 is already in use` | `streamlit run app.py --server.port 8502` |
| Demo data looks wrong | `python run_demo.py --force` regenerates it |
| No candidate vessels | Widen the candidate radius or release window in the sidebar |
| OpenCV import error on a headless server | Use `opencv-python-headless` (already pinned in `requirements.txt`) |

---

## Disclaimer

Prototype analytical result — requires human investigation and verification. Correlation scores indicate investigation priority only and are not evidence of responsibility for a discharge.

