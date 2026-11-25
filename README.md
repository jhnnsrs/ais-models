# ais-models

Biophysical NEURON models exploring how the **axon initial segment (AIS)** shapes
a neuron's firing. The project builds a multi-compartment cell from a spreadsheet
of biophysical parameters, runs current-clamp simulations through the
[Arkitekt](https://arkitekt.live) / [`elektro`](https://pypi.org/project/elektro/)
stack, and compares a cell with a normal AIS against one whose AIS is (almost)
removed.

---

## What this does

The central experiment is **"AIS vs no-AIS"**: take one cell geometry and
biophysics, change *only* the length of the AIS section, and see how differently
the two cells fire under identical drive.

There are two entry points, both building the same cell but asking different
questions:

| Script | Question it answers |
| --- | --- |
| `nico.py` | Drive the soma hard (100–230 pA step + sine + white noise) and compare the fast, irregular spiking of a normal-AIS vs short-AIS cell. |
| `nico_4_5hz.py` | *Tune* each cell's drive to a matched slow **4–5 Hz** tonic rate (rheobase differs per cell), run a long 10 s trace, and export the results. |

Both publish their models and simulations to an Arkitekt server (via `elektro`)
and combine the two cells into a single side-by-side comparison experiment.

---

## Repository layout

```
ais-models/
├── nico.py                  # Entry point 1: hard-drive comparison. Defines the
│                            #   cell topology + build_neuron_model() (reused elsewhere).
├── nico_4_5hz.py            # Entry point 2: sweeps/tunes each cell to 4–5 Hz,
│                            #   runs a long trace, exports traces to a zip.
├── from_excel.py            # Loads biophysics from the .xlsx into elektro's schema,
│                            #   re-attaching each mechanism's NEURON unit.
├── biophysics_simple_fork_210330_v1.xlsx   # Per-compartment biophysical parameters.
├── models/                  # NEURON mechanism definitions (.mod files) — the
│                            #   "mod environment" the server validates against.
├── pyproject.toml           # Project + dependencies (managed by uv).
├── uv.lock                  # Locked dependency versions.
└── .arkitekt/                # Local Arkitekt app folder (git-/docker-ignored credentials).
```

### How the pieces fit together

```
biophysics_simple_fork_210330_v1.xlsx
                │  load_biophysics_from_excel()  (from_excel.py)
                ▼
        BiophysicsInput ──┐
                          ├──►  build_neuron_model()  ──►  NeuronModel  ──►  simulate
   CELL_TOPOLOGY  ────────┘         (nico.py)                                   │
   (soma, dends, AIS, axon)                                                     ▼
   models/*.mod  ──► build_and_zip_environment() ──► ModEnvironment    PublishedRun / Experiment
```

---

## Key concepts

- **The cell** (`CELL_TOPOLOGY` in `nico.py`): a soma with several dendrites, an
  `AIS_dist` section, and a distal axon. Only the AIS **length** is varied between
  the two conditions — `ais_len=28.6` µm (normal) vs `ais_len≈0` (effectively no
  AIS).

- **Biophysics from Excel** (`from_excel.py`): each column of the sheet is a
  compartment; rows are mechanism parameters. The loader re-attaches the correct
  NEURON unit to every value via `SECTION_PARAM_UNITS`. **This map matters** — for
  example `gbar_hNa` is declared in `mS/cm²` while every other conductance is in
  `S/cm²` (a 1000× difference). A wrong unit validates silently and breaks the
  model, so units are mapped explicitly per mechanism rather than guessed.

- **Mechanisms** (`models/*.mod`): standard NEURON channel/pump definitions
  (Na, K, Ca, K(Ca), Ih, leak, etc.). `build_and_zip_environment()` bundles these
  into a `ModEnvironment` that the server compiles and validates parameters
  against.

- **Arkitekt / elektro**: models, environments, simulations and experiments are
  all created on an Arkitekt server through an `Elektro` client. The
  `with connect(Elektro, identifier="nico") as elektro:` block connects and hands back
  that client, which is passed explicitly to every helper (`build_neuron_model`,
  `arun_simulation`, `acombine`, ...).

- **Runs and experiments** (elektro 4): `arun_simulation` publishes a run as one
  `ArrayDataset` per recording/stimulus, timed onto a new clock, and returns a
  `PublishedRun` (`clock`, `recordings`, `stimuli`). `acombine` builds one
  experiment on the first run's clock, places the other runs' clocks onto it at
  0 s, and adds a trace layer per dataset.

---

## Prerequisites

- **Python 3.12** (see `.python-version`).
- [**uv**](https://docs.astral.sh/uv/) for dependency management.
- A reachable **Arkitekt server** — the scripts log in via `connect(Elektro, identifier="nico")` and
  publish models/simulations to it. Without one, the `__main__` blocks will not
  complete.

Core dependencies (`pyproject.toml`): `arkitekt[elektro]>=5`, `elektro[neuron]>=4.2`,
`neuron`, `openpyxl`, `pandas`, `koil`, `kanne` (units).

---

## Setup

```bash
# Install dependencies into a local .venv
uv sync

# (optional) activate the environment
source .venv/bin/activate
```

---

## Running

Both scripts are run as standalone modules. They will connect to your Arkitekt
server, register the mechanism environment, build the two cells, simulate, and
publish a combined experiment.

```bash
# Hard-drive comparison (normal vs short AIS)
uv run python nico.py

# Tune each cell to 4–5 Hz, run a long trace, export traces to a zip
uv run python nico_4_5hz.py
```

`nico_4_5hz.py` additionally writes **`nico_4_5hz_traces.zip`** to the working
directory — one folder per model containing `time_trace.csv`, per-recording and
per-stimulus CSVs, and `metadata.json`.

### Tuning knobs (`nico_4_5hz.py`)

Firing is non-deterministic by design (white noise + high-frequency sine), so the
script tunes the *mean* rate rather than an exact spike count:

- `TARGET_LOW_HZ` / `TARGET_HIGH_HZ` — the target firing band (default 4–5 Hz).
- `CANDIDATE_AMPS_PA_AIS` / `CANDIDATE_AMPS_PA_NO_AIS` — coarse current-clamp
  amplitude sweeps. The two cells have different rheobase, so each gets its own
  range; widen these if a sweep reports the band lies outside the range.
- `SWEEP_DURATION` / `FINAL_DURATION` — short window for tuning, long window for
  the published run.
- `REFINE_ITERATIONS` / `REFINE_POINTS` — how aggressively to zoom into the
  bracket around the target rate after the coarse sweep.

---

## Deployment (Arkitekt)

The scripts are standalone `connect(Elektro)` scripts, not an Arkitekt plugin app. Current
`arkitekt` has no manifest: to package this as a plugin, declare an `App(...)` in
a module and use `arkitekt plugin init` / `arkitekt plugin build <module>:app`.
See the [Arkitekt docs](https://arkitekt.live).

---

## Notes & gotchas

- The Excel loader **raises** if it hits a parameter with no registered unit —
  add the mechanism's unit (from the corresponding `models/<mech>.mod`
  `PARAMETER` block) to `SECTION_PARAM_UNITS` in `from_excel.py`.
- A couple of values are pinned in the loader (`g_pas → 0.0002`, `e_pas → -70 mV`)
  to match the original model's net behaviour; these are flagged with `TODO`
  comments in `from_excel.py`.
- `nico_4_5hz.py` reuses `build_neuron_model` and `acombine` from `nico.py`, so
  the cell geometry and biophysics stay identical across both experiments — only
  the drive differs.
