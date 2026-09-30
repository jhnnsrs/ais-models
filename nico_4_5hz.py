"""
Variation of ``nico.py`` tuned for slow ~4-5 Hz tonic firing.

``nico.py`` drives the soma hard (a 100-230 pA step plus a sine wave and white
noise), which pushes the cell into fast, irregular spiking. Here we keep the same
sine + white-noise drive but tune the sustained current-clamp amplitude down so
the *mean* rate sits in the 4-5 Hz band.

Rheobase depends on the model's biophysics, so we don't hard-code an amplitude.
Instead we sweep candidate clamp amplitudes -- each measured with the sine and
noise present, exactly as in the final run -- count somatic spikes, and pick the
one whose mean rate lands in 4-5 Hz before publishing a long final run.

The white noise (and to a lesser extent the high-frequency sine) makes firing
non-deterministic: the same amplitude gives a slightly different rate each run.
That is by design -- we tune the mean, not an exact spike count. The reusable
model builder is imported from ``nico.py`` so the cell geometry and biophysics
stay identical -- only the drive amplitude changes.
"""

import csv
import io
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from koil import unkoil
from arkitekt import easy

from elektro import Elektro, elektro_service
from elektro.neuron.parse import build_and_zip_environment
from elektro.api.schema import ArrayDataset, NeuronModel
from elektro.neuron.simulate import (
    CurrentClampStimulus,
    PublishedRun,
    SineWaveStimulus,
    WhiteNoiseStimulus,
    VRecord,
    asimulate,
    arun_simulation,
)

from nico import build_neuron_model, acombine  # reuse topology/biophysics + combiner

# Units come from the shared kanne registry (same one nico.py uses) so every
# quantity lives on one registry, rather than a private pint.UnitRegistry().
from kanne.units import pA, ms, s, Hz


# --- Tuning configuration --------------------------------------------------

# Short window for the amplitude sweep (cheaper), long window for the final run.
SWEEP_DURATION = 2 * s
FINAL_DURATION = 10 * s

# Ignore the onset transient when counting spikes so the first big depolarisation
# after current onset doesn't inflate the rate.
SETTLE_MS = 200.0

# An upward crossing of this voltage counts as one spike.
SPIKE_THRESHOLD_MV = 0.0

# Background drive layered on top of the tuned current clamp (same as nico.py).
# Present during BOTH the sweep and the final run so we tune against the real
# stimulus. The noise makes the rate non-deterministic by design.
NOISE_LEVEL = 20 * pA
SINE_AMPLITUDE = 80 * pA
SINE_FREQUENCY = 200 * Hz

# We aim for the midpoint of this band.
TARGET_LOW_HZ = 4.0
TARGET_HIGH_HZ = 5.0
TARGET_MID_HZ = (TARGET_LOW_HZ + TARGET_HIGH_HZ) / 2.0

# Coarse amplitudes (pA) to bracket rheobase. The two cells have very different
# excitability: the intact AIS lowers rheobase (more excitable, fires at less
# current), while the short/absent AIS needs substantially more drive to reach
# the same rate. So we sweep each over its own range -- adjust if a sweep shows
# the band sits outside the range for your biophysics.
CANDIDATE_AMPS_PA_AIS = [20, 30, 40, 50, 60, 70, 80, 100]
CANDIDATE_AMPS_PA_NO_AIS = [100, 140, 180, 220, 260, 300, 340, 380]

# After the coarse sweep, zoom into the bracket around the target rate this many
# times, sampling a handful of interior amplitudes each pass.
REFINE_ITERATIONS = 2
REFINE_POINTS = 4


def build_stimuli(amp_pa: float) -> list:
    """Tuned sustained clamp + the shared sine and white-noise drive.

    Used for both the sweep and the final run so the cell is tuned against the
    exact stimulus it is ultimately published with.
    """
    return [
        CurrentClampStimulus(
            cell="cell_1",
            location="soma",
            position=0.5,
            amp=amp_pa * pA,
            delay=100 * ms,
        ),
        WhiteNoiseStimulus(
            cell="cell_1",
            location="soma",
            position=0.5,
            noise_level=NOISE_LEVEL,
        ),
        SineWaveStimulus(
            cell="cell_1",
            location="soma",
            position=0.5,
            amplitude=SINE_AMPLITUDE,
            frequency=SINE_FREQUENCY,
        ),
    ]


def count_firing_rate(
    time_trace: np.ndarray,
    voltage: np.ndarray,
    settle_ms: float = SETTLE_MS,
    threshold_mv: float = SPIKE_THRESHOLD_MV,
) -> float:
    """Mean firing rate (Hz) from upward threshold crossings, ignoring the onset transient."""
    time_trace = np.asarray(time_trace, dtype=np.float64)
    voltage = np.asarray(voltage, dtype=np.float64)

    mask = time_trace >= settle_ms
    t = time_trace[mask]
    v = voltage[mask]
    if t.size < 2:
        return 0.0

    above = v >= threshold_mv
    # Rising edges only: a False -> True transition is one spike.
    n_spikes = int(np.count_nonzero(~above[:-1] & above[1:]))

    window_s = (t[-1] - t[0]) / 1000.0
    return n_spikes / window_s if window_s > 0 else 0.0


async def sweep_amplitudes(
    model: NeuronModel,
    candidates_pa: list[float],
    process_pool: ProcessPoolExecutor,
) -> list[tuple[float, float]]:
    """Run one current-clamp simulation per candidate amplitude; return (amp_pA, rate_Hz)."""
    results: list[tuple[float, float]] = []
    for amp in candidates_pa:
        sim = await asimulate(
            model=model,
            duration=SWEEP_DURATION,
            records=[
                VRecord(cell="cell_1", location="soma", position=0.5),
            ],
            stims=build_stimuli(amp),
            dt=0.05 * ms,  # 20 kHz sampling
            process_pool=process_pool,
        )
        rate = count_firing_rate(sim.time_trace, sim.recordings[0].values)
        results.append((amp, rate))
        print(f"  {amp:6.1f} pA -> {rate:5.2f} Hz")
    return results


def bracket_for_target(
    sweep: list[tuple[float, float]],
    target_hz: float,
) -> tuple[float, float]:
    """Tightest (lo_amp, hi_amp) whose rates straddle ``target_hz``.

    Firing rate rises with amplitude above rheobase, so we take the highest
    amplitude still at/below target and the lowest still at/above it. If the
    target sits outside every measured rate we fall back to the neighbouring
    amplitudes at the closest end.
    """
    pts = sorted(sweep)  # by amplitude
    amps = [amp for amp, _ in pts]

    below = [amp for amp, rate in pts if rate <= target_hz]
    above = [amp for amp, rate in pts if rate >= target_hz]
    lo = max(below) if below else amps[0]
    hi = min(above) if above else amps[-1]

    if lo == hi:
        # Exact hit or single-sided: widen to the immediate neighbours.
        i = amps.index(lo)
        lo = amps[max(0, i - 1)]
        hi = amps[min(len(amps) - 1, i + 1)]

    return (min(lo, hi), max(lo, hi))


async def tune_amplitude(
    model: NeuronModel,
    process_pool: ProcessPoolExecutor,
    candidate_amps: list[float],
) -> list[tuple[float, float]]:
    """Coarse sweep, then zoom into the bracket around the target rate twice.

    ``candidate_amps`` is the coarse bracket to start from -- pass the AIS or
    No-AIS range depending on the cell's excitability.

    Returns every (amp_pA, rate_Hz) measured across all passes.
    """
    print("Coarse amplitude sweep (somatic spike rate):")
    coarse = await sweep_amplitudes(model, candidate_amps, process_pool)
    measured: dict[float, float] = dict(coarse)

    for iteration in range(1, REFINE_ITERATIONS + 1):
        lo, hi = bracket_for_target(list(measured.items()), TARGET_MID_HZ)
        # Evenly spaced interior amplitudes, skipping any we already measured.
        candidates = [
            round(lo + (hi - lo) * k / (REFINE_POINTS + 1), 2)
            for k in range(1, REFINE_POINTS + 1)
        ]
        candidates = [a for a in candidates if a not in measured]
        if not candidates:
            print(
                f"Refinement {iteration}: bracket [{lo:.2f}, {hi:.2f}] pA "
                "already resolved; stopping early."
            )
            break
        print(f"Refinement {iteration}: honing within [{lo:.2f}, {hi:.2f}] pA")
        refined = await sweep_amplitudes(model, candidates, process_pool)
        measured.update(refined)

    return sorted(measured.items())


def select_amplitude(sweep: list[tuple[float, float]]) -> float:
    """Pick the amplitude whose rate is closest to the middle of the target band."""
    in_band = [
        (amp, rate) for amp, rate in sweep if TARGET_LOW_HZ <= rate <= TARGET_HIGH_HZ
    ]
    pool = in_band if in_band else sweep
    amp_best, rate_best = min(pool, key=lambda ar: abs(ar[1] - TARGET_MID_HZ))
    if not in_band:
        print(
            f"WARNING: no amplitude landed in {TARGET_LOW_HZ}-{TARGET_HIGH_HZ} Hz; "
            f"using closest ({amp_best} pA -> {rate_best:.2f} Hz). "
            "Widen CANDIDATE_AMPS_PA and re-run."
        )
    return amp_best


async def run_tuned_simulation(
    elektro: Elektro,
    model: NeuronModel,
    amp_pa: float,
    process_pool: ProcessPoolExecutor,
) -> PublishedRun:
    """Final, full-length, published run at the tuned amplitude (clamp + sine + noise)."""
    return await arun_simulation(
        elektro,
        model=model,
        duration=FINAL_DURATION,
        records=[
            VRecord(cell="cell_1", location="soma", position=0.5),
            VRecord(cell="cell_1", location="distalAxon", position=0.5),
        ],
        stims=build_stimuli(amp_pa),
        dt=0.05 * ms,  # 20 kHz sampling
        process_pool=process_pool,
    )


async def tune_and_run(
    elektro: Elektro,
    model: NeuronModel,
    process_pool: ProcessPoolExecutor,
    candidate_amps: list[float],
) -> PublishedRun:
    """Tune this cell's drive into the 4-5 Hz band, then publish the long final run."""
    print(f"\n=== Tuning {model.name} ===")
    sweep = await tune_amplitude(model, process_pool, candidate_amps)
    amp_best = select_amplitude(sweep)
    print(f"{model.name}: selected {amp_best} pA")
    return await run_tuned_simulation(elektro, model, amp_best, process_pool)


def _safe_filename(name: str) -> str:
    """Filesystem-safe version of ``name`` (alnum and underscores only)."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _whole_dataset_anchor(elektro: Elektro, dataset: ArrayDataset):
    """The anchor pinned to the whole dataset: where it was recorded/injected, its unit."""
    anchors = elektro.get_array_dataset_anchors(dataset.id).anchors
    return next((a for a in anchors if not a.coordinates), None)


def _write_trace(
    zip_file: zipfile.ZipFile,
    elektro: Elektro,
    dataset: ArrayDataset,
    time_trace_data: np.ndarray,
    subfolder: str,
    prefix: str,
    value_header: str,
) -> None:
    """One trace as ``<subfolder>/<base>.csv`` (time, value) plus its anchor metadata JSON."""
    anchor = _whole_dataset_anchor(elektro, dataset)
    site = anchor and (anchor.recording_site or anchor.stimulus_site)
    base = _safe_filename(
        f"{prefix}_{dataset.id}_{site.location if site else dataset.name}"
    )

    zip_file.writestr(
        f"{subfolder}/{base}.json",
        json.dumps(
            {
                "dataset": dataset.model_dump(mode="json"),
                "anchor": anchor.model_dump(mode="json") if anchor else None,
            },
            indent=2,
        ),
    )

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Time", value_header])
    for t, value in zip(time_trace_data, np.asarray(dataset.data.compute())):
        writer.writerow([t, value])
    zip_file.writestr(f"{subfolder}/{base}.csv", csv_buffer.getvalue())


def export_simulations_to_zip(
    elektro: Elektro,
    runs: list[tuple[NeuronModel, PublishedRun]],
    local_path: str = "experiment_data.zip",
) -> str:
    """Bundle several published runs into a zip, one base folder per labeled model.

    Each run gets its own top-level folder named after its model (e.g.
    "Normal_AIS_4_5_Hz" / "Short_AIS_4_5_Hz"). Inside that folder live the shared
    ``time_trace.csv`` (ms, on the run's clock), a ``metadata.json`` describing the
    model and clock, and a ``recordings/`` and ``stimuli/`` subfolder holding one
    CSV (and a metadata JSON) per trace. The zip is written to ``local_path`` and
    the path returned.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        seen: dict[str, int] = {}
        for model, run in runs:
            # Base folder per labeled model, de-duplicated if names collide.
            folder = _safe_filename(model.name)
            seen[folder] = seen.get(folder, 0) + 1
            if seen[folder] > 1:
                folder = f"{folder}_{seen[folder]}"

            # Every trace of a run shares one sample grid, timed onto the run's clock.
            first = run.datasets[0]
            assert first.intrinsic_system is not None, "dataset has no sample grid"
            time_trace_data = first.intrinsic_system.axis_values_in(
                run.clock, "t", first.shape[0], elektro=elektro
            )

            # Model/clock metadata for this folder.
            zip_file.writestr(
                f"{folder}/metadata.json",
                json.dumps(
                    {
                        "model": model.model_dump(mode="json"),
                        "clock": run.clock.model_dump(mode="json"),
                    },
                    indent=2,
                ),
            )

            # Shared time trace for this model.
            time_buffer = io.StringIO()
            time_writer = csv.writer(time_buffer)
            time_writer.writerow(["Time"])
            for t in time_trace_data:
                time_writer.writerow([t])
            zip_file.writestr(f"{folder}/time_trace.csv", time_buffer.getvalue())

            for stimulus in run.stimuli:
                _write_trace(
                    zip_file,
                    elektro,
                    stimulus,
                    time_trace_data,
                    f"{folder}/stimuli",
                    "stimulus",
                    "Stimulus Value",
                )

            for recording in run.recordings:
                _write_trace(
                    zip_file,
                    elektro,
                    recording,
                    time_trace_data,
                    f"{folder}/recordings",
                    "recording",
                    "Recording Value",
                )

    # Save the export locally on disk.
    with open(local_path, "wb") as f:
        f.write(zip_buffer.getvalue())
    print(f"Wrote local trace export to {local_path}")

    return local_path


if __name__ == "__main__":
    with easy("nico", elektro_service) as elektro:
        # Build and register the mechanism environment.
        zip_file, mechanisms_schema = build_and_zip_environment(
            directory_path="models",
        )
        print("Zip file created at:", zip_file)

        environment = elektro.create_mod_environment(
            name="test_env", zip_file=zip_file, mechanisms=mechanisms_schema
        )

        # Two cells differing only in AIS length -- each tuned to 4-5 Hz separately
        # since their rheobase differs.
        model_ais = build_neuron_model(
            elektro,
            environment=environment,
            ih_modifier=0.001,
            ais_len=28.6,
            synapse_count=4,
            title="Normal AIS 4-5 Hz",
            description="Normal AIS cell tuned just above rheobase for 4-5 Hz tonic firing",
        )

        print(model_ais)

        model_no_ais = build_neuron_model(
            elektro,
            environment=environment,
            ih_modifier=0.001,
            ais_len=0.000001,
            synapse_count=4,
            title="Short AIS 4-5 Hz",
            description="Short/absent AIS cell tuned just above rheobase for 4-5 Hz tonic firing",
        )

        elektro.create_model_collection(
            models=[model_ais, model_no_ais], name="AIS vs No-AIS 4-5 Hz Collection"
        )

        process_pool = ProcessPoolExecutor(max_workers=8)

        # Tune each cell independently into the band, then publish its 10 s run.
        # The intact-AIS cell is more excitable, so it sweeps the lower range;
        # the short-AIS cell needs higher amplitudes for the same rate.
        sim_ais = unkoil(
            tune_and_run, elektro, model_ais, process_pool, CANDIDATE_AMPS_PA_AIS
        )
        sim_no_ais = unkoil(
            tune_and_run, elektro, model_no_ais, process_pool, CANDIDATE_AMPS_PA_NO_AIS
        )

        # Combine both 10 s runs into a single comparison experiment.
        experiment = unkoil(
            acombine, elektro, [sim_ais, sim_no_ais], name="AIS vs No-AIS @ 4-5 Hz"
        )
        print("Published comparison experiment:", experiment.id)

        # Export all traces locally as a zip, one base folder per labeled model.
        exported_path = export_simulations_to_zip(
            elektro,
            [(model_ais, sim_ais), (model_no_ais, sim_no_ais)],
            local_path="nico_4_5hz_traces.zip",
        )
        print("Exported traces to:", exported_path)
