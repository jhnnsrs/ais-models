"""Composable arkitekt_next app for the AIS neuron-model experiment.

This exposes the logic from ``nico.py`` as reusable, composable arkitekt nodes
(see ``app.py`` for the simplest template of this pattern):

  * ``create_ais_model``       -> NeuronModel  (cell with an intact AIS)
  * ``create_non_ais_model``   -> NeuronModel  (cell with a short / absent AIS)
  * ``simulate_to_frequency``  -> Simulation   (tunes ONE model's drive to a target
                                  firing rate, then runs it -- "frequency mode")
  * ``simulate``               -> Experiment   (runs a *list* of models through the
                                  current-clamp + sine + white-noise pipeline and
                                  combines every run into one comparison experiment)

The model builders return ``NeuronModel`` structures and ``simulate`` consumes a
``list[NeuronModel]``, so they chain together on the platform: create a couple of
models, drop them into ``simulate``, get a published ``Experiment`` back.

Run ``python nico_app.py`` to serve the nodes to arkitekt.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
from arkitekt_next import aprogress, easy, register

from elektro.neuron.parse import build_and_zip_environment
from elektro.api.schema import (
    Experiment,
    ModEnvironment,
    NeuronModel,
    Simulation,
    acreate_model_collection,
    create_mod_environment,
    create_model_collection,
)
from elektro.neuron.simulate import (
    CurrentClampStimulus,
    SineWaveStimulus,
    WhiteNoiseStimulus,
    VRecord,
    arun_simulation,
    asimulate,
)

# Reuse the cell geometry / biophysics builder and the simulation combiner so the
# composable app stays in lock-step with nico.py -- only the wiring is new here.
from nico import build_neuron_model, acombine

# Reuse the pure firing-rate helpers from the 4-5 Hz tuning script.
from nico_4_5hz import count_firing_rate, bracket_for_target

# Units come from the shared kanne registry (same one nico.py uses) so every
# quantity lives on one registry -- mixing in a private pint.UnitRegistry() makes
# the scalar coercion re-parse foreign units and risks registry mismatches.
from kanne.units import pA, ms, s, Hz


# --- shared mechanism environment ------------------------------------------


@lru_cache(maxsize=1)
def get_environment() -> ModEnvironment:
    """Build, zip and publish the NEURON mechanism environment once.

    ``ModEnvironment`` is not an arkitekt structure, so it cannot be passed
    between nodes as an argument. Instead every model-builder pulls this cached,
    already-published environment -- it is built (and uploaded) lazily on the
    first model creation and reused thereafter.
    """
    zip_file, mechanisms = build_and_zip_environment(directory_path="models")
    return create_mod_environment(
        name="nico_env", zip_file=zip_file, mechanisms=mechanisms
    )


# --- model creation nodes --------------------------------------------------


@register
def create_ais_model(
    title: str = "Normal AIS Model",
    description: str = "Model with a normal-length axon initial segment",
    ais_len: float = 28.6,
    ih_modifier: float = 0.001,
    synapse_count: int = 4,
) -> NeuronModel:
    """Create a neuron model with an intact axon initial segment (AIS).

    Builds the standard fork-cell topology / biophysics from nico.py and sets the
    AIS to a normal physiological length.

    Args:
        title: Name of the published model.
        description: Human-readable description of the model.
        ais_len: Length (um) of the distal AIS section.
        ih_modifier: Somatic ``gbar_hNa`` conductance value.
        synapse_count: Number of synapses (passed through to the builder).

    Returns:
        NeuronModel: The published neuron model.
    """
    return build_neuron_model(
        environment=get_environment(),
        ih_modifier=ih_modifier,
        ais_len=ais_len,
        synapse_count=synapse_count,
        title=title,
        description=description,
    )


@register
def create_non_ais_model_karlos(
    title: str = "Short AIS Model",
    description: str = "Model with a very short / absent axon initial segment",
    ais_len: float = 0.000001,
    ih_modifier: float = 0.001,
    synapse_count: int = 4,
) -> NeuronModel:
    """Create a neuron model with a short / absent axon initial segment.

    Identical to :func:`create_ais_model` except the AIS is collapsed to a
    near-zero length, modelling a cell without a functional initial segment.

    Args:
        title: Name of the published model.
        description: Human-readable description of the model.
        ais_len: Length (um) of the distal AIS section (kept tiny by default).
        ih_modifier: Somatic ``gbar_hNa`` conductance value.
        synapse_count: Number of synapses (passed through to the builder).

    Returns:
        NeuronModel: The published neuron model.
    """
    return build_neuron_model(
        environment=get_environment(),
        ih_modifier=ih_modifier,
        ais_len=ais_len,
        synapse_count=synapse_count,
        title=title,
        description=description,
    )


# --- experiment pipeline ---------------------------------------------------


def build_stimuli(amp_pa: float) -> list:
    """Sustained somatic current clamp plus the shared sine and white-noise drive.

    This is the same stimulus triplet used throughout nico.py: a step current to
    set the operating point, white noise for trial-to-trial variability, and a
    high-frequency sine to probe the cell.
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
            noise_level=20 * pA,
        ),
        SineWaveStimulus(
            cell="cell_1",
            location="soma",
            position=0.5,
            amplitude=80 * pA,
            frequency=200 * Hz,
        ),
    ]


async def simulate_model(
    model: NeuronModel,
    amp_pa: float,
    duration_s: float,
    process_pool: ProcessPoolExecutor,
) -> Simulation:
    """Run one model through the standard stimulus, recording soma + distal axon."""
    return await arun_simulation(
        model=model,
        duration=duration_s * s,
        records=[
            VRecord(cell="cell_1", location="soma", position=0.5),
            VRecord(cell="cell_1", location="distalAxon", position=0.5),
        ],
        stims=build_stimuli(amp_pa),
        dt=0.05 * ms,  # 20 kHz sampling
        process_pool=process_pool,
    )


async def _sweep_rates(
    model: NeuronModel,
    amps: list[float],
    duration_s: float,
    process_pool: ProcessPoolExecutor,
) -> list[tuple[float, float]]:
    """Run one short, unpublished simulation per amplitude; return (amp_pA, rate_Hz)."""
    results: list[tuple[float, float]] = []
    for amp in amps:
        sim = await asimulate(
            model=model,
            duration=duration_s * s,
            records=[VRecord(cell="cell_1", location="soma", position=0.5)],
            stims=build_stimuli(amp),
            dt=0.05 * ms,  # 20 kHz sampling
            process_pool=process_pool,
        )
        rate = count_firing_rate(sim.time_trace, sim.recordings[0].trace.value)
        results.append((float(amp), rate))
    return results


@register
async def simulate_to_frequency(
    model: NeuronModel,
    target_frequency: float = 4.5,
    tolerance: float = 0.5,
    min_amplitude: float = 20.0,
    max_amplitude: float = 380.0,
    sweep_points: int = 8,
    refine_iterations: int = 2,
    refine_points: int = 4,
    sweep_duration: float = 2.0,
    final_duration: float = 10.0,
) -> Simulation:
    """Tune a single model's drive to a target firing rate, then run it.

    "Frequency mode" for one neuron: the same sine + white-noise drive is layered
    on a sustained current clamp whose amplitude is *tuned* so the cell's somatic
    firing rate lands at ``target_frequency``. We sweep amplitudes coarsely, then
    bracket and refine around the target (firing rate rises monotonically with
    amplitude above rheobase), pick the closest match, and publish a final
    long-duration run at that amplitude.

    Because the white noise makes firing non-deterministic, this tunes the *mean*
    rate rather than an exact spike count.

    Args:
        model: The neuron model to tune and simulate.
        target_frequency: Desired mean somatic firing rate in Hz.
        tolerance: Acceptable +/- band around the target, in Hz.
        min_amplitude: Lowest current-clamp amplitude to sweep, in pA.
        max_amplitude: Highest current-clamp amplitude to sweep, in pA.
        sweep_points: Number of amplitudes in the initial coarse sweep.
        refine_iterations: How many times to zoom into the bracket around target.
        refine_points: Interior amplitudes sampled per refinement pass.
        sweep_duration: Duration (s) of each short tuning simulation.
        final_duration: Duration (s) of the final published run.

    Returns:
        Simulation: The final tuned run (soma + distal axon recordings).
    """
    with ProcessPoolExecutor(max_workers=8) as process_pool:
        candidates = [
            round(float(a), 2)
            for a in np.linspace(min_amplitude, max_amplitude, sweep_points)
        ]
        await aprogress(5, f"Coarse sweep over {len(candidates)} amplitudes")
        measured: dict[float, float] = dict(
            await _sweep_rates(model, candidates, sweep_duration, process_pool)
        )

        for iteration in range(1, refine_iterations + 1):
            lo, hi = bracket_for_target(list(measured.items()), target_frequency)
            interior = [
                round(lo + (hi - lo) * k / (refine_points + 1), 2)
                for k in range(1, refine_points + 1)
            ]
            interior = [a for a in interior if a not in measured]
            if not interior:
                break
            await aprogress(
                20 + iteration * 20,
                f"Refinement {iteration}: honing within [{lo:.1f}, {hi:.1f}] pA",
            )
            measured.update(
                await _sweep_rates(model, interior, sweep_duration, process_pool)
            )

        sweep = sorted(measured.items())
        in_band = [
            (amp, rate)
            for amp, rate in sweep
            if abs(rate - target_frequency) <= tolerance
        ]
        pool = in_band if in_band else sweep
        amp_best, rate_best = min(pool, key=lambda ar: abs(ar[1] - target_frequency))
        await aprogress(
            80,
            f"Tuned to {amp_best:.1f} pA (~{rate_best:.2f} Hz); running final {final_duration}s sim",
        )

        return await simulate_model(model, amp_best, final_duration, process_pool)


@register
async def simulate(
    models: list[NeuronModel],
    name: str = "AIS Comparison Experiment",
    amplitude: float = 150.0,
    duration: float = 1.0,
    collect: bool = True,
) -> Experiment:
    """Run a list of neuron models through the experiment pipeline.

    Every model is driven with the same current-clamp + sine + white-noise
    stimulus, voltage is recorded at the soma and the distal axon, and all runs
    are combined into a single comparison ``Experiment``. Optionally the input
    models are also grouped into a ``ModelCollection``.

    Args:
        models: The neuron models to simulate and compare (e.g. an AIS and a
            non-AIS model from the create nodes above).
        name: Name of the published comparison experiment.
        amplitude: Sustained current-clamp amplitude in pA, applied to every
            model.
        duration: Simulation duration in seconds.
        collect: If True, also publish a ModelCollection grouping the models.

    Returns:
        Experiment: The combined comparison experiment.
    """
    if not models:
        raise ValueError("simulate() needs at least one model")

    with ProcessPoolExecutor(max_workers=8) as process_pool:
        simulations: list[Simulation] = []
        for i, model in enumerate(models):
            await aprogress(int(100 * i / len(models)), f"Simulating {model.name}")
            simulations.append(
                await simulate_model(model, amplitude, duration, process_pool)
            )

    if collect:
        await acreate_model_collection(models=models, name=f"{name} Collection")

    await aprogress(100, "Combining runs into experiment")
    return await acombine(simulations, name=name)


if __name__ == "__main__":
    with easy("nicoff") as e:
        e.rekuest.run(force=True)
