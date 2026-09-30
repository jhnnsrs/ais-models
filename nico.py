from concurrent.futures import ProcessPoolExecutor
from elektro import Elektro, elektro_service
from elektro.neuron.parse import build_and_zip_environment
from koil import unkoil
from elektro.api.schema import (
    Experiment,
    ModEnvironment,
    NeuronModel,
    ModelConfigInput,
    CellInput,
    TopologyInput,
    SectionInput,
    ConnectionInput,
)
from arkitekt import easy
from elektro.neuron.simulate import (
    PublishedRun,
    SineWaveStimulus,
    WhiteNoiseStimulus,
    arun_simulation,
    VRecord,
    CurrentClampStimulus,
)
from from_excel import load_biophysics_from_excel
from kanne.units import mV, um, ms, pA, s, Hz

# Owrite with cordless
CELL_TOPOLOGY = TopologyInput(
    sections=[
        SectionInput(
            category="soma",
            id="soma",
            nseg=25,
            diam=30 * um,
            length=30 * um,
            # Soma is the one section with cm != 1 in the biophysics sheet.
            cm="1.1 uF/cm^2",
            parent=None,
        ),
        SectionInput(
            category="dend",
            id="dendrite",
            nseg=10,
            diam=1.0 * um,
            length=120 * um,
            parent=ConnectionInput(parent="soma", parent_location=0),
        ),
        SectionInput(
            category="dend",
            id="apicalDend1",
            nseg=10,
            diam=1.0 * um,
            length=120 * um,
            parent=ConnectionInput(parent="dendrite", parent_location=1),
        ),
        SectionInput(
            category="dend",
            id="apicalDend2",
            nseg=10,
            diam=1.0 * um,
            length=120 * um,
            parent=ConnectionInput(parent="dendrite", parent_location=1),
        ),
        SectionInput(
            category="dend",
            id="apicalDend3",
            nseg=10,
            diam=1.0 * um,
            length=120 * um,
            parent=ConnectionInput(parent="dendrite", parent_location=1),
        ),
        SectionInput(
            category="dend",
            id="basalDend1",
            nseg=10,
            diam=1.0 * um,
            length=120 * um,
            parent=ConnectionInput(parent="soma", parent_location=0.7),
        ),
        SectionInput(
            category="AIS_dist",
            id="ais_dist",
            nseg=40,
            diam=1.22 * um,
            length=30 * um,
            parent=ConnectionInput(parent="soma", parent_location=1.0),
        ),
        SectionInput(
            category="axon",
            id="distalAxon",
            nseg=30,
            diam=1.0 * um,
            length=300 * um,
            parent=ConnectionInput(parent="ais_dist", parent_location=1.0),
        ),
    ]
)

CELL_BIO = load_biophysics_from_excel("biophysics_simple_fork_210330_v1.xlsx")


def build_neuron_model(
    elektro: Elektro,
    environment: ModEnvironment,
    ih_modifier: float,
    ais_len: float,
    synapse_count: int,
    title: str,
    description: str,
) -> NeuronModel:
    topo = CELL_TOPOLOGY.model_copy(deep=True)
    topo.model_config["frozen"] = False
    topo.model_config["validate_assignment"] = True
    topo.get_section_for_id("ais_dist").length = ais_len * um

    biophysics = CELL_BIO.model_copy(deep=True)

    biophysics.model_config["frozen"] = False
    biophysics.model_config["validate_assignment"] = True

    # These are unit-bearing quantities (validated against the mod environment),
    # so overrides have to carry the mechanism's NEURON unit — see
    # SECTION_PARAM_UNITS in from_excel.py. hNa's gbar is in mS/cm2; the inaT/ikdT/
    # imZ conductances are in S/cm2.
    biophysics.get_compartment_for_id("soma").get_section_param_for_id(
        "gbar_hNa"
    ).distribution.value = f"{ih_modifier} mS/cm2"

    biophysics.get_compartment_for_id("AIS_dist").get_section_param_for_id(
        "gnabar_inaT"
    ).distribution.value = "5 S/cm2"
    biophysics.get_compartment_for_id("AIS_dist").get_section_param_for_id(
        "gkbar_ikdT"
    ).distribution.value = "0.000001 S/cm2"
    biophysics.get_compartment_for_id("AIS_dist").get_section_param_for_id(
        "gkbar_imZ"
    ).distribution.value = "0.00001 S/cm2"

    print(
        biophysics.get_compartment_for_id("AIS_dist")
        .get_section_param_for_id("gnabar_inaT")
        .distribution.value
    )

    config = ModelConfigInput(
        cells=[
            CellInput(
                id="cell_1",
                biophysics=biophysics,
                topology=topo,
            )
        ],
        netSynapses=[],
        netStimulators=[],
        netConnections=[],
        # Axial resistivity (former per-compartment "Ra" global, uniform at
        # 200 Ω·cm in the sheet) is now a model-wide cable default.
        ra="200 ohm*cm",
        vInit=-70 * mV,
        temperature="37 °C",
    )

    model = elektro.create_neuronmodel(
        environment=environment,
        name=title,
        config=config,
        description=description,
    )
    return model


async def calculate_ih_sag_from_model(
    elektro: Elektro,
    model: NeuronModel,
) -> PublishedRun:
    """
    Calculate the ih sag from the model.
    """

    process_pool = ProcessPoolExecutor(max_workers=8)

    simulation = await arun_simulation(
        elektro,
        model=model,
        duration=1 * s,
        records=[
            VRecord(
                cell="cell_1",
                location="distalAxon",
                position=0.5,
            )
        ],
        stims=[
            CurrentClampStimulus(
                cell="cell_1",
                location="soma",
                position=0.5,
                amp=100 * pA,
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
        ],
        dt=0.05 * ms,  # 20 kHz sampling
        process_pool=process_pool,
    )

    # inspect(s)
    return simulation


async def calculate_from_aisless_from_model(
    elektro: Elektro,
    model: NeuronModel,
) -> PublishedRun:
    """
    Calculate the ih sag from the model.
    """

    process_pool = ProcessPoolExecutor(max_workers=8)

    simulation = await arun_simulation(
        elektro,
        model=model,
        duration=1 * s,
        records=[
            VRecord(
                cell="cell_1",
                location="distalAxon",
                position=0.6,
            )
        ],
        stims=[
            CurrentClampStimulus(
                cell="cell_1",
                location="soma",
                position=0.5,
                amp=230 * pA,
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
        ],
        dt=0.05 * ms,  # 20 kHz sampling
        process_pool=process_pool,
    )

    return simulation


async def acombine(
    elektro: Elektro, runs: list[PublishedRun], name="Combined Experiment"
) -> Experiment:
    """
    Combine several published runs into one experiment, overlaid on a shared clock.

    Every run is published on its own clock starting at t=0, so the experiment
    adopts the first run's clock as its world and every other run's clock is
    placed onto it with no offset. Each recording and stimulus becomes a trace layer.
    """

    world = runs[0].clock
    combined = await elektro.acreate_experiment(
        name=name,
        description="Combined Experiment",
        coordinate_system=world.id,
    )

    for run in runs[1:]:
        await elektro.acreate_clock_offset(
            clock=run.clock.id, onto=world.id, offset=0 * s, drift_ppm=0.0
        )

    for run in runs:
        for dataset in run.datasets:
            await elektro.acreate_trace_layer(
                experiment=combined.id, dataset=dataset.id, name=dataset.name
            )

    return combined


if __name__ == "__main__":
    with easy("nico", elektro_service) as elektro:
        # find_ih_modifier()

        # Build the schema
        zip_file, mechanisms_schema = build_and_zip_environment(
            directory_path="models",
        )
        print("Zip file created at:", zip_file)

        environment = elektro.create_mod_environment(
            name="test_env", zip_file=zip_file, mechanisms=mechanisms_schema
        )

        model0 = build_neuron_model(
            elektro,
            environment=environment,
            ih_modifier=0.021,
            ais_len=28.6,
            synapse_count=4,
            title="Normal AIS Model GG",
            description="Model with normal AIS length",
        )

        answer1 = unkoil(calculate_ih_sag_from_model, elektro, model=model0)

        print(answer1.stimuli[0].data.compute())

        model1 = build_neuron_model(
            elektro,
            environment=environment,
            ih_modifier=0.021,
            ais_len=0.000001,
            synapse_count=4,
            title="Short AIS Model",
            description="Model with very short AIS length",
        )

        elektro.create_model_collection(
            models=[model0, model1], name="Short AIS Model Collection"
        )

        answer = unkoil(calculate_from_aisless_from_model, elektro, model=model1)

        unkoil(acombine, elektro, [answer1, answer], name="AIS vs No AIS Experiment")
