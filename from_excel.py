from elektro.api.schema import (
    BiophysicsInput,
    DistributionInput,
    SectionParamMapInput,
    IonInput,
    CompartmentInput,
)
import pandas as pd


# NEURON range-variable units, taken verbatim from each mechanism's PARAMETER
# block in the .mod files under ./environment (the "mod environment" the server
# validates against). The Excel stores bare numbers in each mechanism's native
# unit, so the loader has to re-attach that unit before the value can validate.
#
# This has to be an explicit, mechanism-by-mechanism map rather than a heuristic
# (e.g. "anything starting with g is S/cm2"): gbar_hNa is declared in mmho/cm2
# (mS/cm2), a 1000x different scale from the mho/cm2 (== S/cm2) used by every
# other conductance. Getting that wrong silently is exactly the blocking bug.
SECTION_PARAM_UNITS = {
    # Maximal conductances — .mod declares (mho/cm2), i.e. S/cm2.
    "gnabar_inaT": "S/cm2",
    "gkbar_ikdT": "S/cm2",
    "gkbar_imZ": "S/cm2",
    "g_pas": "S/cm2",  # NEURON built-in pas leak conductance
    "gnabar_hha_old": "S/cm2",
    "gkbar_hha_old": "S/cm2",
    "gkabar_kap": "S/cm2",
    "gkabar_kad": "S/cm2",
    "gcalbar_cal": "S/cm2",
    "gcalbar_calH": "S/cm2",
    "gcatbar_cat": "S/cm2",
    "gcabar_somacar": "S/cm2",
    "gcabar_car": "S/cm2",
    "gbar_kca": "S/cm2",
    "gkbar_mykca": "S/cm2",
    "gnabar_hha2": "S/cm2",
    "gkbar_hha2": "S/cm2",
    # hNa's gbar is declared in mmho/cm2 (mS/cm2) — NOT mho/cm2. See hNa.mod.
    "gbar_hNa": "mS/cm2",
    # Voltages / half-potentials / slopes / shifts — .mod declares (mV).
    "vtraub_inaT": "mV",
    "shift_inaT": "mV",
    "vtraub_ikdT": "mV",
    "el_hha_old": "mV",
    "K_hNa": "mV",
    "vhalf_hNa": "mV",
}


def load_biophysics_from_excel(infile: str) -> BiophysicsInput:
    df = pd.read_excel(infile, sheet_name="Sheet1")

    # %%

    df.iloc[8:33]
    # %%

    compartments = []

    for index, comp in enumerate(df.columns):
        if index < 2:
            continue

        mechanisms = df.iloc[0, index]
        mechanisms = [val.strip() for val in mechanisms.split(" ") if val != ""]
        mechanisms = set(mechanisms)

        subframe = df.iloc[8:33]
        subframe = subframe.iloc[:, [0, 1, index]]

        section_params = []

        for row in subframe.itertuples():
            param_name = row[1]
            param_value = row[3]
            param_description = row[2]

            if param_value == "-" or pd.isna(param_value):
                continue

            if isinstance(param_value, str):
                param_value = float(param_value)

            if pd.isna(param_description):
                param_description = None

            if param_name == "g_pas":
                param_value = 0.0002  # TODO: WHY THIS FREAKING OVERWRITE?

            mechanism = "_".join(param_name.split("_")[1:])

            unit = SECTION_PARAM_UNITS.get(param_name)
            if unit is None:
                raise ValueError(
                    f"No unit registered for section parameter {param_name!r}. "
                    f"Add it to SECTION_PARAM_UNITS using the unit from the "
                    f"mechanism's PARAMETER block in ./environment/{mechanism}.mod."
                )

            section_params.append(
                SectionParamMapInput(
                    mechanism=mechanism,
                    param=param_name,
                    # The scalar value now lives on a (uniform) distribution rather
                    # than directly on the param mapping, and must carry its NEURON
                    # unit so it validates against the mod environment.
                    distribution=DistributionInput(value=f"{param_value} {unit}"),
                    description=param_description,
                )
            )

        # The old schema carried a per-compartment "global" (non-mechanistic)
        # parameter block. That concept is gone: reversal potentials are now ion
        # settings, the leak reversal (e_pas) is just a `pas` section parameter, and
        # the cable properties (Ra, cm) live on the topology / model config (applied
        # in nico.py). We translate each former global row into its new home here.
        global_frame = df.iloc[
            [1, 3, 4, 6, 7]
        ]  # TODO: WHy are we omiting ##? we dont know because neuron is shit
        globalframe = global_frame.iloc[:, [0, 1, index]]

        # ena/ek -> per-compartment ion reversal potentials (NEURON e<ion>).
        ion_for_param = {"ena": "na", "ek": "k"}
        ions = []

        for row in globalframe.itertuples():
            param_name = row[1]
            param_value = row[3]
            param_description = row[2]

            if param_value == "-" or pd.isna(param_value):
                continue

            if isinstance(param_value, str):
                param_value = float(param_value)

            if pd.isna(param_description):
                param_description = None

            if param_name in ion_for_param:
                ions.append(
                    IonInput(
                        ion=ion_for_param[param_name],
                        reversal_potential=f"{param_value} mV",
                    )
                )
            elif param_name == "e_pas":
                # Leak reversal now rides on the `pas` mechanism (present in every
                # compartment). Pinned to -70 mV to match the original net value
                # (the old loader forced -68, then nico.py overwrote it to -70).
                section_params.append(
                    SectionParamMapInput(
                        mechanism="pas",
                        param="e_pas",
                        distribution=DistributionInput(value="-70 mV"),
                        description=param_description,
                    )
                )
            # Ra and cm are section-level cable properties in the new schema; they
            # are set on the topology / model config in nico.py, not here.

        compartments.append(
            CompartmentInput(
                mechanisms=mechanisms,
                id=comp,
                section_params=section_params,
                ions=ions,
            )
        )

    return BiophysicsInput(compartments=compartments)
