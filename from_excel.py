from elektro.api.schema import (
    BiophysicsInput,
    SectionParamMapInput,
    GlobalParamMapInput,
    CompartmentInput,
)
import pandas as pd


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

            section_params.append(
                SectionParamMapInput(
                    mechanism=mechanism,
                    param=param_name,
                    value=param_value,
                    description=param_description,
                )
            )

        global_frame = df.iloc[
            [1, 3, 4, 6, 7]
        ]  # TODO: WHy are we omiting ##? we dont know because neuron is shit
        globalframe = global_frame.iloc[:, [0, 1, index]]

        global_params = []

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

            if param_name == "e_pas":
                param_value = -68  # TODO: WHY THIS FREAKING OVERWRITE?

            global_params.append(
                GlobalParamMapInput(
                    param=param_name, value=param_value, description=param_description
                )
            )

        compartments.append(
            CompartmentInput(
                mechanisms=mechanisms,
                id=comp,
                section_params=section_params,
                global_params=global_params,
            )
        )

    return BiophysicsInput(compartments=compartments)
