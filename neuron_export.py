"""Composable arkitekt_next app for exporting / importing whole neuron models.

A ``NeuronModel`` on the platform is only useful together with the NEURON
*mechanisms* (the compiled ``.mod`` sources) of its ``ModEnvironment`` -- the
config references mechanisms by name (``gbar_hNa``, ``inaT`` ...) and is
meaningless without them. This app bundles both into a single, self-contained
zip archive so a model can be moved between deployments (or archived) and later
re-created with its environment intact.

Nodes
-----
* ``export_neuron_model``  NeuronModel -> File   pack model + mechanisms into a zip
* ``import_neuron_model``   File -> NeuronModel   re-create environment + model from a zip

Archive layout (``*.neuronmodel.zip``)
--------------------------------------
    manifest.json       format id + version + the model / environment names
    model.json          {"name", "config"}  -- the full ModelConfig, unit-strings intact
    environment.json    {"name", "mechanisms"}  -- the parsed MechanismInput list
    mechanisms.zip       the ModEnvironment's original .mod source bundle

The config and mechanism blobs are just the pydantic input models dumped to
JSON (``by_alias``, ``mode="json"``) so the dimensionful quantities ("-70 mV",
"37 °C") round-trip as strings and re-validate on import without any manual
unit handling.

Run ``python neuron_export.py`` to serve the nodes to arkitekt.
"""

import json
import os
import tempfile
import zipfile

from arkitekt_next import aprogress, easy, register

from elektro.api.schema import (
    File,
    MechanismInput,
    ModelConfigInput,
    NeuronModel,
    ParameterInput,
    acreate_mod_environment,
    acreate_neuronmodel,
    afrom_file_like,
)

# Bump when the on-disk layout changes in a backwards-incompatible way.
ARCHIVE_FORMAT = "elektro.neuronmodel"
ARCHIVE_VERSION = 1


# --- (de)serialisation helpers ---------------------------------------------


def _mechanism_to_input(mechanism) -> MechanismInput:
    """Turn a fetched ``Mechanism`` back into the ``MechanismInput`` needed to
    re-create a ``ModEnvironment``.

    The stored mechanism only carries the parameter *ports* (key + kind), which
    is all ``createModEnvironment`` needs -- default values live in the ``.mod``
    sources that travel alongside in ``mechanisms.zip``.
    """
    return MechanismInput(
        name=mechanism.name,
        description=None,
        parameters=[
            ParameterInput(key=port.key, label=port.key, kind=port.kind)
            for port in mechanism.parameters
        ],
    )


def _write_archive(model: NeuronModel, mechanisms_zip: str, out_path: str) -> str:
    """Assemble the export zip from an already-fetched model and its downloaded
    mechanism bundle. Returns ``out_path``."""
    manifest = {
        "format": ARCHIVE_FORMAT,
        "version": ARCHIVE_VERSION,
        "model": model.name,
        "environment": model.environment.name,
    }
    # ``by_alias`` + ``mode="json"`` so unit scalars serialise to their pint
    # strings (e.g. "-70 mV") and the config re-validates verbatim on import.
    model_doc = {
        "name": model.name,
        "config": model.config.model_dump(by_alias=True, mode="json"),
    }
    environment_doc = {
        "name": model.environment.name,
        "mechanisms": [
            _mechanism_to_input(m).model_dump(by_alias=True, mode="json")
            for m in model.environment.mechanisms
        ],
    }

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("model.json", json.dumps(model_doc, indent=2))
        archive.writestr("environment.json", json.dumps(environment_doc, indent=2))
        archive.write(mechanisms_zip, "mechanisms.zip")

    return out_path


def _read_archive(archive_path: str, workdir: str):
    """Unpack an export zip. Returns ``(model_doc, environment_doc, mechanisms_zip_path)``."""
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        missing = {"model.json", "environment.json", "mechanisms.zip"} - names
        if missing:
            raise ValueError(
                f"{archive_path} is not a neuron-model archive (missing {', '.join(sorted(missing))})"
            )

        manifest = json.loads(archive.read("manifest.json")) if "manifest.json" in names else {}
        if manifest.get("format", ARCHIVE_FORMAT) != ARCHIVE_FORMAT:
            raise ValueError(f"Unknown archive format: {manifest.get('format')!r}")

        model_doc = json.loads(archive.read("model.json"))
        environment_doc = json.loads(archive.read("environment.json"))

        mechanisms_zip_path = os.path.join(workdir, "mechanisms.zip")
        with open(mechanisms_zip_path, "wb") as handle:
            handle.write(archive.read("mechanisms.zip"))

    return model_doc, environment_doc, mechanisms_zip_path


# --- nodes ------------------------------------------------------------------


@register
async def export_neuron_model(
    model: NeuronModel,
    name: str = "",
) -> File:
    """Export a neuron model together with its mechanisms into a single zip archive.

    Downloads the model's ``ModEnvironment`` mechanism bundle, packs it next to a
    JSON serialisation of the model config and environment metadata, and uploads
    the resulting ``*.neuronmodel.zip`` as a platform ``File`` (which can then be
    downloaded, archived, or fed straight into :func:`import_neuron_model`).

    Args:
        model: The neuron model to export.
        name: Filename for the exported archive. Defaults to the model name.

    Returns:
        File: The uploaded zip archive.
    """
    archive_name = (name or model.name or "neuron_model").strip()
    if not archive_name.endswith(".zip"):
        archive_name = f"{archive_name}.neuronmodel.zip"

    with tempfile.TemporaryDirectory() as workdir:
        await aprogress(10, f"Downloading mechanisms for '{model.environment.name}'")
        mechanisms_zip = await model.environment.store.adownload(
            file_name=os.path.join(workdir, "mechanisms.zip")
        )

        await aprogress(60, "Packing archive")
        archive_path = _write_archive(
            model, mechanisms_zip, os.path.join(workdir, archive_name)
        )

        await aprogress(80, "Uploading archive")
        file = await afrom_file_like(name=archive_name, file=archive_path)

    await aprogress(100, "Done")
    return file


@register
async def import_neuron_model(
    file: File,
    name: str = "",
) -> NeuronModel:
    """Import a neuron model from an archive produced by :func:`export_neuron_model`.

    Downloads the ``*.neuronmodel.zip``, re-creates the ``ModEnvironment`` from the
    bundled mechanisms, and re-creates the ``NeuronModel`` from the stored config.
    A fresh environment and model are created -- importing the same archive twice
    yields two independent models.

    Args:
        file: The exported zip archive (a File on the platform).
        name: Name for the imported model. Defaults to the name stored in the archive.

    Returns:
        NeuronModel: The newly created neuron model.
    """
    with tempfile.TemporaryDirectory() as workdir:
        await aprogress(10, "Downloading archive")
        archive_path = await file.store.adownload(
            file_name=os.path.join(workdir, "import.neuronmodel.zip")
        )

        await aprogress(30, "Reading archive")
        model_doc, environment_doc, mechanisms_zip = _read_archive(archive_path, workdir)

        mechanisms = [
            MechanismInput.model_validate(m) for m in environment_doc["mechanisms"]
        ]

        await aprogress(55, f"Re-creating environment '{environment_doc['name']}'")
        environment = await acreate_mod_environment(
            name=environment_doc["name"],
            zip_file=mechanisms_zip,
            mechanisms=mechanisms,
        )

        await aprogress(80, "Re-creating model")
        config = ModelConfigInput.model_validate(model_doc["config"])
        model = await acreate_neuronmodel(
            name=(name or model_doc["name"]).strip(),
            config=config,
            environment=environment.id,
        )

    await aprogress(100, "Done")
    return model


if __name__ == "__main__":
    with easy("neuron-export") as e:
        e.rekuest.run(force=True)
