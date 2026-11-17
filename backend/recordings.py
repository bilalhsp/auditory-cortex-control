"""Discover recordings using the existing dataset catalog and MAT trial metadata.

No generator, feature extractor, model, or scientific package initializer is
imported. Paths remain private to this filesystem boundary.
"""

from pathlib import Path
from typing import List, Literal, Tuple
from uuid import NAMESPACE_URL, uuid5

import yaml

from backend.config import Settings
from backend.models import Recording, RecordingCatalogResult


class CatalogUnavailable(Exception):
    pass


class RecordingCatalog:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def _resolved_path(path: Path) -> Path:
        try:
            return path.resolve()
        except RuntimeError as error:
            # Python 3.8 reports cyclic filesystem links as RuntimeError.
            raise ValueError("Cannot resolve recording location") from error

    def _locations(self) -> Tuple[Path, Path]:
        repository = Path(__file__).resolve().parents[1]
        data = self.settings.neural_data_dir
        if data is None:
            config = yaml.safe_load((repository / "auditory_cortex/config.yml").read_text())
            data = Path(config["neural_data_dir"])
        metadata = self.settings.recording_metadata_path or (
            repository / "auditory_cortex/neural_data/ucdavis_active/configs/ucdavisBAK_config.yml"
        )
        return data, metadata

    @staticmethod
    def _stimulus_families(path: Path, repeats: int) -> List[Literal["TIMIT", "mVocs"]]:
        import numpy as np
        from scipy.io import loadmat
        from scipy.io.matlab import MatReadError

        # This is the legacy dataset's trial-name/type convention. Only load
        # TrialStimData, not neural spikes, feature caches, or model weights.
        try:
            trials = loadmat(str(path), variable_names=["TrialStimData"],
                             squeeze_me=True, struct_as_record=False)["TrialStimData"]
        except (MatReadError, IndexError) as error:
            raise ValueError("Unreadable trial metadata") from error
        found = set()
        allowed = {"S1T", "BMA3", "BMT{}".format(repeats), "BMM{}".format(repeats)}
        for trial in np.atleast_1d(trials).flat:
            names = np.atleast_1d(trial.StimulusName).ravel()
            types = np.atleast_1d(trial.StimulusType).ravel()
            if len(names) != len(types):
                raise ValueError("Trial metadata lengths differ")
            for name, kind in zip(names, types):
                if str(kind) not in allowed:  # Includes exclusion of S2 targets.
                    continue
                parts = [part for part in str(name).replace("\\", "/").split("/") if part]
                if len(parts) >= 2:
                    found.add(parts[-2].lower())
        result: List[Literal["TIMIT", "mVocs"]] = []
        if "timit" in found:
            result.append("TIMIT")
        if "mvox" in found:
            result.append("mVocs")
        return result

    def list(self, dataset: str) -> RecordingCatalogResult:
        try:
            data, metadata = self._locations()
            directory = self._resolved_path(data / dataset / "Data")
            if not directory.is_dir():
                raise ValueError("Recording directory unavailable")
            rows = yaml.safe_load(metadata.read_text())["sessions_info"]
            if not isinstance(rows, list):
                raise ValueError("Invalid catalog")
            result = RecordingCatalogResult(recordings=[])
            identities = set()
            for row in rows:
                session, name, repeats = row["id"], row["session"], row["num_repeats"]
                if (not isinstance(session, int) or isinstance(session, bool)
                        or not isinstance(name, str) or Path(name).name != name
                        or Path(name).suffix.lower() != ".mat" or session in identities
                        or not isinstance(repeats, int) or repeats < 1):
                    raise ValueError("Invalid catalog entry")
                identities.add(session)
                try:
                    path = self._resolved_path(directory / name)
                    path.relative_to(directory)
                    families = self._stimulus_families(path, repeats)
                except (OSError, ValueError, TypeError, KeyError, AttributeError, NotImplementedError):
                    result.unavailable_count += 1
                    continue
                if not families:
                    continue
                result.recordings.append(Recording(
                    id=str(uuid5(NAMESPACE_URL, "auditory-cortex:existing:{}:{}:{}".format(dataset, session, name))),
                    display_name=Path(name).stem.replace("_", " · "),
                    dataset_name=dataset, stimulus_families=families,
                ))
            return result
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
            raise CatalogUnavailable("Recording catalog is unavailable. Check the server's recording configuration and access permissions.") from error
