from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class ExperimentConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    stimulus_family: Literal["TIMIT", "mVocs"] = "TIMIT"
    dataset_name: Literal["ucdavisBAK"] = "ucdavisBAK"
    model_name: Literal["deepspeech2"] = "deepspeech2"
    layer_id: Literal[2] = 2
    bin_width_ms: Literal[50] = 50
    receptive_field_ms: Literal[200] = 200
    gpu_count: StrictInt = Field(ge=0)


class Recording(BaseModel):
    id: str
    display_name: str
    dataset_name: str
    source: Literal["existing", "upload"] = "existing"
    stimulus_families: List[Literal["TIMIT", "mVocs"]]


class RecordingCatalogResult(BaseModel):
    recordings: List[Recording]
    unavailable_count: int = 0


class RecordingSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording_id: UUID


class Experiment(BaseModel):
    id: str
    created_at: str
    configuration: ExperimentConfiguration
    recording: Optional[Recording] = None
    recording_revision: int = 0


class UploadResult(BaseModel):
    validation_status: Literal["valid"] = "valid"
    experiment: Experiment
