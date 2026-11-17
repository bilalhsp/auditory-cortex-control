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


class FitSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    recording_id: UUID
    recording_revision: StrictInt = Field(ge=1)
    seed: StrictInt = Field(default=42, ge=0, le=4294967295)


class ChannelCorrelation(BaseModel):
    channel_id: str
    correlation: Optional[float] = None


class Fit(BaseModel):
    id: str
    job_id: str
    experiment_id: str
    recording: Recording
    recording_revision: int
    configuration: ExperimentConfiguration
    seed: int
    state: Literal["queued", "running", "succeeded", "failed", "interrupted"] = "queued"
    stage: str = "queued"
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    artifacts_ready: bool = False
    correlations: List[ChannelCorrelation] = Field(default_factory=list)
    best_channel: Optional[ChannelCorrelation] = None
