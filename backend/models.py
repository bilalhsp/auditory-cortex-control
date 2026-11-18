from typing import List, Literal, Optional, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


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
    session_id: Optional[str] = None


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
    state: Literal["queued", "running", "succeeded", "failed", "interrupted", "cancelling", "cancelled"] = "queued"
    stage: str = "queued"
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    artifacts_ready: bool = False
    correlations: List[ChannelCorrelation] = Field(default_factory=list)
    best_channel: Optional[ChannelCorrelation] = None


GenerationObjective = Literal['stretch', 'one-hot']
GENERATION_OBJECTIVES = get_args(GenerationObjective)
GENERATION_DURATIONS = (1, 2, 3, 4)
GENERATION_CLIP_COUNTS = (1, 2, 4, 8, 10)


def validate_clip_count(value: int) -> int:
    if value not in GENERATION_CLIP_COUNTS:
        raise ValueError('Choose 1, 2, 4, 8, or 10 clips')
    return value


class GenerationSubmission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: UUID
    channel_id: str = Field(min_length=1)
    objective: GenerationObjective
    duration_seconds: StrictInt = Field(default=2, ge=GENERATION_DURATIONS[0], le=GENERATION_DURATIONS[-1])
    clip_count: StrictInt = Field(default=4, ge=1, le=10)
    seed: StrictInt = Field(default=42, ge=0, le=4294967295)

    @field_validator('clip_count')
    @classmethod
    def supported_clip_count(cls, value: int) -> int:
        return validate_clip_count(value)

    @staticmethod
    def context_fields():
        return ('channel_id', 'objective', 'duration_seconds', 'clip_count', 'seed')


class GeneratedClip(BaseModel):
    filename: str
    sha256: str
    frames: int
    sample_rate: int = 16000
    subtype: str = 'PCM_16'


class Generation(BaseModel):
    id: str
    job_id: str
    fit_id: str
    experiment_id: str
    configuration: ExperimentConfiguration
    channel_id: str
    objective: GenerationObjective
    duration_seconds: int
    clip_count: int
    seed: int
    used_gpu_count: int = 1
    unused_gpu_count: int
    state: Literal['queued', 'running', 'succeeded', 'failed', 'interrupted', 'cancelling', 'cancelled'] = 'queued'
    stage: str = 'queued'
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    exit_code: Optional[int] = None
    artifacts_ready: bool = False
    clips: List[GeneratedClip] = Field(default_factory=list)


class GenerationTarget(BaseModel):
    model_config = ConfigDict(extra='forbid')
    channel_id: str = Field(min_length=1)
    objective: GenerationObjective


class BatchSubmission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: UUID
    gpu_count: StrictInt = Field(ge=1)
    targets: List[GenerationTarget] = Field(min_length=1)
    duration_seconds: StrictInt = Field(default=2, ge=GENERATION_DURATIONS[0], le=GENERATION_DURATIONS[-1])
    clip_count: StrictInt = Field(default=4, ge=1, le=10)
    seed: StrictInt = Field(default=42, ge=0, le=4294967295)

    @field_validator('clip_count')
    @classmethod
    def supported_clip_count(cls, value: int) -> int:
        return validate_clip_count(value)

    @field_validator('targets')
    @classmethod
    def unique_targets(cls, values):
        if len({(value.channel_id, value.objective) for value in values}) != len(values):
            raise ValueError('Targets must be unique channel/objective pairs')
        return values


class Batch(BaseModel):
    id: str
    job_id: str
    fit_id: str
    experiment_id: str
    configuration: ExperimentConfiguration
    gpu_count: int
    used_gpu_count: int
    unused_gpu_count: int
    targets: List[Generation]
    state: Literal['queued', 'running', 'succeeded', 'partial', 'failed', 'interrupted', 'cancelling', 'cancelled'] = 'queued'
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    artifacts_ready: bool = False


class ApplicationSession(BaseModel):
    id: str
    created_at: str
    status: Literal['open', 'closing', 'closed']
    closed_at: Optional[str] = None
    timezone: str


class RetryRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: UUID
