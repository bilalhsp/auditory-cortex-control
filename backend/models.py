from typing import Literal

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


class Experiment(BaseModel):
    id: str
    created_at: str
    configuration: ExperimentConfiguration
