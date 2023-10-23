from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

COMPUTE_TYPE_DESCRIPTION = """
    Model computation type or a dictionary mapping a device name to the 
    computation type (possible values are: default, auto, int8, int8_float32, 
    int8_float16, int8_bfloat16, int16, float16, bfloat16, float32)
    """
INTER_THREADS_DESCRIPTION = """Maximum number of parallel generations."""
INTRA_THREADS_DESCRIPTION = """
    Number of OpenMP threads per encoder (0 to use a default value)."""
NUM_ENCODERS_DESCRITPION = """Number of encoders backing this instance."""
DEVICE_INDEX_DESCRIPTION = """
    List of device IDs where this encoder is running on."""

class Ct2Configs(BaseSettings):
    model_id: str = Field(
        "thenlper/gte-base", 
        description="Model ID on HuggingFace Hub",
        examples=["thenlper/gte-base"],
    )
    device_index: str | None = Field(
        None,
        description=DEVICE_INDEX_DESCRIPTION,
        examples="0,1")    
    compute_type: str = Field("auto", description=COMPUTE_TYPE_DESCRIPTION) 
    inter_threads: int | None = Field(None, description=INTER_THREADS_DESCRIPTION)
    intra_threads: int | None = Field(None, description=INTRA_THREADS_DESCRIPTION)
    num_encoders: int | None = Field(None, description=NUM_ENCODERS_DESCRITPION)
    model_path: str =  ""

    @field_validator("model_path", mode="after")
    def model_path_fn(cls, v, values):
        return f"data/{values.data['model_id'].split('/')[1]}" 

    @field_validator("device_index", mode="after")
    def device_index_fn(cls, v):
        """Transform string in a list"""
        if v:
            return v.split(",") 
        else:
            return v