from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

BATCH_WAIT_TIMEOUT_S_DESCRIPTION = """
    batch_wait_timeout_s controls how long Serve should wait for a batch 
    once the first request arrives.
    """
MAX_BATCH_SIZE_DESCRIPTION = """
    max_batch_size controls the size of the batch. Once the first request 
    arrives, the batching decorator will wait for a full batch 
    (up to max_batch_size) until batch_wait_timeout_s is reached. If the 
    timeout is reached, the batch will be sent to the model regardless 
    the batch size.
    """
CORS_ALLOW_ORIGINS_DESCRIPTION = """CORS origins. Comma separated values."""

class TESConfigs(BaseSettings):
    max_batch_size: int = Field(64, description=MAX_BATCH_SIZE_DESCRIPTION)
    batch_wait_timeout_s: float = Field(0.1, description=BATCH_WAIT_TIMEOUT_S_DESCRIPTION)
    version: str = Field(description="TSE version")
    cors_allow_origins: str = Field("*", description=CORS_ALLOW_ORIGINS_DESCRIPTION)
    # necessary only for dev teste, for production configure in k8s manifest
    num_gpus: float = Field(0.0, description="Num GPUs avaliables")
    num_cpus: float = Field(2.0, description="Num CPUs avaliables")
    memory: int = Field(4096, description="Memory avaliable")
    max_replicas: int = Field(2, description="Max server replicas")

    @field_validator("cors_allow_origins", mode="after")
    def cors_allow_origins_fn(cls, v):
        """Transform string in a list"""
        if v:
            return v.split(",") 
        else:
            return v    

tes_configs = TESConfigs()    