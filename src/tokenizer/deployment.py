import os
from typing import Any

import numpy as np
from tokenizers import Tokenizer
from ray import serve


@serve.deployment(
    name="Tokenizer",
    user_config=dict(max_batch_size=32, batch_wait_timeout_s=0.1),
    ray_actor_options={
        "num_gpus": 0.0,
        "num_cpus": 1.0,
        "memory": 2048,
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "initial_replicas": 1,
    },
)
class TokenizerDeployment:
    def __init__(self):
        self._get_envs()
        self.tokenizer = Tokenizer.from_pretrained(self.model_id) 

    def _get_envs(self) -> None:
        self.model_id = os.getenv("MODEL_ID", default="thenlper/gte-base")
        self._handle_batch.set_max_batch_size(int(os.getenv("MAX_BATCH_SIZE", default=32)))
        self._handle_batch.set_batch_wait_timeout_s(float(os.getenv("BATCH_WAIT_TIMEOUT_S", default="0.1")))          

    def reconfigure(self, config: dict[str, Any]) -> None:
        self._handle_batch.set_max_batch_size(config.get("max_batch_size", 2))
        self._handle_batch.set_batch_wait_timeout_s(
            config.get("batch_wait_timeout_s", 0.1)
        )   

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def _handle_batch(self, tokens_list: list[list[int]]) -> list[list[float]]:
        try:
            embeddings = self.encoder.forward_batch(tokens_list)
            embeddings_array = np.array(embeddings)
            return embeddings_array.tolist()
        except Exception as e:
            raise str(e)
        
    async def predict(self, tokens: list[int]) -> list[float]:
        return await self._handle_batch(tokens)  
    