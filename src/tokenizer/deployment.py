import os
from typing import Any

import numpy as np
from tokenizers import Tokenizer
from ray import serve

from config.app_config import AppConfig


@serve.deployment(
    name="Tokenizer",
    user_config=dict(max_batch_size=16, batch_wait_timeout_s=0.1),
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
        self.app_config = AppConfig()
        self._set_dynamic_batch_configs()
        self.tokenizer = Tokenizer.from_pretrained(self.app_config.model_id) 

    def _set_dynamic_batch_configs(self):
        self._handle_batch.set_max_batch_size(self.app_config.max_batch_size_tokenizer)
        self._handle_batch.set_batch_wait_timeout_s(self.app_config.batch_wait_timeout_s)          

    def reconfigure(self, config: dict[str, Any]) -> None:
        self._handle_batch.set_max_batch_size(config.get("max_batch_size", 2))
        self._handle_batch.set_batch_wait_timeout_s(
            config.get("batch_wait_timeout_s", 0.1)
        )   

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.1)
    async def _handle_batch(self, tokens_list: list[list[int]]) -> list[list[float]]:
        try:
            embeddings = self.encoder.forward_batch(tokens_list)
            embeddings_array = np.array(embeddings)
            return embeddings_array.tolist()
        except Exception as e:
            raise str(e)
        
    async def predict(self, tokens: list[int]) -> list[float]:
        return await self._handle_batch(tokens)  
    