import os
from typing import Any

import ctranslate2
from ray import serve

from config.app_config import AppConfig


@serve.deployment(
    name="Encoder",
    user_config=dict(max_batch_size=64, batch_wait_timeout_s=0.1),
    ray_actor_options={
        "num_gpus": 0.0,
        "num_cpus": 2.0,
        "memory": 4096,
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "initial_replicas": 1,
    },
)
class EncoderDeployment:
    def __init__(self):
        self.app_config = AppConfig()
        self._check_if_model_exist() 
        self._set_dynamic_batch_configs()       
        self.encoder = ctranslate2.Encoder(
            self.app_config.model_path, 
            device="auto", 
            compute_type=self.app_config.compute_type
        )

    def _check_if_model_exist(self) -> None:
        if not os.path.exists(self.app_config.model_path):
            self._download_and_convert_model()

    def _download_and_convert_model(self):
        os.system(f"!ct2-transformers-converter --model {self.app_config.model_id} --output_dir {self.app_config.model_path}")

    def _set_dynamic_batch_configs(self):
        self._handle_batch.set_max_batch_size(self.app_config.max_batch_size_encoder)
        self._handle_batch.set_batch_wait_timeout_s(self.app_config.batch_wait_timeout_s)            

    def reconfigure(self, config: dict[str, Any]) -> None:
        self._handle_batch.set_max_batch_size(config.get("max_batch_size", 64))
        self._handle_batch.set_batch_wait_timeout_s(
            config.get("batch_wait_timeout_s", 0.1)
        )

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.1)
    async def _handle_batch(self, tokens_list: list[list[int]]) -> list[list[float]]:
        try:
             embeddings = self.encoder.forward_batch(tokens_list)
             return embeddings.tolist()
        except Exception as e:
            raise str(e)
        
    async def encode(self, tokens: list[int]) -> list[float]:
        return await self._handle_batch(tokens)  