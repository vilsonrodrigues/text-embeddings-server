import os

import ctranslate2
from ray import serve


@serve.deployment(
    name="Encoder",
    user_config=dict(max_batch_size=32, batch_wait_timeout_s=0.1),
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
        self._get_envs()
        self._check_if_model_exist()        
        self.encoder = ctranslate2.Encoder(self.model_path, device="auto", compute_type=self.compute_type)

    def _get_envs(self):
        self.model_id = os.getenv("MODEL_ID", default="thenlper/gte-base")
        self.model_path = f"data/{self.model_id.split('/')[1]}"        
        self.compute_type = os.getenv("DTYPE", default="float16")
        self._handle_batch.set_max_batch_size(int(os.getenv("MAX_BATCH_SIZE", default=32)))
        self._handle_batch.set_batch_wait_timeout_s(float(os.getenv("BATCH_WAIT_TIMEOUT_S", default="0.1")))        

    def _check_if_model_exist(self) -> None:
        if not os.path.exists(self.model_path):
            self._download_and_convert_model()

    def reconfigure(self, config: dict) -> None:
        self._handle_batch.set_max_batch_size(config.get("max_batch_size", 32))
        self._handle_batch.set_batch_wait_timeout_s(
            config.get("batch_wait_timeout_s", 0.1)
        )

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def _handle_batch(self, tokens_list: list[list[int]]) -> list[list[float]]:
        try:
             embeddings = self.encoder.forward_batch(tokens_list)
             return embeddings.tolist()
        except Exception as e:
            raise str(e)
        
    async def predict(self, tokens: list[int]) -> list[float]:
        return await self._handle_batch(tokens)  