import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from ray import serve

from configs.tes import tes_configs
from server.schemas import (
    EmbedRequest, 
    Info, 
    OpenAICompactRequest,
    OpenAICompactResponse,
)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=tes_configs.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@serve.deployment(
    name="TextEmbeddingsServer",
    ray_actor_options={
        "num_gpus": tes_configs.num_gpus,
        "num_cpus": tes_configs.num_cpus,
        "memory": tes_configs.memory,
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": tes_configs.max_replicas,
        "initial_replicas": 1,
    },
)
@serve.ingress(app)
class TextEmbeddingsServer:
    def __init__(self, encoder):
        self._encoder = encoder   
        self._set_dynamic_batch_configs()
        self._encoder.check_if_model_exists()
        self._encoder.load_encoder()

    def _format_openai_response(self, embeddings: list[list[float]]) -> OpenAICompactResponse:
        response_list = [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ]
        response = {
            "data": response_list,
            "model": self._encoder.configs.model_id,
            "object": "list"
        }
        return response

    def reconfigure(self, config: dict[str, Any]) -> None:
        self._handle_batch.set_max_batch_size(config.get("max_batch_size", 64))
        self._handle_batch.set_batch_wait_timeout_s(
            config.get("batch_wait_timeout_s", 0.1)
        )

    def _set_dynamic_batch_configs(self):
        self._handle_batch.set_max_batch_size(tes_configs.max_batch_size)
        self._handle_batch.set_batch_wait_timeout_s(tes_configs.batch_wait_timeout_s)         

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.1)
    async def _handle_batch(self, text_list: list[list[int]]) -> list[list[float]]:
        try:
            embeddings = self._encoder.encode(text_list)
            return await embeddings
        except:
            raise HTTPException(status_code=424, detail="Inference failed")

    @app.post("/embed", response_model=list[list[float]], status_code=status.HTTP_200_OK)
    async def embed(self, data: EmbedRequest):
        """Get Embeddings"""
        if isinstance(data.inputs, list):
            tasks = [self._handle_batch(text) for text in data.inputs]
            embeddings = await asyncio.gather(*tasks)
        elif isinstance(data.inputs, str):
            embeddings = await self._handle_batch(data.inputs)
        return embeddings

    @app.post("/openai", response_model=OpenAICompactResponse, status_code=status.HTTP_200_OK)
    async def embed_openai(self, data: OpenAICompactRequest):
        """OpenAI compatible route"""
        if isinstance(data.inputs, list):
            tasks = [self._handle_batch(text) for text in data.inputs]
            embeddings = await asyncio.gather(*tasks)
        elif isinstance(data.inputs, str):
            embeddings = await self._handle_batch(data.inputs)
        embeddings_openai = self._format_openai_response(embeddings)
        return embeddings_openai

    @app.get("/info", response_model=Info, status_code=status.HTTP_200_OK)
    def info(self):
        return {
            "backend": self._encoder.backend,
            "max_client_batch_size": tes_configs.max_batch_size,
            "model_dtype": tes_configs.compute_type,
            "model_id": self._encoder.configs.model_id,
            "version": tes_configs.version,            
        }

    @app.get("/", response_model=dict[str, str], status_code=status.HTTP_200_OK)
    def home(self):
        return {"Text Embeddings Server":f"v{tes_configs.version}"}        