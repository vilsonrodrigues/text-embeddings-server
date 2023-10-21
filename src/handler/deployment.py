import os
import asyncio

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from ray import serve
from ray.serve.handle import DeploymentHandle

from config.app_config import AppConfig
from handler.schemas import (
    EmbedRequest, 
    Info, 
    OpenAICompactRequest,
    OpenAICompactResponse,
)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ALLOW_ORIGIN", default="*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@serve.deployment(
    name="TextEmbeddingsServer",
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
@serve.ingress(app)
class TextEmbeddingsServer:
    def __init__(self, encoder, tokenizer):        
        self.app_config = AppConfig()
        self._encoder: DeploymentHandle = encoder.options(use_new_handle_api=True)
        self._tokenizer: DeploymentHandle = tokenizer.options(use_new_handle_api=True)

    def format_openai_response(self, embeddings: list[list[float]]) -> OpenAICompactResponse:
        response_list = [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ]
        response = {
            "data": response_list,
            "model": self.app_config.model_id,
            "object": "list"
        }
        return response

    @app.get("/info", response_model=Info, status_code=status.HTTP_200_OK)
    def info(self):
        return {
            "max_client_batch_size": self.app_config.max_batch_size_encoder,
            "model_dtype": self.app_config.compute_type,
            "model_id": self.app_config.model_id,
            "version": self.app_config.version,
        }

    async def _route(self, text: str) -> list[float]:
        try:
            embedding = self._encoder.encode.remote(self._tokenizer.tokenize.remote(text))
            return await embedding
        except:
            raise HTTPException(status_code=424, detail="Inference failed")

    @app.post("/embed", response_model=list[list[float]], status_code=status.HTTP_200_OK)
    async def embed(self, data: EmbedRequest):
        """Get Embeddings"""
        if isinstance(data.inputs, list):
            tasks = [self._route(text) for text in data.inputs]
            embeddings = await asyncio.gather(*tasks)
        elif isinstance(data.inputs, str):
            embeddings = await self._route(data.inputs)
        return embeddings

    @app.post("/openai", response_model=OpenAICompactResponse, status_code=status.HTTP_200_OK)
    async def predict(self, data: OpenAICompactRequest):
        """OpenAI compatible route"""
        if isinstance(data.inputs, list):
            tasks = [self._route(text) for text in data.inputs]
            embeddings = await asyncio.gather(*tasks)
        elif isinstance(data.inputs, str):
            embeddings = await self._route(data.inputs)
        embeddings_openai = self.format_openai_response(embeddings)
        return embeddings_openai
