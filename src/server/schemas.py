from pydantic import BaseModel

class Input(BaseModel):
	oneOf: str | list[str]

class EmbedRequest(BaseModel):
	inputs: Input

class OpenAICompactRequest(BaseModel):
    inputs = Input
    model: str | None = None

class OpenAICompactEmbedding(BaseModel):
	embedding: list[float]
	index: int
	object: str

class OpenAICompactResponse(BaseModel):
	data: OpenAICompactEmbedding
	model: str
	object: str

class Info(BaseModel):
    backend: str
    max_client_batch_size: int
    model_dtype: str
    model_id: str
    version: str