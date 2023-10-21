from encoder.deployment import EncoderDeployment
from handler.deployment import TextEmbeddingsServer
from tokenizer.deployment import TokenizerDeployment

tes = TextEmbeddingsServer.bind(EncoderDeployment.bind(), TokenizerDeployment.bind())