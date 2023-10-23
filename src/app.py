from encoders.ct2 import EncoderCt2
from server.deployment import TextEmbeddingsServer

tes = TextEmbeddingsServer.bind(EncoderCt2())