import os

class AppConfig:
    def __init__(self):
        self.model_id = os.getenv("MODEL_ID", default="thenlper/gte-base")
        self.model_path = f"data/{self.model_id.split('/')[1]}"        
        self.compute_type = os.getenv("DTYPE", default="float16")
        self.max_batch_size_encoder = int(os.getenv("MAX_BATCH_SIZE_ENCODER", default=64))
        self.max_batch_size_tokenizer = int(os.getenv("MAX_BATCH_SIZE_TOKENIZER", default=16))
        self.batch_wait_timeout_s = float(os.getenv("BATCH_WAIT_TIMEOUT_S", default="0.1")) 
        self.version = os.getenv("VERSION")