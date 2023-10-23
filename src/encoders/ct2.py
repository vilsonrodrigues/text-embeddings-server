import os
import subprocess

import ctranslate2
import numpy as np

from configs.ct2 import Ct2Configs


class EncoderCt2:         
    def __init__(self): 
        self.configs = Ct2Configs()   
        self.backend = "Ctranslate2"

    def check_if_model_exists(self) -> None:
        """
        Check if the model already exists in the specified directory. 
        If not, download and convert the model.
        """
        if not os.path.exists(self.configs.model_path):
            self._download_and_convert_model()

    def _download_and_convert_model(self) -> None:
        """
        Download and convert a model using ct2-transformers-converter.

        This function downloads a model specified by model_id and converts it to
        the desired format, saving it to the directory specified by model_path.

        raises Exception: If the download and conversion process fails.
        """
        try:
            command = (
                f"ct2-transformers-converter --model {self.configs.model_id} "
                f"--output_dir {self.configs.model_path}"
            )
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error during model download and conversion: {e}") from e
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}") from e

    def load_encoder(self) -> None:
        """ Load a Ctranslate2 Encoder model"""
        args = {}
        args["model_path"] = self.configs.model_path
        args["device"] = "auto"
        args["compute_type"] = self.configs.compute_type

        if self.configs.device_index:
            args["device_index"] = self.configs.device_index
        if self.configs.compute_type:
            args["compute_type"] = self.configs.compute_type
        if self.configs.inter_threads:
            args["inter_threads"] = self.configs.inter_threads
        if self.configs.intra_threads:
            args["intra_threads"] = self.configs.intra_threads
        if self.configs.num_encoders:
            args["num_encoders"] = self.configs.num_encoders
        
        self.encoder = ctranslate2.Encoder(*args)       

    async def encode(self, text_list: list[list[str]]) -> list[list[float]]:
        """ Recive a list of list of text and convert in embeddings"""
        embeddings = self.encoder.forward_batch(text_list).pooler_output
        embeddings_array = np.array(embeddings)
        return embeddings_array.tolist()