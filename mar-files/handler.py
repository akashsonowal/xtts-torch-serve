import os
import torch
from ts.torch_handler.base_handler import BaseHandler
from ts.handler_utils.utils import send_intermediate_predict_response

from xtts_v2 import XTTSVocalizer
from gcs_bucket import download_all_files_in_folder

class TorchServeXTTSV2Handler(BaseHandler):
    def __init__(self):
        super(TorchServeXTTSV2Handler, self).__init__()
        self.model = None
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        gcs_model_path = os.getenv("MODEL_PATH", "gs://swiss-knife/org_ag/vocalizer/xttsv2_mixed")
        checkpoint_dir = os.path.join(model_dir, "xtts_artifacts")
        model_weights = "model.pth"
        speaker_reference_file = "clipped_first_15_seconds.wav"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            download_all_files_in_folder(gcs_model_path, checkpoint_dir)

        self.model = XTTSVocalizer(checkpoint_dir, model_weights, speaker_reference_file)
        self.model.load_model(use_deepspeed=torch.cuda.is_available())
        self.initialized = True
        print("Model initialized...")

    def preprocess(self, requests):
        if not isinstance(requests, list):
            requests = [requests]

        if not requests:
            raise ValueError("No requests provided.")

        req_data = requests[0]
        input_data = req_data.get("data") or req_data.get("body")
        
        if input_data is None:
            raise ValueError("Request payload must contain a 'data' or 'body' key with a non-empty value.")
        
        if isinstance(input_data, dict):
            input_data = input_data.get("data", input_data)
            if isinstance(input_data, dict):
                input_data = str(input_data)

        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")
        
        return input_data.strip()

    def inference(self, input_text):
        return self.model.predict(input_text)

    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)

        input_text = self.preprocess(data)
        response_generator = self.inference(input_text)

        for response in response_generator:
            send_intermediate_predict_response(
                [response], context.request_ids, "Intermediate response", 200, context
            )
        return []