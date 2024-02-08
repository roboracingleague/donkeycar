import torch
from segment_anything import sam_model_registry, SamPredictor

class Segmentation:
    def __init__(self, cfg):
        self.sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
