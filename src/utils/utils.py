import os
import torch
from huggingface_hub import hf_hub_download 

try:
    from capsules.DepthEstimation.src.classes.depth_anything_v2.dpt import DepthAnythingV2
    from capsules.DepthEstimation.src.classes.depth_anything_3.api import DepthAnything3
except ModuleNotFoundError:
    from src.classes.depth_anything_v2.dpt import DepthAnythingV2
    from src.classes.depth_anything_3.api import DepthAnything3

class ModelLoader:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cpu"
        self.model = None

    def _determine_device(self) -> str:
        executor_cfg = self.config.get("executor", {}).get("value", {})
        configs = executor_cfg.get("configs", {})
        model_version_data = configs.get("configModelVersion", {}).get("value", {})
        
        device_cfg = model_version_data.get("configDevice", {}).get("value", "CPU")
        
        if device_cfg == "GPU" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self) -> dict:
        self.device = self._determine_device()
        
        executor_cfg = self.config.get("executor", {}).get("value", {})
        configs = executor_cfg.get("configs", {})
        model_version_data = configs.get("configModelVersion", {})
        selected_version = model_version_data.get("name") # "Version2" veya "Version3"
        
        if selected_version == "Version2":
            repo_id = "depth-anything/Depth-Anything-V2-Small"
            filename = "depth_anything_v2_vits.pth"
            
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            
            self.model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device).eval()
            
        elif selected_version == "Version3":
            self.model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
            self.model = self.model.to(device=torch.device(self.device))
            
        return {
            "model": self.model,
            "device": self.device
        }
