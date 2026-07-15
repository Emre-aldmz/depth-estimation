import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2 

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
        selected_version = model_version_data.get("name") 
        
        weights_dir = os.path.join(os.path.dirname(__file__), "../weights")
        
        if selected_version == "Version2":
            model_file = model_version_data.get("value", {}).get("depthModel", {}).get("value", "da2_small.pth")
            model_path = os.path.join(weights_dir, model_file)
            
            self.model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device).eval()
            
        elif selected_version == "Version3":
            from depth_anything_3.api import DepthAnything3
            model_file = model_version_data.get("value", {}).get("depthModel", {}).get("value", "da3_small.pth")
            model_path = os.path.join(weights_dir, model_file)
            
            if os.path.exists(model_path):
                self.model = DepthAnything3.from_pretrained(model_path)
            else:
                self.model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
                
            self.model = self.model.to(device=torch.device(self.device))
            
        return {
            "model": self.model,
            "device": self.device
        }
