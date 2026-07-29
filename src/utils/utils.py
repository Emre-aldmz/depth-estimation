import os
import sys
import torch
from huggingface_hub import hf_hub_download

current_dir = os.path.dirname(os.path.abspath(__file__))
classes_dir = os.path.abspath(os.path.join(current_dir, "../classes"))
if classes_dir not in sys.path:
    sys.path.append(classes_dir)

# V2 Import (Bağımsız Güvenli Blok)
try:
    from capsules.DepthEstimation.src.classes.depth_anything_v2.dpt import DepthAnythingV2
except ModuleNotFoundError:
    try:
        from src.classes.depth_anything_v2.dpt import DepthAnythingV2
    except ModuleNotFoundError:
        pass

# V3 Import (Bağımsız Güvenli Blok - omegaconf hatasını yutar ve çökmeyi engeller)
try:
    from capsules.DepthEstimation.src.classes.depth_anything_3.api import DepthAnything3
except ModuleNotFoundError:
    try:
        from src.classes.depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError:
        pass

class ModelLoader:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cpu"
        self.model = None

    def _determine_device(self) -> str:
        executor_cfg = self.config.get("executor", {}).get("value", {})
        configs = executor_cfg.get("configs", {})
        model_version_data = configs.get("configModelVersion", {}).get("value", {})
        
        device_data = model_version_data.get("configDevice", {}).get("value", {})
        device_name = device_data.get("name", "ConfigDeviceCPU")
        
        if "GPU" in device_name and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self) -> dict:
        self.device = self._determine_device()
        
        executor_cfg = self.config.get("executor", {}).get("value", {})
        configs = executor_cfg.get("configs", {})
        model_version_data = configs.get("configModelVersion", {}).get("value", {})
        selected_version = model_version_data.get("name")
        
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

