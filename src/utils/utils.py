import os
import sys
import torch
from huggingface_hub import hf_hub_download

from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.logger import LoggerManager

logger = LoggerManager()

current_dir = os.path.dirname(os.path.abspath(__file__))
classes_dir = os.path.abspath(os.path.join(current_dir, "../classes"))
if classes_dir not in sys.path:
    sys.path.append(classes_dir)

try:
    from capsules.DepthEstimation.src.classes.depth_anything_v2.dpt import DepthAnythingV2
except Exception:
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception:
        DepthAnythingV2 = None

try:
    from capsules.DepthEstimation.src.classes.depth_anything_3.api import DepthAnything3
except Exception:
    try:
        from depth_anything_3.api import DepthAnything3
    except Exception:
        DepthAnything3 = None


class ModelLoader:
    def __init__(self, config: dict):
        self.config = config
        self.application = Application()
        self.device = "cpu"
        self.model = None

    def _determine_device(self) -> str:
        config_device = self.application.get_param(config=self.config, name="ConfigDevice")
        if config_device == "GPU" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self) -> dict:
        self.device = self._determine_device()

        selected_version = self.application.get_param(config=self.config, name="ConfigModelVersion")

        logger.info(f"DepthEstimation - Loading model: version={selected_version}, device={self.device}")

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

        logger.info(f"DepthEstimation - Model loaded successfully: {type(self.model)}")

        return {
            "model": self.model,
            "device": self.device
        }
