import os
import sys
import json
import torch
import gdown

from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.logger import LoggerManager

logger = LoggerManager()

current_dir = os.path.dirname(os.path.abspath(__file__))
classes_dir = os.path.abspath(os.path.join(current_dir, "../classes"))
if classes_dir not in sys.path:
    sys.path.append(classes_dir)

CACHE_DIR = "/storage/models/depth_estimation"

MODEL_REGISTRY = {
    "V2_Small": {
        "version": "v2",
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "drive_id": "1xLfZ8wbbgeNFlFBsmwQXPC15ryYuppCP",
        "filename": "da2_vits.pth",
    },
    "V2_Base": {
        "version": "v2",
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "drive_id": "1aFtb4M3885dZFUJpb4xFJvutYPYuGm-1",
        "filename": "da2_vitb.pth",
    },
    "V2_Large": {
        "version": "v2",
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "drive_id": "1FN53WqHKl5VrZx-bRg190eZTVmWSiJfd",
        "filename": "da2_vitl.pth",
    },
    "V3_Small": {
        "version": "v3",
        "model_name": "da3-small",
        "drive_id": "1RTlRxlLlcInRZOi9JVFzCZmx_vsEp6x0",
        "config_json": {
            "model_name": "da3-small",
            "config": {
                "__object__": {"path": "depth_anything_3.model.da3", "name": "DepthAnything3Net", "args": "as_params"},
                "net": {
                    "__object__": {"path": "depth_anything_3.model.dinov2.dinov2", "name": "DinoV2", "args": "as_params"},
                    "name": "vits", "out_layers": [5, 7, 9, 11],
                    "alt_start": 4, "qknorm_start": 4, "rope_start": 4, "cat_token": True
                },
                "head": {
                    "__object__": {"path": "depth_anything_3.model.dualdpt", "name": "DualDPT", "args": "as_params"},
                    "dim_in": 768, "output_dim": 2, "features": 64, "out_channels": [48, 96, 192, 384]
                },
                "cam_enc": {
                    "__object__": {"path": "depth_anything_3.model.cam_enc", "name": "CameraEnc", "args": "as_params"},
                    "dim_out": 384
                },
                "cam_dec": {
                    "__object__": {"path": "depth_anything_3.model.cam_dec", "name": "CameraDec", "args": "as_params"},
                    "dim_in": 768
                }
            }
        },
    },
    "V3_Base": {
        "version": "v3",
        "model_name": "da3-base",
        "drive_id": "1J7Xty8VBzek1cLi4HEvklB4jizI-KrIX",
        "config_json": {
            "model_name": "da3-base",
            "config": {
                "__object__": {"path": "depth_anything_3.model.da3", "name": "DepthAnything3Net", "args": "as_params"},
                "net": {
                    "__object__": {"path": "depth_anything_3.model.dinov2.dinov2", "name": "DinoV2", "args": "as_params"},
                    "name": "vitb", "out_layers": [5, 7, 9, 11],
                    "alt_start": 4, "qknorm_start": 4, "rope_start": 4, "cat_token": True
                },
                "head": {
                    "__object__": {"path": "depth_anything_3.model.dualdpt", "name": "DualDPT", "args": "as_params"},
                    "dim_in": 1536, "output_dim": 2, "features": 128, "out_channels": [96, 192, 384, 768]
                },
                "cam_enc": {
                    "__object__": {"path": "depth_anything_3.model.cam_enc", "name": "CameraEnc", "args": "as_params"},
                    "dim_out": 768
                },
                "cam_dec": {
                    "__object__": {"path": "depth_anything_3.model.cam_dec", "name": "CameraDec", "args": "as_params"},
                    "dim_in": 1536
                }
            }
        },
    },
    "V3_Large": {
        "version": "v3",
        "model_name": "da3-large",
        "drive_id": "1P4AX22xXfidrXAU3P_3AFif54EmTQTQ8",
        "config_json": {
            "model_name": "da3-large",
            "config": {
                "__object__": {"path": "depth_anything_3.model.da3", "name": "DepthAnything3Net", "args": "as_params"},
                "net": {
                    "__object__": {"path": "depth_anything_3.model.dinov2.dinov2", "name": "DinoV2", "args": "as_params"},
                    "name": "vitl", "out_layers": [11, 15, 19, 23],
                    "alt_start": 8, "qknorm_start": 8, "rope_start": 8, "cat_token": True
                },
                "head": {
                    "__object__": {"path": "depth_anything_3.model.dualdpt", "name": "DualDPT", "args": "as_params"},
                    "dim_in": 2048, "output_dim": 2, "features": 256, "out_channels": [256, 512, 1024, 1024]
                },
                "cam_enc": {
                    "__object__": {"path": "depth_anything_3.model.cam_enc", "name": "CameraEnc", "args": "as_params"},
                    "dim_out": 1024
                },
                "cam_dec": {
                    "__object__": {"path": "depth_anything_3.model.cam_dec", "name": "CameraDec", "args": "as_params"},
                    "dim_in": 2048
                }
            }
        },
    },
}


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

    def _download_from_drive(self, drive_id: str, output_path: str):
        """Google Drive'dan dosyayı indir. Zaten varsa atla (cache)."""
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logger.info(f"DepthEstimation - Downloading model from Google Drive to: {output_path}")
            gdown.download(id=drive_id, output=output_path, quiet=False)
            logger.info(f"DepthEstimation - Download complete: {output_path}")
        else:
            logger.info(f"DepthEstimation - Model already cached: {output_path}")

    def _load_v2(self, reg: dict) -> None:
        """V2 modeli yükle: Drive'dan .pth indir → DepthAnythingV2 oluştur → state_dict yükle."""
        
        try:
            from capsules.DepthEstimation.src.classes.depth_anything_v2.dpt import DepthAnythingV2
        except Exception:
            from depth_anything_v2.dpt import DepthAnythingV2

        model_path = os.path.join(CACHE_DIR, "v2", reg["filename"])
        self._download_from_drive(drive_id=reg["drive_id"], output_path=model_path)

        self.model = DepthAnythingV2(
            encoder=reg["encoder"],
            features=reg["features"],
            out_channels=reg["out_channels"]
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()

    def _load_v3(self, reg: dict) -> None:
        """V3 modeli yükle: Drive'dan .safetensors indir → config.json oluştur → from_pretrained."""
        
        try:
            from capsules.DepthEstimation.src.classes.depth_anything_3.api import DepthAnything3
        except Exception:
            from depth_anything_3.api import DepthAnything3

        model_name = reg["model_name"]
        model_dir = os.path.join(CACHE_DIR, "v3", model_name)
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        config_path = os.path.join(model_dir, "config.json")

        self._download_from_drive(drive_id=reg["drive_id"], output_path=safetensors_path)

        if not os.path.exists(config_path):
            os.makedirs(model_dir, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(reg["config_json"], f, indent=2)
            logger.info(f"DepthEstimation - Created config.json at: {config_path}")

        self.model = DepthAnything3.from_pretrained(model_dir, local_files_only=True)
        self.model = self.model.to(device=torch.device(self.device))

    def load_model(self) -> dict:
        self.device = self._determine_device()
        selected_model = self.application.get_param(config=self.config, name="ConfigModelVersion")

        logger.info(f"DepthEstimation - Loading model: key={selected_model}, device={self.device}")

        if selected_model not in MODEL_REGISTRY:
            raise ValueError(f"DepthEstimation - Unknown model key: {selected_model}")

        reg = MODEL_REGISTRY[selected_model]

        if reg["version"] == "v2":
            self._load_v2(reg)
        elif reg["version"] == "v3":
            self._load_v3(reg)

        logger.info(f"DepthEstimation - Model loaded: {type(self.model)}")

        return {
            "model": self.model,
            "device": self.device
        }

