import numpy as np
import torch
import matplotlib.pyplot as plt

class DepthInference:
    def __init__(self, context, image_value, image_uid):
        self.context = context
        self.image_value = image_value 
        self.image_uid = image_uid
        
        self.model = self.context.weight
        
        self.selected_version = self.context.model_version_config
        
    def run(self):
        raw_image = self.image_value 
        
        with torch.no_grad():
            if self.selected_version == "Version2":
                depth = self.model.infer_image(raw_image)
            
            elif self.selected_version == "Version3":
                depth = self.model.infer_image(raw_image)
            
            else:
                depth = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype=np.float32)

        depth_min = float(depth.min())
        depth_max = float(depth.max())
        depth_mean = float(depth.mean())
        
        if depth_max - depth_min > 0:
            normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = np.zeros_like(depth)
        
        normalized_depth = normalized_depth.astype(np.uint8)
        
        colormap = plt.get_cmap('inferno')
        depth_colormap = (colormap(normalized_depth / 255.0)[..., :3] * 255).astype(np.uint8)
        
        depth_stats = [
            {
                "Image ID": self.image_uid,
                "Min Depth": round(depth_min, 4),
                "Max Depth": round(depth_max, 4),
                "Mean Depth": round(depth_mean, 4),
                "Resolution": f"{depth.shape[1]}x{depth.shape[0]}"
            }
        ]
        
        self.context.depth_results.append({
            "uid": self.image_uid,
            "raw_depth": depth_stats,
            "depth_image_bgr": depth_colormap
        })
        
        print(f"[BİLGİ] {self.image_uid} ID'li görselin derinlik analizi başarıyla tamamlandı!")

