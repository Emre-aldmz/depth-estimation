import numpy as np
import torch
import matplotlib.pyplot as plt

class DepthInference:
    def __init__(self, context, image_value, image_uid):
        self.context = context
        self.image_value = image_value 
        self.image_uid = image_uid
        
        self.model = self.context.weight
        
        model_version_data = self.context.model_version_config.get("value", {})
        self.selected_version = model_version_data.get("name") 

    def run(self):
        raw_image = self.image_value 
        
        with torch.no_grad():
            if self.selected_version == "Version2":
                depth = self.model.infer_image(raw_image)
            
            elif self.selected_version == "Version3":
                depth = self.model.infer_image(raw_image)
            
            else:
                depth = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype=np.float32)

        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max - depth_min > 0:
            normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = np.zeros_like(depth)
        
        normalized_depth = normalized_depth.astype(np.uint8)
        
        # OpenCV yerine Matplotlib ile Inferno renk haritası (RGB formatında döner)
        colormap = plt.get_cmap('inferno')
        depth_colormap = (colormap(normalized_depth / 255.0)[..., :3] * 255).astype(np.uint8)
        
        self.context.depth_results.append({
            "uid": self.image_uid,
            "raw_depth": depth.tolist(), 
            "depth_image_bgr": depth_colormap
        })
        
        print(f"[BİLGİ] {self.image_uid} ID'li görselin derinlik analizi başarıyla tamamlandı!")
