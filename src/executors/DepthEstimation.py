import os
import sys
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.model import Image as ImageModel
from sdks.novavision.src.media.image import Image as ImageHelper
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor

from capsules.DepthEstimation.src.models.PackageModel import PackageModel
from capsules.DepthEstimation.src.utils.utils import ModelLoader
from capsules.DepthEstimation.src.classes.DepthInference import DepthInference
from capsules.DepthEstimation.src.utils.response import build_response_depth

class DepthEstimation(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        
        self.request.model = PackageModel(**(self.request.data))
        
        self.images = self.request.get_param("inputImage")
        self.device_config = self.request.get_param("ConfigDevice")
        self.model_version_config = self.request.get_param("ConfigModelVersion") 
        
        self.weight = self.bootstrap.get("model")
        self.select_device = self.bootstrap.get("device")
        
        self.depth_results = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model_loader = ModelLoader(config=config).load_model()
        return model_loader

    def run(self):
        img = ImageHelper.get_frame(img=self.images, redis_db=self.redis_db)
        
        if img and img.value is not None:
            DepthInference(self, img.value, img.uID).run()
        
        for result in self.depth_results:
            depth_numpy = result["depth_image_bgr"]
            depth_bgr = depth_numpy[:, :, ::-1].copy()
            
            img_uID = str(uuid.uuid4())
            depth_img = ImageModel(
                name="DepthMap_" + img_uID,
                uID=img_uID,
                mimeType="image/png",
                encoding="bytes",
                value=depth_bgr,
                r_key='',
                type="Image"
            )
            depth_img = ImageHelper.set_frame(img=depth_img, package_uID=self.uID, redis_db=self.redis_db)
            result["depth_image_model"] = depth_img
        
        packageModel = build_response_depth(context=self)
        return packageModel

if "__main__" == __name__:
    Executor(sys.argv[1]).run()
