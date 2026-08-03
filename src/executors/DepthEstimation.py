import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
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
        
        self.images = self.request.get_param('inputImage')
        self.device_config = self.request.get_param('ConfigDevice')
        self.model_version_config = self.request.get_param('ConfigModelVersion') 
        
        self.weight = self.bootstrap.get('model')
        self.select_device = self.bootstrap.get('device')
        self.depth_results = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model_loader = ModelLoader(config=config).load_model()
        return model_loader

    def run(self):
        import traceback
        try:
            if isinstance(self.images, list):
                temp_images = dict(self.images[0]) if isinstance(self.images[0], dict) else self.images[0]
            elif isinstance(self.images, dict):
                temp_images = dict(self.images)
            else:
                temp_images = self.images
                
            if isinstance(temp_images, dict):
                temp_images.pop('timestamp', None)
                temp_images.pop('metadata', None)
                
            img = Image.get_frame(img=temp_images, redis_db=self.redis_db)
            
            DepthInference(self, img.value, img.uID).run()
            packageModel = build_response_depth(context=self)
            
            return packageModel
        except Exception as e:
            traceback.print_exc()
            raise e

if '__main__' == __name__:
    Executor(sys.argv[1]).run()
