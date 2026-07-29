import cv2
import base64
from sdks.novavision.src.helper.package import PackageHelper
from sdks.novavision.src.base.model import Image as ImageModel

from capsules.DepthEstimation.src.models.PackageModel import (
    PackageModel, 
    PackageConfigs, 
    ConfigExecutor, 
    DepthEstimationExecutor, 
    DepthResponse, 
    DepthOutputs, 
    OutputDepthImage, 
    OutputDepthArray
)

def build_response_depth(context):
    
    out_images_list = []
    out_arrays_list = []

    for result in context.depth_results:
        
        _, buffer = cv2.imencode('.png', result["depth_image_bgr"])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        depth_image_obj = ImageModel(
            UID=result["uid"] + "_depth",
            mime_type="image/png",
            encoding="base64",
            value=img_base64
        )
        
        out_images_list.append(depth_image_obj)
        out_arrays_list.append(result["raw_depth"])
    
    out_image = OutputDepthImage(value=out_images_list)
    out_array = OutputDepthArray(value=out_arrays_list)
    
    depth_outputs = DepthOutputs(outputDepthImage=out_image, outputDepthArray=out_array)
    depth_response = DepthResponse(outputs=depth_outputs)
    depth_executor = DepthEstimationExecutor(value=depth_response)
    
    executor = ConfigExecutor(value=depth_executor)
    package_configs = PackageConfigs(executor=executor)
    
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    packageModel = package.build_model(context)
    
    return packageModel
