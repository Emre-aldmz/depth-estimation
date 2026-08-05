import cv2
import base64
import numpy as np
from sdks.novavision.src.helper.package import PackageHelper
from sdks.novavision.src.base.model import Image as ImageModel
from sdks.novavision.src.media.image import Image as SDKImage

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

    results = context.depth_results[0]
    depth_img_bgr = results['depth_image_bgr']
    
    _, buffer = cv2.imencode('.jpg', depth_img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    depth_image_obj = ImageModel(
        uID=results['uid'] + '_depth',
        name='outputDepthImage',
        mimeType='image/jpeg',
        encoding='base64',
        value=img_base64,
        type='Image'
    )
    
    out_image = OutputDepthImage(value=depth_image_obj)
    out_array = OutputDepthArray(value=results['raw_depth'])
    
    depth_outputs = DepthOutputs(outputDepthImage=out_image, outputDepthArray=out_array)
    depth_response = DepthResponse(outputs=depth_outputs)
    depth_executor = DepthEstimationExecutor(value=depth_response)
    
    executor = ConfigExecutor(value=depth_executor)
    package_configs = PackageConfigs(executor=executor)
    
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    packageModel = package.build_model(context)
    
    return packageModel
