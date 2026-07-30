from sdks.novavision.src.helper.package import PackageHelper
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
    
    if context.depth_results:
        result = context.depth_results[0]
        
        # set_frame ile Redis'e yazılmış ImageModel objesini doğrudan kullan
        out_image = OutputDepthImage(value=result["depth_image_model"])
        out_array = OutputDepthArray(value=result["raw_depth"])
    else:
        out_image = OutputDepthImage(value=[])
        out_array = OutputDepthArray(value=[])
    
    depth_outputs = DepthOutputs(outputDepthImage=out_image, outputDepthArray=out_array)
    depth_response = DepthResponse(outputs=depth_outputs)
    depth_executor = DepthEstimationExecutor(value=depth_response)
    
    executor = ConfigExecutor(value=depth_executor)
    package_configs = PackageConfigs(executor=executor)
    
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    packageModel = package.build_model(context)
    
    return packageModel
