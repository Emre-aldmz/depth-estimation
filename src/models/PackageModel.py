from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import  Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config, Image

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

class OutputDepthImage(Output):
    name: Literal["outputDepthImage"] = "outputDepthImage"
    value: Union[List[Image], Image]
    type: Literal["Images"] = "Images"

    class Config:
        title = "Depth Map Image"

class OutputDepthArray(Output):
    name: Literal["outputDepthArray"] = "outputDepthArray"
    value: list
    type: Literal["list"] = "list"

    class Config:
        title = "Raw Depth Data"


class ConfigDeviceGPU(Config):
    name: Literal["ConfigDeviceGPU"] = "ConfigDeviceGPU"
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU (CUDA)"

class ConfigDeviceCPU(Config):
    name: Literal["ConfigDeviceCPU"] = "ConfigDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"

class ConfigDevice(Config):
    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Union[ConfigDeviceCPU, ConfigDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"
        json_schema_extra = {"shortDescription": "Processing Device"}

class DepthModelV3(Config):
    """ Depth Anything V3 - Small """
    name: Literal["DepthModelV3"] = "DepthModelV3"
    value: str = "da3_small.pth"
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    restart: Literal[True] = True

    class Config:
        title = "V3 Model Path"

class ModelVersionV3(Config):
    depthModel: DepthModelV3
    configDevice: ConfigDevice
    name: Literal["Version3"] = "Version3"
    value: Literal["Version3"] = "Version3"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Depth Anything V3"

class DepthModelV2(Config):
    """ Depth Anything V2 - Small """
    name: Literal["DepthModelV2"] = "DepthModelV2"
    value: str = "da2_small.pth"
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"
    restart: Literal[True] = True

    class Config:
        title = "V2 Model Path"

class ModelVersionV2(Config):
    depthModel: DepthModelV2
    configDevice: ConfigDevice
    name: Literal["Version2"] = "Version2"
    value: Literal["Version2"] = "Version2"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Depth Anything V2"

class ConfigModelVersion(Config):
    name: Literal["ConfigModelVersion"] = "ConfigModelVersion"
    value: Union[ModelVersionV3, ModelVersionV2]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Model Version"
        json_schema_extra = {"shortDescription": "Select DA Version"}

class DepthInputs(Inputs):
    inputImage: InputImage

class DepthConfigs(Configs):
    configModelVersion: ConfigModelVersion

class DepthOutputs(Outputs):
    outputDepthImage: OutputDepthImage
    outputDepthArray: OutputDepthArray

class DepthRequest(Request):
    inputs: Optional[DepthInputs]
    configs: DepthConfigs

    class Config:
        json_schema_extra = {"target": "configs"}

class DepthResponse(Response):
    outputs: DepthOutputs

class DepthEstimationExecutor(Config):
    name: Literal["DepthEstimation"] = "DepthEstimation"
    value: Union[DepthRequest, DepthResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Depth Estimation"
        json_schema_extra = {"target": {"value": 0}}

class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[DepthEstimationExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"
        json_schema_extra = {"target": "value"} 

class PackageConfigs(Configs):
    executor: ConfigExecutor

class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["DepthEstimation"] = "DepthEstimation"
    UID: str = "DE_1001001" 
