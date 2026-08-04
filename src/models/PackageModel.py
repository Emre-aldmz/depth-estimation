from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config, Image

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
        return "object"

    class Config:
        title = "Image"

class OutputDepthImage(Output):
    name: Literal["outputDepthImage"] = "outputDepthImage"
    value: Image
    type: Literal["Images"] = "Images"
    listen: Literal["continuous"] = "continuous"
    branch: Literal["forward"] = "forward"
    publish: Literal["stream"] = "stream"

    class Config:
        title = "Depth Map Image"

class OutputDepthArray(Output):
    name: Literal["outputDepthArray"] = "outputDepthArray"
    value: list
    type: Literal["list"] = "list"
    listen: Literal["continuous"] = "continuous"
    branch: Literal["forward"] = "forward"

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

class ModelVersionV2Small(Config):
    name: Literal["V2_Small"] = "V2_Small"
    value: Literal["V2_Small"] = "V2_Small"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "dev2_small"

class ModelVersionV2Base(Config):
    name: Literal["V2_Base"] = "V2_Base"
    value: Literal["V2_Base"] = "V2_Base"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "dev2_base"

class ModelVersionV2Large(Config):
    name: Literal["V2_Large"] = "V2_Large"
    value: Literal["V2_Large"] = "V2_Large"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "dev2_large"

class ModelVersionV3Small(Config):
    name: Literal["V3_Small"] = "V3_Small"
    value: Literal["V3_Small"] = "V3_Small"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "dev3_small"

class ModelVersionV3Base(Config):
    name: Literal["V3_Base"] = "V3_Base"
    value: Literal["V3_Base"] = "V3_Base"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "dev3_base"

class ModelVersionV3Large(Config):
    name: Literal["V3_Large"] = "V3_Large"
    value: Literal["V3_Large"] = "V3_Large"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "dev3_large"

class DepthAnythingV2(Config):
    name: Literal["DepthAnythingV2"] = "DepthAnythingV2"
    value: Union[ModelVersionV2Small, ModelVersionV2Base, ModelVersionV2Large]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "DepthAnythingV2"

class DepthAnythingV3(Config):
    name: Literal["DepthAnythingV3"] = "DepthAnythingV3"
    value: Union[ModelVersionV3Small, ModelVersionV3Base, ModelVersionV3Large]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "DepthAnythingV3"

class ConfigModelVersion(Config):
    name: Literal["ConfigModelVersion"] = "ConfigModelVersion"
    value: Union[DepthAnythingV2, DepthAnythingV3]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Model Version"
        json_schema_extra = {"shortDescription": "Select Depth Model"}

class DepthInputs(Inputs):
    inputImage: InputImage

class DepthConfigs(Configs):
    configModelVersion: ConfigModelVersion
    configDevice: ConfigDevice

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
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[DepthEstimationExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }

class PackageConfigs(Configs):
    executor: ConfigExecutor

class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["DepthEstimation"] = "DepthEstimation"
    UID: str = "DE_1001001"
