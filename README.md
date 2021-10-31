# ONNX-Inference-AWS-Lambda

#### The blog post for this repo on Medium

https://medium.com/@christiansafka/onnx-inference-with-python-in-aws-lambda-f09d08530e87

#

### `onnx_export.py`
Example of exporting a model to onnx format and validating against the PyTorch model.  You'll need to change the model, as EfficientNetAutoEncoder isn't included here.

### `preprocessing.py`
Function which resizes, center crops, handles grayscale/rgba format, and normalizes images ready for inference by image-net trained models.

### `lambda_function.py`
The lambda function which accepts base64 image input, preprocesses, generates feature vector, and responds to api gateway

### `api_reponse.py`
Helper function to unify response codes returned from lambda`