import torch
import torch.onnx
import onnxruntime
import numpy as np
from efficientnet_pytorch.model import EfficientNetAutoEncoder

model = EfficientNetAutoEncoder.from_pretrained('efficientnet-b0')
model.eval()

dummy_input = torch.rand(1, 3, 224, 224)


# # Export the model
dynamic_axes = {'input' : {0 : 'batch_size'}, 
                            'output' : {0 : 'batch_size'}}

torch.onnx.export(model,                     # model being run
                  ##since model is in the cuda mode, input also need to be
                  dummy_input,              # model input (or a tuple for multiple inputs)
                  "efficientnet_autoencoder.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'],
                  dynamic_axes=dynamic_axes,
)

# Test if ONNX results match PyTorch results
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ae_output, latent_fc_output = model(dummy_input)

ort_session = onnxruntime.InferenceSession("efficientnet_autoencoder.onnx")

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(latent_fc_output[:]), ort_outs[1][:], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

