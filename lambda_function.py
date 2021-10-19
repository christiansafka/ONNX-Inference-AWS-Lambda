import json
import onnxruntime
import base64

from api_response import respond
from preprocess import preprocess_image


def lambda_handler(event, context):
    # Get image
    post_body = json.loads(event["body"])
    img_base64 = post_body.get('image')
    if img_base64 is None:
        return respond(False, None, "No image parameter received")

    try:
        b64image = base64.b64decode(img_base64)
    except Exception as e:
        print(e)
        return respond(False, None, "Couldn't decode base64 string")


    model_expected_im_size = 224

    np_image = preprocess_image(b64image, model_expected_im_size)
    
    ort_session = onnxruntime.InferenceSession("efficientnet_autoencoder.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: np_image}
    ort_outs = ort_session.run(None, ort_inputs)
    im_vector = ort_outs[1][0].tolist()


    return json.dumps(respond(True, im_vector))

