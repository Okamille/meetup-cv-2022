import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from skimage.transform import resize

model = load(Path(__file__).parent / "model" / "mnist_model.joblib")


def preprocess_image(data_uri: str) -> np.array:
    image = plt.imread(data_uri)
    image_rescaled = resize(image, (28, 28), mode="constant", anti_aliasing=False)
    image_rescaled = (rgb2gray(image_rescaled) * 255).astype(int)
    return image_rescaled.reshape(1, -1)


def rgb2gray(rgb):
    return 1 - np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def predict_from_event(event, context):
    data_uri = event["body"]
    result = model.predict(preprocess_image(data_uri))
    body = {"result": int(result[0])}

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }

    return response
