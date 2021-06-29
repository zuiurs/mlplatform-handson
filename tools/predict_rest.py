import json
import os
import requests
import sys

import numpy as np
from tensorflow import keras

SERVING_IP = os.environ.get('SV_IP')
SERVING_PORT = os.environ.get('SV_PORT')
MODEL_NAME = os.environ.get('MODEL_NAME')

# e.g., fmnist.dev.example.com
SERVING_HOST = os.environ.get('SV_HOST')

# e.g., user@example.com
USERNAME = os.environ.get('KF_USERNAME')
PASSWORD = os.environ.get('KF_PASSWORD')


def get_authtoken() -> str:
    session = requests.Session()
    response = session.get(f'http://{SERVING_IP}:{SERVING_PORT}')
    login_data = {'login': USERNAME, 'password': PASSWORD}
    session.post(response.url, headers={'Content-Type': 'application/x-www-form-urlencoded'}, data=login_data)
    session_cookie = session.cookies.get_dict()['authservice_session']
    print(f'Session Token: {session_cookie}')
    return session_cookie


def main(samples: int = 3):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # preprocess
    test_images = test_images / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    
    image = test_images[0:samples]

    token = get_authtoken()

    headers = {'Content-Type': 'application/json', 'Host': SERVING_HOST, 'Cookie': f'authservice_session={token}'}
    endpoint = f'http://{SERVING_IP}:{SERVING_PORT}/v1/models/{MODEL_NAME}'
    print(f'Endpoint: {endpoint}')
    print(f'Host: {SERVING_HOST}')

    print('Check model endpoint:')
    model_stats = requests.get(endpoint, headers=headers)
    print(model_stats.text)

    print('Predict:')
    data = json.dumps({'signature_name': 'serving_default', 'instances': image.tolist()})
    json_response = requests.post(endpoint+':predict', data=data, headers=headers)
    print(json_response.text)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    predictions = json.loads(json_response.text)['predictions']
    for i in range(len(predictions)):
        max_label = np.argmax(predictions[i])
        expected = class_names[test_labels[i]]
        got = class_names[max_label]
        prefix = " "
        if expected != got:
            prefix = "X"
        print(f'{prefix} Result {i}: {got} (answer: {expected})')


if __name__ == '__main__':
    if SERVING_IP is None or SERVING_PORT is None or SERVING_HOST is None or MODEL_NAME is None or USERNAME is None or PASSWORD is None:
        print(f'please set empty env SV_IP={SERVING_IP}, SV_PORT={SERVING_PORT}, '
               f'SV_HOST={SERVING_HOST} and MODEL_NAME={MODEL_NAME}')
        sys.exit()

    samples = int(sys.argv[1])
    main(samples)
