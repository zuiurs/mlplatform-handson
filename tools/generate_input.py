import json
import sys

import numpy as np
from tensorflow import keras

def main(samples: int = 3):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # preprocess
    test_images = test_images / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    
    image = test_images[0:samples]

    output = open('input.json', 'w')
    data = json.dump({"signature_name": "serving_default", "instances": image.tolist()}, output, indent=2)


if __name__ == '__main__':
    samples = int(sys.argv[1])
    main(samples)
