import os
import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from kfp.components import InputPath, OutputPath


@func_to_container_op
def load_data(
        train_images_path: OutputPath('npy'),
        train_labels_path: OutputPath('npy'),
        test_images_path: OutputPath('npy'),
        test_labels_path: OutputPath('npy')
):
    import os
    import subprocess
    subprocess.run(['pip', 'install', 'tensorflow', 'numpy'])
    from tensorflow import keras
    import numpy as np

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Rename to OutputPath
    np.save(train_images_path, train_images)
    os.rename(f'{train_images_path}.npy', train_images_path)
    np.save(train_labels_path, train_labels)
    os.rename(f'{train_labels_path}.npy', train_labels_path)
    np.save(test_images_path, test_images)
    os.rename(f'{test_images_path}.npy', test_images_path)
    np.save(test_labels_path, test_labels)
    os.rename(f'{test_labels_path}.npy', test_labels_path)
    print(f'Train Images: {train_images_path}\nTrain Labels: {train_labels_path}\n'
            + f'Test Images: {test_images_path}\nTest Labels: {test_labels_path}') 


@func_to_container_op
def preprocess(
        train_images_path: InputPath('npy'),
        test_images_path: InputPath('npy'),
        processed_train_images_path: OutputPath('npy'),
        processed_test_images_path: OutputPath('npy')
):
    import os
    import subprocess
    subprocess.run(['pip', 'install', 'numpy'])
    import numpy as np

    train_images = np.load(train_images_path)
    test_images = np.load(test_images_path)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    np.save(processed_train_images_path, train_images)
    os.rename(f'{processed_train_images_path}.npy', processed_train_images_path)
    np.save(processed_test_images_path, test_images)
    os.rename(f'{processed_test_images_path}.npy', processed_test_images_path)
    print(f'Preprocessed Train Images: {processed_train_images_path}')
    print(f'Preprocessed Test Images: {processed_test_images_path}')


@func_to_container_op
def train(
        epochs: int,
        processed_train_images_path: InputPath('npy'),
        train_labels_path: InputPath('npy'),
        model_path: OutputPath('zip')
):
    import subprocess
    import shutil
    import glob
    import tempfile
    subprocess.run(['pip', 'install', 'numpy', 'tensorflow'])
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np

    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
            strides=2, activation='relu', name='Conv1'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
    ])

    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    model.summary()

    train_images = np.load(processed_train_images_path)
    train_labels = np.load(train_labels_path)

    model.fit(train_images, train_labels, epochs=epochs)

    # Save model
    model_dir = tempfile.mkdtemp()
    keras.models.save_model(model, model_dir, save_format='tf')

    print('[Model Directory]')
    files = glob.glob(model_dir + '/*')
    for f in files:
        print(f)

    shutil.make_archive('model', 'zip', root_dir=model_dir)
    # Rename to OutputPath
    shutil.move('model.zip', model_path)


@func_to_container_op
def evaluate(
        model_path: InputPath('zip'),
        processed_test_images_path: InputPath('npy'),
        test_labels_path: InputPath('npy')
):
    import subprocess
    import tempfile
    import zipfile
    subprocess.run(['pip', 'install', 'numpy', 'tensorflow'])
    from tensorflow import keras
    import numpy as np

    # Load model
    model_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(model_path) as z:
        z.extractall(model_dir)

    model = keras.models.load_model(model_dir)

    test_images = np.load(processed_test_images_path)
    test_labels = np.load(test_labels_path)

    loss, acc = model.evaluate(test_images, test_labels)
    print(f'Loss: {loss}\nAccuracy: {acc}')


@dsl.pipeline(
  name='Fashion MNIST Pipeline',
  description='This pipeline provides Training/Serving for Fashion MNIST'
)
def pipeline(
        epochs='5'
):
    load_data_op = load_data()

    preprocess_op = preprocess(
            train_images=load_data_op.outputs['train_images'],
            test_images=load_data_op.outputs['test_images']
    )

    train_op = train(
            epochs=epochs,
            processed_train_images=preprocess_op.outputs['processed_train_images'],
            train_labels=load_data_op.outputs['train_labels']
    )

    evaluate(
            model=train_op.outputs['model'],
            processed_test_images=preprocess_op.outputs['processed_test_images'],
            test_labels=load_data_op.outputs['test_labels']
    )


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, os.path.splitext(__file__)[0] + '.yaml')
