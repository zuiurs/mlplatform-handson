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
    subprocess.run(['pip', 'install', 'numpy'])
    import numpy as np

    # TODO: Write code here!
    print('Write code here!')

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
    import subprocess
    subprocess.run(['pip', 'install', 'numpy'])
    import numpy as np

    train_images = np.load(train_images_path)
    test_images = np.load(test_images_path)

    # TODO: Write code here!
    print('Write code and pass processed image properly via OutputPath here!')


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

    # TODO: Write code here!
    print('Write code here!')

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

    # TODO: Write code here!
    print('Write code and print loss/acc here!')


@dsl.pipeline(
  name='Fashion MNIST Pipeline',
  description='This pipeline provides Training/Serving for Fashion MNIST'
)
def pipeline(
        epochs='5'
):
    # TODO: Write code here!
    print('Write code here!')


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, os.path.splitext(__file__)[0] + '.yaml')
