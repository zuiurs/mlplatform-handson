import os
import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from kfp.components import load_component_from_url
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
) -> float:
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
    
    return acc


@func_to_container_op
def check(
        accuracy: float,
        threshold: str
) -> bool:
    print(f'check start')
    print(f'{accuracy} > {threshold} -> {accuracy > float(threshold)}')
    return accuracy > float(threshold)


@func_to_container_op
def upload(
        model_path: InputPath('zip'),
        project_id: str,
        bucket_name: str,
        dir_name: str
) -> str:
    import subprocess
    import tempfile
    import zipfile
    import glob
    import os
    subprocess.run(['pip', 'install', 'google-cloud-storage'])
    from google.cloud import storage as gcs

    client = gcs.Client(project_id)
    bucket = client.get_bucket(bucket_name)
    blob_gcs = bucket.blob(dir_name)

    # e.g., /tmp/tmp-xxxx
    model_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(model_path) as z:
        z.extractall(model_dir)

    for local_file in glob.glob(model_dir + '/**', recursive=True):
        if os.path.isfile(local_file):
            # /{dir_name}/model/saved_model.pb
            blob_gcs = bucket.blob(dir_name+local_file.replace(model_dir, '/0001'))
            blob_gcs.upload_from_filename(local_file)

    model_uri = f'gs://{bucket_name}/{dir_name}'
    print(f'Uploaded {model_path} to {model_uri}')
    return model_uri


kfserving_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/'
                                       'master/components/kubeflow/kfserving/component.yaml')


@dsl.pipeline(
  name='Fashion MNIST Pipeline',
  description='This pipeline provides Training/Serving for Fashion MNIST'
)
def pipeline(
        project_id,
        bucket_name,
        epochs='5',
        threshold='0.8',
        model_directory='kfp'
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

    eval_op = evaluate(
            model=train_op.outputs['model'],
            processed_test_images=preprocess_op.outputs['processed_test_images'],
            test_labels=load_data_op.outputs['test_labels']
    )

    # TODO: Call check component

    today = ('{{workflow.creationTimestamp.Y}}'
             '{{workflow.creationTimestamp.m}}'
             '{{workflow.creationTimestamp.d}}'
             '{{workflow.creationTimestamp.H}}'
             '{{workflow.creationTimestamp.M}}'
             '{{workflow.creationTimestamp.S}}')
    dir_name = '{}/{}'.format(model_directory, today)

    upload_op = upload(
            model=train_op.outputs['model'],
            project_id=project_id,
            bucket_name=bucket_name,
            dir_name=dir_name
    )

    # TODO: Call kfserving component


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, os.path.splitext(__file__)[0] + '.yaml')
