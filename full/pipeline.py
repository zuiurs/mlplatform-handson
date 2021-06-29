import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from kfp.components import load_component_from_url
from kfp.components import InputPath, OutputPath


kfserving_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/'
                                       'master/components/kubeflow/kfserving/component.yaml')

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

    # np.save() will automatically give the extension .npy,
    # so change it back to match OutputPath.
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
    model.summary()

    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    train_images = np.load(processed_train_images_path)
    train_labels = np.load(train_labels_path)

    model.fit(train_images, train_labels, epochs=epochs)

    model_dir = tempfile.mkdtemp()
    keras.models.save_model(model, model_dir, save_format='tf')

    print('[Model Directory]')
    files = glob.glob(model_dir + '/*')
    for f in files:
        print(f)

    shutil.make_archive('model', 'zip', root_dir=model_dir)
    # shutil.make_archive() will automatically give the extension .zip,
    # so change it back to match OutputPath.
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

    test_images = np.load(processed_test_images_path)
    test_labels = np.load(test_labels_path)

    model_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(model_path) as z:
        z.extractall(model_dir)

    model = keras.models.load_model(model_dir)
    loss, acc = model.evaluate(test_images, test_labels)
    print(f'Loss: {loss}\nAccuracy: {acc}')
    
    return acc


@func_to_container_op
def check(
        accuracy: float,
        threshold: str
) -> bool:
    print(f'check start')
    return accuracy > float(threshold)


# upload uploads extracted model not zip
# because InferenceService requires model
# path as directory.
@func_to_container_op
def upload(
        model_path: InputPath('zip'),
        project_id: str,
        bucket_name: str,
        gcs_path: str
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
    blob_gcs = bucket.blob(gcs_path)

    # e.g., /tmp/tmp-xxxx
    model_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(model_path) as z:
        z.extractall(model_dir)

    for local_file in glob.glob(model_dir + '/**', recursive=True):
        if os.path.isfile(local_file):
            # /{gcs_path}/model/saved_model.pb
            blob_gcs = bucket.blob(gcs_path+local_file.replace(model_dir, '/0001'))
            blob_gcs.upload_from_filename(local_file)

    model_uri = f'gs://{bucket_name}/{gcs_path}'
    print(f'Uploaded {model_path} to {model_uri}')
    return model_uri


@func_to_container_op
def envcheck():
    import os
    for k, v in os.environ.items():
        print(f'{k}: {v}')


@dsl.pipeline(
  name='Fashion MNIST Pipeline',
  description='This pipeline provides Training/Serving for Fashion MNIST'
)
def pipeline(
        project_id,
        bucket_name,
        # If defined as int, it will occur warning.
        epochs='5',
        threshold='0.8',
        model_directory='kfp'
):
    envcheck()

    load_data_task = load_data()

    preprocess_task = preprocess(
            train_images=load_data_task.outputs['train_images'],
            test_images=load_data_task.outputs['test_images']
    )

    train_task = train(
            epochs=epochs,
            processed_train_images=preprocess_task.outputs['processed_train_images'],
            train_labels=load_data_task.outputs['train_labels']
    )

    eval_task = evaluate(
            model=train_task.outputs['model'],
            processed_test_images=preprocess_task.outputs['processed_test_images'],
            test_labels=load_data_task.outputs['test_labels']
    )

    check_task = check(eval_task.output, threshold)

    # Failed case #1
    #   In the below case, we have to accept or convert the
    #   threshold as a float, so it is not adopted. Furthermore,
    #   in the latter case, it cannot be compiled.
    # 
    #   ```
    #   with dsl.Condition(eval_task.output >= threshold):
    #   ```
    #
    #
    # Failed case #2
    #   If it doesn't evaluate to "True", an erro will occur.
    #
    #   ```
    #   with dsl.Condition(check_task.output):
    #   ```
    #   Error: AttributeError: 'PipelineParam' object has no attribute 'operand1'
    #
    with dsl.Condition(check_task.output == True):
        today = '{{workflow.creationTimestamp.Y}}{{workflow.creationTimestamp.m}}{{workflow.creationTimestamp.d}}{{workflow.creationTimestamp.H}}{{workflow.creationTimestamp.M}}{{workflow.creationTimestamp.S}}'
        gcs_path = '{}/{}'.format(model_directory, today)

        # Do not omit `_path` since it is not an InputPath.
        upload_task = upload(
                model=train_task.outputs['model'],
                project_id=project_id,
                bucket_name=bucket_name,
                gcs_path=gcs_path
        )

        kfserving_op(
                action='apply',
                model_name='fmnist',
                model_uri=upload_task.output,
                namespace='dev',
                framework='tensorflow',
                service_account='default-editor'
        )


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, 'pipeline.yaml')
