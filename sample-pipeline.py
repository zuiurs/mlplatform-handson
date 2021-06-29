import kfp
from kfp import dsl
from kfp.components import func_to_container_op


@func_to_container_op
def add(
        number1: int,
        number2: int
) -> int:
    return number1 + number2


@func_to_container_op
def square(
        number: int
) -> int:
    return number * number


@func_to_container_op
def show(
        number: int
):
    print(number)


@dsl.pipeline(
  name='Kubeflow pipelines sample',
  description='This is sample pipeline.'
)
def pipeline(
        a='1',
        b='2'
):
    add_op = add(a, b)
    square_op = square(add_op.output)
    show(square_op.output)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, 'sample.yaml')
