# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Miscellaneous helper utils for Tensorflow."""

import numpy as np
import tensorflow as tf

from typing import Any, Iterable, List, Union

TfExpression = Union[tf.Tensor, tf.Variable, tf.Operation]
"""A type that represents a valid Tensorflow expression."""

TfExpressionEx = Union[TfExpression, int, float, np.ndarray]
"""A type that can be converted to a valid Tensorflow expression."""


def run(*args, **kwargs) -> Any:
    """Run the specified ops in the default session."""
    assert_tf_initialized()
    return tf.get_default_session().run(*args, **kwargs)


def is_tf_expression(x: Any) -> bool:
    """Check whether the input is a valid Tensorflow expression, i.e., Tensorflow Tensor, Variable, or Operation."""
    return isinstance(x, (tf.Tensor, tf.Variable, tf.Operation))


def shape_to_list(shape: Iterable[tf.Dimension]) -> List[Union[int, None]]:
    """Convert a Tensorflow shape to a list of ints."""
    return [dim.value for dim in shape]


def flatten(x: TfExpressionEx) -> TfExpression:
    """Shortcut function for flattening a tensor."""
    with tf.name_scope("Flatten"):
        return tf.reshape(x, [-1])


def log2(x: TfExpressionEx) -> TfExpression:
    """Logarithm in base 2."""
    with tf.name_scope("Log2"):
        return tf.log(x) * np.float32(1.0 / np.log(2.0))


def exp2(x: TfExpressionEx) -> TfExpression:
    """Exponent in base 2."""
    with tf.name_scope("Exp2"):
        return tf.exp(x * np.float32(np.log(2.0)))


def lerp(a: TfExpressionEx, b: TfExpressionEx, t: TfExpressionEx) -> TfExpressionEx:
    """Linear interpolation."""
    with tf.name_scope("Lerp"):
        return a + (b - a) * t


def lerp_clip(a: TfExpressionEx, b: TfExpressionEx, t: TfExpressionEx) -> TfExpression:
    """Linear interpolation with clip."""
    with tf.name_scope("LerpClip"):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


def absolute_name_scope(scope: str) -> tf.name_scope:
    """Forcefully enter the specified name scope, ignoring any surrounding scopes."""
    return tf.name_scope(scope + "/")


def init_tf(config_dict: dict = None) -> None:
    """Initialize TensorFlow session using good default settings."""
    if tf.get_default_session() is None:
        tf.set_random_seed(np.random.randint(1 << 31))
        create_session(config_dict, force_as_default=True)


def assert_tf_initialized():
    """Check that TensorFlow session has been initialized."""
    if tf.get_default_session() is None:
        raise RuntimeError("No default TensorFlow session found. Please call dnnlib.tflib.tfutil.init_tf().")


def create_session(config_dict: dict = None, force_as_default: bool = False) -> tf.Session:
    """Create tf.Session based on config dict."""
    config = tf.ConfigProto()

    if config_dict is not None:
        for key, value in config_dict.items():
            fields = key.split(".")
            obj = config

            for field in fields[:-1]:
                obj = getattr(obj, field)

            setattr(obj, fields[-1], value)

    session = tf.Session(config=config)

    if force_as_default:
        # pylint: disable=protected-access
        session._default_session = session.as_default()
        session._default_session.enforce_nesting = False
        session._default_session.__enter__() # pylint: disable=no-member

    return session


def init_uninitialized_vars(target_vars: List[tf.Variable] = None) -> None:
    """Initialize all tf.Variables that have not already been initialized.

    Equivalent to the following, but more efficient and does not bloat the tf graph:
    tf.variables_initializer(tf.report_uninitialized_variables()).run()
    """
    assert_tf_initialized()
    if target_vars is None:
        target_vars = tf.global_variables()

    test_vars = []
    test_ops = []

    with tf.control_dependencies(None):  # ignore surrounding control_dependencies
        for var in target_vars:
            assert is_tf_expression(var)

            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(":0", "/IsVariableInitialized:0"))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)

                with absolute_name_scope(var.name.split(":")[0]):
                    test_ops.append(tf.is_variable_initialized(var))

    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    run([var.initializer for var in init_vars])


def set_vars(var_to_value_dict: dict) -> None:
    """Set the values of given tf.Variables.

    Equivalent to the following, but more efficient and does not bloat the tf graph:
    tfutil.run([tf.assign(var, value) for var, value in var_to_value_dict.items()]
    """
    assert_tf_initialized()
    ops = []
    feed_dict = {}

    for var, value in var_to_value_dict.items():
        assert is_tf_expression(var)

        try:
            setter = tf.get_default_graph().get_tensor_by_name(var.name.replace(":0", "/setter:0"))  # look for existing op
        except KeyError:
            with absolute_name_scope(var.name.split(":")[0]):
                with tf.control_dependencies(None):  # ignore surrounding control_dependencies
                    setter = tf.assign(var, tf.placeholder(var.dtype, var.shape, "new_value"), name="setter")  # create new setter

        ops.append(setter)
        feed_dict[setter.op.inputs[1]] = value

    run(ops, feed_dict)


def create_var_with_large_initial_value(initial_value: np.ndarray, *args, **kwargs):
    """Create tf.Variable with large initial value without bloating the tf graph."""
    assert_tf_initialized()
    assert isinstance(initial_value, np.ndarray)
    zeros = tf.zeros(initial_value.shape, initial_value.dtype)
    var = tf.Variable(zeros, *args, **kwargs)
    set_vars({var: initial_value})
    return var
