# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Helper for managing networks."""

import types
import inspect
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from typing import List, Tuple, Union

from . import tfutil
from .. import util

from .tfutil import TfExpression, TfExpressionEx

_import_handlers = []  # Custom import handlers for dealing with legacy data in pickle import.
_import_modules = []  # Temporary modules create during pickle import.


def import_handler(handler_func):
    """Function decorator for declaring custom import handlers."""
    _import_handlers.append(handler_func)
    return handler_func


class Network:
    """Generic network abstraction.

    Acts as a convenience wrapper for a parameterized network construction
    function, providing several utility methods and convenient access to
    the inputs/outputs/weights.

    Network objects can be safely pickled and unpickled for long-term
    archival purposes. The pickling works reliably as long as the underlying
    network construction function is defined in a standalone Python module
    that has no side effects or application-specific imports.

    Args:
        name: Network name. Used to select TensorFlow name and variable scopes.
        func_name: Fully qualified name of the underlying network construction function.
        static_kwargs: Keyword arguments to be passed in to the network construction function.

    Attributes:
        name: User-specified name, defaults to build func name if None.
        scope: Unique TF graph scope, derived from the user-specified name.
        static_kwargs: Arguments passed to the user-supplied build func.
        num_inputs: Number of input tensors.
        num_outputs: Number of output tensors.
        input_shapes: Input tensor shapes (NC or NCHW), including minibatch dimension.
        output_shapes: Output tensor shapes (NC or NCHW), including minibatch dimension.
        input_shape: Short-hand for input_shapes[0].
        output_shape: Short-hand for output_shapes[0].
        input_templates: Input placeholders in the template graph.
        output_templates: Output tensors in the template graph.
        input_names: Name string for each input.
        output_names: Name string for each output.
        vars: All variables (local_name => var).
        trainables: Trainable variables (local_name => var).
    """

    def __init__(self, name: str = None, func_name: str = None, **static_kwargs):
        tfutil.assert_tf_initialized()
        assert isinstance(name, str) or name is None
        assert isinstance(func_name, str)  # must not be None
        assert util.is_pickleable(static_kwargs)
        self._init_fields()
        self.name = name
        self.static_kwargs = util.EasyDict(static_kwargs)

        # Init build func.
        module, self._build_func_name = util.get_module_from_obj_name(func_name)
        self._build_module_src = inspect.getsource(module)
        self._build_func = util.get_obj_from_module(module, self._build_func_name)

        # Init graph.
        self._init_graph()
        self.reset_vars()

    def _init_fields(self) -> None:
        self.name = None
        self.scope = None
        self.static_kwargs = util.EasyDict()
        self.num_inputs = 0
        self.num_outputs = 0
        self.input_shapes = [[]]
        self.output_shapes = [[]]
        self.input_shape = []
        self.output_shape = []
        self.input_templates = []
        self.output_templates = []
        self.input_names = []
        self.output_names = []
        self.vars = OrderedDict()
        self.trainables = OrderedDict()

        self._build_func = None  # User-supplied build function that constructs the network.
        self._build_func_name = None  # Name of the build function.
        self._build_module_src = None  # Full source code of the module containing the build function.
        self._run_cache = dict()  # Cached graph data for Network.run().

    def _init_graph(self) -> None:
        # Collect inputs.
        self.input_names = []

        for param in inspect.signature(self._build_func).parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD and param.default is param.empty:
                self.input_names.append(param.name)

        self.num_inputs = len(self.input_names)
        assert self.num_inputs >= 1

        # Choose name and scope.
        if self.name is None:
            self.name = self._build_func_name

        self.scope = tf.get_default_graph().unique_name(self.name.replace("/", "_"), mark_as_used=False)

        # Build template graph.
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            assert tf.get_variable_scope().name == self.scope

            with tfutil.absolute_name_scope(self.scope):  # ignore surrounding name_scope
                with tf.control_dependencies(None):  # ignore surrounding control_dependencies
                    self.input_templates = [tf.placeholder(tf.float32, name=name) for name in self.input_names]
                    assert callable(self._build_func)
                    out_expr = self._build_func(*self.input_templates, is_template_graph=True, **self.static_kwargs)

        # Collect outputs.
        assert tfutil.is_tf_expression(out_expr) or isinstance(out_expr, tuple)
        self.output_templates = [out_expr] if tfutil.is_tf_expression(out_expr) else list(out_expr)
        self.num_outputs = len(self.output_templates)
        assert self.num_outputs >= 1
        assert all(tfutil.is_tf_expression(t) for t in self.output_templates)

        # Check that input and output shapes are defined.
        if any(t.shape.ndims is None for t in self.input_templates):
            raise ValueError("Network input shapes not defined. Please call x.set_shape() for each input.")
        if any(t.shape.ndims is None for t in self.output_templates):
            raise ValueError("Network output shapes not defined. Please call x.set_shape() where applicable.")

        # Populate remaining fields.
        self.input_shapes = [tfutil.shape_to_list(t.shape) for t in self.input_templates]
        self.output_shapes = [tfutil.shape_to_list(t.shape) for t in self.output_templates]
        self.input_shape = self.input_shapes[0]
        self.output_shape = self.output_shapes[0]
        self.output_names = [t.name.split("/")[-1].split(":")[0] for t in self.output_templates]
        self.vars = OrderedDict([(self.get_var_local_name(var), var) for var in tf.global_variables(self.scope + "/")])
        self.trainables = OrderedDict([(self.get_var_local_name(var), var) for var in tf.trainable_variables(self.scope + "/")])

    def reset_vars(self) -> None:
        """Run initializers for all variables defined by this network."""
        tfutil.run([var.initializer for var in self.vars.values()])

    def reset_trainables(self) -> None:
        """Run initializers for all trainable variables defined by this network."""
        tfutil.run([var.initializer for var in self.trainables.values()])

    def get_output_for(self, *in_expr: TfExpression, return_as_list: bool = False, **dynamic_kwargs) -> Union[TfExpression, List[TfExpression]]:
        """Get TensorFlow expression(s) for the output(s) of this network, given the inputs."""
        assert len(in_expr) == self.num_inputs
        assert not all(expr is None for expr in in_expr)

        all_kwargs = dict(self.static_kwargs)
        all_kwargs.update(dynamic_kwargs)

        with tf.variable_scope(self.scope, reuse=True):
            assert tf.get_variable_scope().name == self.scope
            valid_inputs = [expr for expr in in_expr if expr is not None]
            final_inputs = []
            for expr, name, shape in zip(in_expr, self.input_names, self.input_shapes):
                if expr is not None:
                    expr = tf.identity(expr, name=name)
                else:
                    expr = tf.zeros([tf.shape(valid_inputs[0])[0]] + shape[1:], name=name)
                final_inputs.append(expr)
            assert callable(self._build_func)
            out_expr = self._build_func(*final_inputs, **all_kwargs)

        assert tfutil.is_tf_expression(out_expr) or isinstance(out_expr, tuple)

        if return_as_list:
            out_expr = [out_expr] if tfutil.is_tf_expression(out_expr) else list(out_expr)

        return out_expr

    def get_var_local_name(self, var_or_global_name: Union[TfExpression, str]) -> str:
        """Get the local name of a given variable, excluding any surrounding name scopes."""
        assert tfutil.is_tf_expression(var_or_global_name) or isinstance(var_or_global_name, str)
        global_name = var_or_global_name if isinstance(var_or_global_name, str) else var_or_global_name.name
        assert global_name.startswith(self.scope + "/")
        local_name = global_name[len(self.scope) + 1:]
        local_name = local_name.split(":")[0]
        return local_name

    def find_var(self, var_or_local_name: Union[TfExpression, str]) -> TfExpression:
        """Find variable by local or global name."""
        assert tfutil.is_tf_expression(var_or_local_name) or isinstance(var_or_local_name, str)
        return self.vars[var_or_local_name] if isinstance(var_or_local_name, str) else var_or_local_name

    def get_var(self, var_or_local_name: Union[TfExpression, str]) -> np.ndarray:
        """Get the value of a given variable as NumPy array.
        Note: This method is very inefficient -- prefer to use tfutil.run(list_of_vars) whenever possible."""
        return self.find_var(var_or_local_name).eval()

    def set_var(self, var_or_local_name: Union[TfExpression, str], new_value: Union[int, float, np.ndarray]) -> None:
        """Set the value of a given variable based on the given NumPy array.
        Note: This method is very inefficient -- prefer to use tfutil.set_vars() whenever possible."""
        tfutil.set_vars({self.find_var(var_or_local_name): new_value})

    def __getstate__(self) -> dict:
        """Pickle export."""
        return {
            "version": 2,
            "name": self.name,
            "static_kwargs": dict(self.static_kwargs),
            "build_module_src": self._build_module_src,
            "build_func_name": self._build_func_name,
            "variables": list(zip(self.vars.keys(), tfutil.run(list(self.vars.values()))))}

    def __setstate__(self, state: dict) -> None:
        """Pickle import."""
        tfutil.assert_tf_initialized()
        self._init_fields()

        # Execute custom import handlers.
        for handler in _import_handlers:
            state = handler(state)

        # Set basic fields.
        assert state["version"] == 2
        self.name = state["name"]
        self.static_kwargs = util.EasyDict(state["static_kwargs"])
        self._build_module_src = state["build_module_src"]
        self._build_func_name = state["build_func_name"]

        # Parse imported module.
        module = types.ModuleType("_tfutil_network_import_module_%d" % len(_import_modules))
        exec(self._build_module_src, module.__dict__) # pylint: disable=exec-used
        self._build_func = util.get_obj_from_module(module, self._build_func_name)
        _import_modules.append(module)  # avoid gc

        # Init graph.
        self._init_graph()
        self.reset_vars()
        tfutil.set_vars({self.find_var(name): value for name, value in state["variables"]})

    def clone(self, name: str = None) -> "Network":
        """Create a clone of this network with its own copy of the variables."""
        # pylint: disable=protected-access
        net = object.__new__(Network)
        net._init_fields()
        net.name = name if name is not None else self.name
        net.static_kwargs = util.EasyDict(self.static_kwargs)
        net._build_module_src = self._build_module_src
        net._build_func_name = self._build_func_name
        net._build_func = self._build_func
        net._init_graph()
        net.copy_vars_from(self)
        return net

    def copy_vars_from(self, src_net: "Network") -> None:
        """Copy the values of all variables from the given network."""
        names = [name for name in self.vars.keys() if name in src_net.vars]
        tfutil.set_vars(tfutil.run({self.vars[name]: src_net.vars[name] for name in names}))

    def copy_trainables_from(self, src_net: "Network") -> None:
        """Copy the values of all trainable variables from the given network."""
        names = [name for name in self.trainables.keys() if name in src_net.trainables]
        tfutil.set_vars(tfutil.run({self.vars[name]: src_net.vars[name] for name in names}))

    def convert(self, new_func_name: str, new_name: str = None, **new_static_kwargs) -> "Network":
        """Create new network with the given parameters, and copy all variables from this network."""
        if new_name is None:
            new_name = self.name
        static_kwargs = dict(self.static_kwargs)
        static_kwargs.update(new_static_kwargs)
        net = Network(name=new_name, func_name=new_func_name, **static_kwargs)
        net.copy_vars_from(self)
        return net

    def setup_as_moving_average_of(self, src_net: "Network", beta: TfExpressionEx = 0.99, beta_nontrainable: TfExpressionEx = 0.0) -> tf.Operation:
        """Construct a TensorFlow op that updates the variables of this network
        to be slightly closer to those of the given network."""
        with tfutil.absolute_name_scope(self.scope):
            with tf.name_scope("MovingAvg"):
                ops = []

                for name, var in self.vars.items():
                    if name in src_net.vars:
                        cur_beta = beta if name in self.trainables else beta_nontrainable
                        new_value = tfutil.lerp(src_net.vars[name], var, cur_beta)
                        ops.append(var.assign(new_value))

                return tf.group(*ops)

    def run(self,
            *in_arrays: Tuple[Union[np.ndarray, None], ...],
            return_as_list: bool = False,
            print_progress: bool = False,
            minibatch_size: int = None,
            num_gpus: int = 1,
            assume_frozen: bool = False,
            out_mul: float = 1.0,
            out_add: float = 0.0,
            out_shrink: int = 1,
            out_dtype: np.dtype = None,
            **dynamic_kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]:
        """Run this network for the given NumPy array(s), and return the output(s) as NumPy array(s).

        Args:
            return_as_list: True = return a list of NumPy arrays, False = return a single NumPy array, or a tuple if there are multiple outputs.
            print_progress: Print progress to the console? Useful for very large input arrays.
            minibatch_size: Maximum minibatch size to use, None = disable batching.
            num_gpus: Number of GPUs to use.
            assume_frozen: Improve multi-GPU perf by assuming that trainables are not going to change.
            out_mul: Multiplicative constant to apply to the output(s).
            out_add: Additive constant to apply to the output(s).
            out_shrink: Shrink the spatial dimensions of the output(s) by the given factor.
            out_dtype: Convert the output to the specified data type.
            dynamic_kwargs: Additional keyword arguments to pass into the network construction function.
        """
        assert len(in_arrays) == self.num_inputs
        assert not all(arr is None for arr in in_arrays)
        num_items = in_arrays[0].shape[0]

        if minibatch_size is None:
            minibatch_size = num_items

        key = str([list(sorted(dynamic_kwargs.items())), num_gpus, out_mul, out_add, out_shrink, out_dtype])

        # Build graph.
        if key not in self._run_cache:
            with tfutil.absolute_name_scope(self.scope + "/Run"), tf.control_dependencies(None):
                with tf.device("/cpu:0"):
                    in_expr = [tf.placeholder(tf.float32, name=name) for name in self.input_names]
                    in_split = list(zip(*[tf.split(x, num_gpus) for x in in_expr]))

                out_split = []
                for gpu in range(num_gpus):
                    with tf.device("/gpu:%d" % gpu):
                        net = self.clone() if assume_frozen else self
                        out_expr = net.get_output_for(*in_split[gpu], return_as_list=True, **dynamic_kwargs)

                        if out_mul != 1.0:
                            out_expr = [x * out_mul for x in out_expr]

                        if out_add != 0.0:
                            out_expr = [x + out_add for x in out_expr]

                        if out_shrink > 1:
                            ksize = [1, 1, out_shrink, out_shrink]
                            out_expr = [tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding="VALID", data_format="NCHW") for x in out_expr]

                        if out_dtype is not None:
                            if tf.as_dtype(out_dtype).is_integer:
                                out_expr = [tf.round(x) for x in out_expr]
                            out_expr = [tf.saturate_cast(x, out_dtype) for x in out_expr]

                        out_split.append(out_expr)

                with tf.device("/cpu:0"):
                    out_expr = [tf.concat(outputs, axis=0) for outputs in zip(*out_split)]
                    self._run_cache[key] = in_expr, out_expr

        # Run minibatches.
        in_expr, out_expr = self._run_cache[key]
        out_arrays = [np.empty([num_items] + tfutil.shape_to_list(expr.shape)[1:], expr.dtype.name) for expr in out_expr]

        for mb_begin in range(0, num_items, minibatch_size):
            if print_progress:
                print("\r%d / %d" % (mb_begin, num_items), end="")

            mb_end = min(mb_begin + minibatch_size, num_items)
            mb_num = mb_end - mb_begin
            mb_in = [src[mb_begin : mb_end] if src is not None else np.zeros([mb_num] + shape[1:]) for src, shape in zip(in_arrays, self.input_shapes)]
            mb_out = tf.get_default_session().run(out_expr, dict(zip(in_expr, mb_in)))

            for dst, src in zip(out_arrays, mb_out):
                dst[mb_begin: mb_end] = src

        # Done.
        if print_progress:
            print("\r%d / %d" % (num_items, num_items))

        if not return_as_list:
            out_arrays = out_arrays[0] if len(out_arrays) == 1 else tuple(out_arrays)

        return out_arrays

    def list_ops(self) -> List[TfExpression]:
        prefix = self.scope + '/'
        return [op for op in tf.get_default_graph().get_operations() if op.name.startswith(prefix)]

    def list_layers(self) -> List[Tuple[str, TfExpression, List[TfExpression]]]:
        """Returns a list of (name, output_expr, trainable_vars) tuples corresponding to
        individual layers of the network. Mainly intended to be used for reporting."""
        layers = []

        def recurse(scope, parent_ops, level):
            prefix = scope + "/"
            ops = [op for op in parent_ops if op.name == scope or op.name.startswith(prefix)]

            # Ignore specific patterns.
            if any(p in scope for p in ["/Shape", "/strided_slice", "/Cast", "/concat"]):
                return

            # Does not contain leaf nodes => expand immediate children.
            if level == 0 or all("/" in op.name[len(prefix):] for op in ops):
                visited = set()

                for op in ops:
                    suffix = op.name[len(prefix):]

                    if "/" in suffix:
                        suffix = suffix[:suffix.index("/")]

                    if suffix not in visited:
                        recurse(prefix + suffix, ops, level + 1)
                        visited.add(suffix)
                return

            # Filter out irrelevant ops within variable name scopes.
            layer_vars = [op for op in ops if op.type.startswith("Variable")]
            for var in layer_vars:
                prefix = var.name + "/"
                ops = [op for op in ops if not op.name.startswith(prefix)]

            # Dig up the details for this layer.
            layer_name = scope[len(self.scope) + 1:]
            layer_output = ops[-1].outputs[0]
            layer_trainables = [op.outputs[0] for op in layer_vars if self.get_var_local_name(op.name) in self.trainables]
            layers.append((layer_name, layer_output, layer_trainables))

        recurse(self.scope, self.list_ops(), 0)
        return layers

    def print_layers(self, title: str = None, hide_layers_with_no_params: bool = False) -> None:
        """Print a summary table of the network structure."""
        if title is None:
            title = self.name

        print()
        print("%-28s%-12s%-24s%-24s" % (title, "Params", "OutputShape", "WeightShape"))
        print("%-28s%-12s%-24s%-24s" % (("---",) * 4))

        total_params = 0

        for layer_name, layer_output, layer_trainables in self.list_layers():
            weights = [var for var in layer_trainables if var.name.endswith("/weight:0")]
            num_params = sum(np.prod(tfutil.shape_to_list(var.shape)) for var in layer_trainables)
            total_params += num_params

            if hide_layers_with_no_params and num_params == 0:
                continue

            print("%-28s%-12s%-24s%-24s" % (
                layer_name,
                num_params if num_params else "-",
                layer_output.shape,
                weights[0].shape if len(weights) == 1 else "-"))

        print("%-28s%-12s%-24s%-24s" % (("---",) * 4))
        print("%-28s%-12s%-24s%-24s" % ("Total", total_params, "", ""))
        print()

    def setup_weight_histograms(self, title: str = None) -> None:
        """Construct summary ops to include histograms of all trainable parameters in TensorBoard."""
        if title is None:
            title = self.name

        with tf.name_scope(None), tf.device(None), tf.control_dependencies(None):
            for local_name, var in self.trainables.items():
                if "/" in local_name:
                    p = local_name.split("/")
                    name = title + "_" + p[-1] + "/" + "_".join(p[:-1])
                else:
                    name = title + "_toplevel/" + local_name

                tf.summary.histogram(name, var)
