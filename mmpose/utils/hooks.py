# Copyright (c) OpenMMLab. All rights reserved.
import functools


class OutputHook:

    def __init__(self, module, outputs=None, as_tensor=False):
        self.outputs = outputs
        self.as_tensor = as_tensor
        self.layer_outputs = {}
        self.register(module)

    def register(self, module):

        def hook_wrapper(name):

            def hook(model, input, output):
                if self.as_tensor:
                    self.layer_outputs[name] = output
                else:
                    if isinstance(output, list):
                        self.layer_outputs[name] = [
                            out.detach().cpu().numpy() for out in output
                        ]
                    else:
                        self.layer_outputs[name] = output.detach().cpu().numpy(
                        )

            return hook

        self.handles = []
        if isinstance(self.outputs, (list, tuple)):
            for name in self.outputs:
                try:
                    layer = rgetattr(module, name)
                    h = layer.register_forward_hook(hook_wrapper(name))
                except ModuleNotFoundError as module_not_found:
                    raise ModuleNotFoundError(
                        f'Module {name} not found') from module_not_found
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rsetattr(obj, attr, val):
    """Set the value of a nested attribute of an object.

    This function splits the attribute path and sets the value of the
    nested attribute. If the attribute path is nested (e.g., 'x.y.z'), it
    traverses through each attribute until it reaches the last one and sets
    its value.

    Args:
        obj (object): The object whose attribute needs to be set.
        attr (str): The attribute path in dot notation (e.g., 'x.y.z').
        val (any): The value to set at the specified attribute path.
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursively get a nested attribute of an object.

    This function splits the attribute path and retrieves the value of the
    nested attribute. If the attribute path is nested (e.g., 'x.y.z'), it
    traverses through each attribute. If an attribute in the path does not
    exist, it returns the value specified as the third argument.

    Args:
        obj (object): The object whose attribute needs to be retrieved.
        attr (str): The attribute path in dot notation (e.g., 'x.y.z').
        *args (any): Optional default value to return if the attribute
            does not exist.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
