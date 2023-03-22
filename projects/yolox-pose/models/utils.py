# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple


class OutputSaveObjectWrapper:
    """A wrapper class that saves the output of function calls on an object."""

    def __init__(self, obj: Any) -> None:
        self.obj = obj
        self.log = defaultdict(list)

    def __getattr__(self, attr: str) -> Any:
        """Overrides the default behavior when an attribute is accessed.

        - If the attribute is callable, hooks the attribute and saves the
        returned value of the function call to the log.
        - If the attribute is not callable, saves the attribute's value to the
        log and returns the value.
        """
        orig_attr = getattr(self.obj, attr)

        if not callable(orig_attr):
            self.log[attr].append(orig_attr)
            return orig_attr

        def hooked(*args: Tuple, **kwargs: Dict) -> Any:
            """The hooked function that logs the return value of the original
            function."""
            result = orig_attr(*args, **kwargs)
            self.log[attr].append(result)
            return result

        return hooked

    def clear(self):
        """Clears the log of function call outputs."""
        self.log.clear()

    def __deepcopy__(self, memo):
        """Only copy the object when applying deepcopy."""
        other = type(self)(deepcopy(self.obj))
        memo[id(self)] = other
        return other


class OutputSaveFunctionWrapper:
    """A class that wraps a function and saves its outputs.

    This class can be used to decorate a function to save its outputs. It wraps
    the function with a `__call__` method that calls the original function and
    saves the results in a log attribute.

    Args:
        func (Callable): A function to wrap.
        spec (Optional[Dict]): A dictionary of global variables to use as the
            namespace for the wrapper. If `None`, the global namespace of the
            original function is used.
    """

    def __init__(self, func: Callable, spec: Optional[Dict]) -> None:
        """Initializes the OutputSaveFunctionWrapper instance."""
        assert callable(func)
        self.log = []
        self.func = func
        self.func_name = func.__name__

        if isinstance(spec, dict):
            self.spec = spec
        elif hasattr(func, '__globals__'):
            self.spec = func.__globals__
        else:
            raise ValueError

    def __call__(self, *args, **kwargs) -> Any:
        """Calls the wrapped function with the given arguments and saves the
        results in the `log` attribute."""
        results = self.func(*args, **kwargs)
        self.log.append(results)
        return results

    def __enter__(self) -> None:
        """Enters the context and sets the wrapped function to be a global
        variable in the specified namespace."""
        self.spec[self.func_name] = self
        return self.log

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exits the context and resets the wrapped function to its original
        value in the specified namespace."""
        self.spec[self.func_name] = self.func
