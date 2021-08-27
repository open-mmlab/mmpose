# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict

import numpy as np
from mmcv import Timer


class StopWatch:
    r"""A helper class to measure FPS and detailed time consuming of each phase
    in a video processing loop or similar scenarios.

    Args:
        window (int): The sliding window size to calculate the running average
            of the time consuming.

    Example::
        >>> from mmpose.utils import StopWatch
        >>> import time
        >>> stop_watch = StopWatch(window=10)
        >>> with stop_watch.timeit('total'):
        >>>     time.sleep(0.1)
        >>>     # 'timeit' support nested use
        >>>     with stop_watch.timeit('phase1'):
        >>>         time.sleep(0.1)
        >>>     with stop_watch.timeit('phase2'):
        >>>         time.sleep(0.2)
        >>>     time.sleep(0.2)
        >>> report = stop_watch.report()
    """

    def __init__(self, window=1):
        self._record = defaultdict(list)
        self._timer_stack = []
        self.window = window

    def timeit(self, timer_name='_FPS_'):
        """Timing a code snippet with an assigned name.

        Args:
            timer_name (str): The unique name of the interested code snippet to
                handle multiple timers and generate reports. Note that '_FPS_'
                is a special key that the measurement will be in `fps` instead
                of `millisecond`. Also see `report` and `report_strings`.
                Default: '_FPS_'.
        Note:
            This function should always be used in a `with` statement, as shown
            in the example.
        """
        self._timer_stack.append((timer_name, Timer()))
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, trackback):
        timer_name, timer = self._timer_stack.pop()
        self._record[timer_name].append(timer.since_start())
        self._record[timer_name] = self._record[timer_name][-self.window:]

    def report(self):
        """Report timing information.

        Returns:
            dict: The key is the timer name and the value is the corresponding
                average time consuming.
        """
        result = {
            name: np.mean(vals) * 1000.
            for name, vals in self._record.items()
        }
        return result

    def report_strings(self):
        """Report timing information in texture strings.

        Returns:
            list(str): Each element is the information string of a timed event,
                in format of '{timer_name}: {time_in_ms}'. Specially, if
                timer_name is '_FPS_', the result will be converted to
                fps.
        """
        result = self.report()
        strings = []
        if '_FPS_' in result:
            fps = 1000. / result.pop('_FPS_')
            strings.append(f'FPS: {fps:>5.1f}')
        strings += [f'{name}: {val:>3.0f}' for name, val in result.items()]
        return strings

    def reset(self):
        self._record = defaultdict(list)
        self._timer_stack = []
