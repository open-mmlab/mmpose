# Copyright (c) OpenMMLab. All rights reserved.
import time
from contextlib import contextmanager
from typing import Optional


@contextmanager
def limit_max_fps(fps: Optional[float]):
    t_start = time.time()
    try:
        yield
    finally:
        t_end = time.time()
        if fps is not None:
            t_sleep = 1.0 / fps - t_end + t_start
            if t_sleep > 0:
                time.sleep(t_sleep)
