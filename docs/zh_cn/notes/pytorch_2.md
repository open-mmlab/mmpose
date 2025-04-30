# PyTorch 2.0 Compatibility and Benchmarks

MMPose 1.0.0 is now compatible with PyTorch 2.0, ensuring that users can leverage the latest features and performance improvements offered by the PyTorch 2.0 framework when using MMPose. With the integration of inductor, users can expect faster model speeds. The table below shows several example models:

| Model     |     Training Speed      |    Memory     |
| :-------- | :---------------------: | :-----------: |
| ViTPose-B | 29.6% ↑ (0.931 → 0.655) | 10586 → 10663 |
| ViTPose-S | 33.7% ↑ (0.563 → 0.373) |  6091 → 6170  |
| HRNet-w32 | 12.8% ↑ (0.553 → 0.482) | 9849 → 10145  |
| HRNet-w48 | 37.1% ↑ (0.437 → 0.275) |  7319 → 7394  |
| RTMPose-t | 6.3% ↑ (1.533 → 1.437)  |  6292 → 6489  |
| RTMPose-s | 13.1% ↑ (1.645 → 1.430) |  9013 → 9208  |

- Pytorch 2.0 test, add projects doc and refactor by @LareinaM in [PR#2136](https://github.com/open-mmlab/mmpose/pull/2136)
