# Print Entire Config

Officially provided config files inherit multiple config files, which can facilitate management and reduce redundant code. But sometimes we want to know what the default parameter values that are not explicitly written in the configuration file are. MMPose provides `tools/analysis_tools/print_config.py` to print the entire configuration information verbatim.

```shell
python tools/analysis_tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```
