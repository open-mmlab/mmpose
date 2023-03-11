# 打印全部配置信息

官方提供的配置文件，有时候继承了多个配置文件，这样做可以方便管理，减少冗余代码。但有时候我们希望知道配置文件中没有写明的默认参数值是什么，MMPose 提供了 `tools/analysis_tools/print_config.py` 来逐字逐句打印全部的配置信息。

```shell
python tools/analysis_tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```
