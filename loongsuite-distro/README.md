# LoongSuite Distro

LoongSuite Python Agent 的 Distro 包，提供 LoongSuite 定制的 OpenTelemetry 配置与工具链。

## Installation

```bash
pip install loongsuite-distro
```

可选依赖（OTLP exporter）：

```bash
pip install loongsuite-distro[otlp]
```

## Features

1. **LoongSuite Distro**：提供 `OTEL_PYTHON_DISTRO=loongsuite` 对应的 OpenTelemetry 配置入口。  
2. **LoongSuite Bootstrap**：提供 `loongsuite-bootstrap` / `loongsuite-instrument` 命令行能力。

## Recommended usage

与仓库根目录 `README.md` 保持一致，推荐流程如下：

1. 安装 distro：

```bash
pip install loongsuite-distro
```

2. 安装 instrumentations（任选其一）：

```bash
# 方式 A：安装 release 对应的全部组件
loongsuite-bootstrap -a install --latest

# 方式 B：按当前环境自动探测，仅安装命中的埋点
loongsuite-bootstrap -a install --latest --auto-detect

# 指定版本示例
loongsuite-bootstrap -a install --version X.Y.Z
```

3. 使用 `loongsuite-instrument` 启动应用：

```bash
loongsuite-instrument \
  --traces_exporter console \
  --metrics_exporter console \
  --service_name demo \
  python demo.py
```

## Manual configuration

也可以通过环境变量显式指定 distro：

```bash
export OTEL_PYTHON_DISTRO=loongsuite
export OTEL_PYTHON_CONFIGURATOR=loongsuite
```

## References

- [LoongSuite Python Agent](https://github.com/alibaba/loongsuite-python)
- [Root README](../README.md)
