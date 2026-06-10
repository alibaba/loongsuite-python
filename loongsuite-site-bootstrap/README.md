# loongsuite-site-bootstrap

在**不修改业务代码、不改用 `loongsuite-instrument` 启动命令**的前提下，通过 **`site-packages` 中的 `.pth` 行**在解释器早期触发一次 `import`，从而在进程内执行与 [`opentelemetry-instrument`](https://github.com/open-telemetry/opentelemetry-python-contrib)（`sitecustomize` → `initialize()`）等价的 OpenTelemetry 自动注入逻辑。

与 `loongsuite-bootstrap` 的职责**不同**：本包只负责启动早期注入，**不会**安装任何 Instrumentation 包。可与 `loongsuite-bootstrap` 配合使用（推荐），也可按需自行 `pip install` 所需的 `opentelemetry-instrumentation-*` / `loongsuite-instrumentation-*`。

## 安装

```bash
pip install loongsuite-site-bootstrap
```

随后请安装 Instrumentation 与 Exporter。推荐用 [loongsuite-bootstrap](../loongsuite-distro/README.rst) 安装：

```bash
loongsuite-bootstrap -a install --latest
# 或仅安装当前环境已存在依赖对应的埋点
loongsuite-bootstrap -a install --latest --auto-detect
```

也可以手动安装，例如：

```bash
pip install opentelemetry-exporter-otlp loongsuite-instrumentation-langchain
```

## 配置优先级

1. **进程环境变量**（导出、容器 env、启动参数等）：已存在的键**不会被** `bootstrap-config.json` 覆盖。  
2. **`~/.loongsuite/bootstrap-config.json`**：仅在 **`LOONGSUITE_PYTHON_SITE_BOOTSTRAP` 实际处于开启状态**时才会读取；对文件中出现的键，仅当环境中**尚未设置**时写入 `os.environ`（便于默认带上 OTLP、exporter 等，子进程可继承）。若环境中已显式设置该开关且为“关闭”含义，则**不读** JSON、也不做自动注入。

JSON 根节点须为对象；键必须为字符串。值的类型会转成字符串再写入环境变量：`bool` → `true` / `false`，`int` / `float` → 十进制字符串，`str` → 原样，`null` 跳过，其它类型 → 紧凑 JSON 字符串。

示例 `~/.loongsuite/bootstrap-config.json`：

```json
{
  "LOONGSUITE_PYTHON_SITE_BOOTSTRAP": "True",
  "OTEL_SERVICE_NAME": "my-app",
  "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"
}
```

若文件不存在、无法读取或 JSON 无效，则跳过文件（无效时可能打出一条 `logging` 告警）；不影响 Python 正常启动。

## 启用方式

默认**不执行**任何 OTel 逻辑（避免影响环境中所有 Python 进程）。可在 **`bootstrap-config.json` 或环境变量** 中开启后，本包在进程启动早期执行自动注入：

```bash
export LOONGSUITE_PYTHON_SITE_BOOTSTRAP=True
```

 **`True`** 字符串视为开启（不区分大小写，如 `true` / `TRUE`）；**其它任何取值**（含空字符串等）均视为关闭。

启用时会（若尚未设置）默认：

- `OTEL_PYTHON_DISTRO=loongsuite`
- `OTEL_PYTHON_CONFIGURATOR=loongsuite`

从而使用 [`loongsuite-distro`](../loongsuite-distro) 中的 `LoongSuiteDistro` / `LoongSuiteConfigurator`（与 `loongsuite-instrument` + `OTEL_PYTHON_DISTRO=loongsuite` 一致）。上述两项仍使用 **`setdefault`**，在 JSON 补齐之后执行；若环境或 JSON 已为同名变量赋值，则保持已有取值。

## 成功提示与静默模式

默认情况下，为保持兼容，`initialize()` 成功后会向 **stdout** 打印一行：

```text
loongsuite-site-bootstrap: started successfully (OpenTelemetry auto-instrumentation initialized).
```

如果应用把 stdout 作为协议输出、交互终端或前端日志流，可关闭这行成功提示：

```bash
export LOONGSUITE_PYTHON_SITE_BOOTSTRAP_LOG_SUCCESS=False
```

关闭成功提示不影响失败日志：自动注入失败时仍会通过 bootstrap logger 记录错误。若需要无 stdout 的成功确认，可让 bootstrap 写状态文件：

```bash
export LOONGSUITE_PYTHON_SITE_BOOTSTRAP_STATUS_FILE=/tmp/loongsuite-site-bootstrap-status.json
```

成功时文件内容类似：

```json
{"initialized":true,"pid":12345,"version":"0.6.0"}
```

成功后 bootstrap 还会在当前进程内设置 `LOONGSUITE_PYTHON_SITE_BOOTSTRAP_STARTED=true`，也可以通过 `loongsuite_site_bootstrap.is_initialized()` 读取当前 bootstrap 状态。真正的业务接入是否成功仍应以导出端是否收到对应 trace / metrics 为准。

## 行为说明

- 安装后 wheel 会在 `site-packages` 根目录释放 `loongsuite-site-bootstrap.pth`，其中含一行 `import loongsuite_site_bootstrap`，依赖 CPython `site` 对 `.pth` 中 `import` 行的标准行为。
- 不使用 `python -S`（禁用 `site`）时才会生效。
- 作用范围是**当前 Python 环境**内所有启用了本 bootstrap 的进程，不仅是某一应用入口。
- 本包自带一个仅绑定在 `loongsuite_site_bootstrap` logger 上的 `StreamHandler`，不依赖应用是否已配置 `logging`。

## 卸载

```bash
pip uninstall loongsuite-site-bootstrap
```

卸载后 `.pth` 会随包移除；若曾手动复制 `.pth`，需自行清理。
