# LoongSuite Python Agent 发布完整指南

本文档涵盖 LoongSuite Python Agent 的架构设计、发布流程、用户安装、开发调试及维护指南。

## 目录

- [1. 项目概述](#1-项目概述)
  - [1.1 项目背景](#11-项目背景)
  - [1.2 模块介绍](#12-模块介绍)
  - [1.3 发布渠道](#13-发布渠道)
- [2. 发布原理](#2-发布原理)
  - [2.1 发布架构](#21-发布架构)
  - [2.2 核心脚本](#22-核心脚本)
  - [2.3 构建流程详解](#23-构建流程详解)
- [3. 终端用户指南](#3-终端用户指南)
  - [3.1 快速开始](#31-快速开始)
  - [3.2 安装选项](#32-安装选项)
  - [3.3 使用探针](#33-使用探针)
- [4. 开发者指南](#4-开发者指南)
  - [4.1 环境准备](#41-环境准备)
  - [4.2 本地开发](#42-本地开发)
  - [4.3 运行测试](#43-运行测试)
- [5. 维护者指南：发布新版本](#5-维护者指南发布新版本)
  - [5.1 版本号策略](#51-版本号策略)
  - [5.2 本地验证 (Dry Run)](#52-本地验证-dry-run)
  - [5.3 正式发布](#53-正式发布)
- [6. 维护者指南：同步上游代码](#6-维护者指南同步上游代码)
  - [6.1 同步原理](#61-同步原理)
  - [6.2 同步步骤](#62-同步步骤)
  - [6.3 冲突处理](#63-冲突处理)
- [7. 故障排查](#7-故障排查)
- [8. 相关文件索引](#8-相关文件索引)

---

## 1. 项目概述

### 1.1 项目背景

LoongSuite Python Agent 是基于 [OpenTelemetry Python Contrib](https://github.com/open-telemetry/opentelemetry-python-contrib) 的 Fork 项目，主要扩展了对大模型（GenAI）框架的可观测性支持。

**为什么需要 Fork？**

- 上游 `opentelemetry-util-genai` 功能有限，我们需要扩展它
- 我们新增的 `instrumentation-loongsuite/*` 依赖扩展后的 `util-genai`
- 为避免依赖冲突，我们将扩展后的包重命名为 `loongsuite-*` 前缀发布

### 1.2 模块介绍

| 模块类型 | 源目录 | 说明 |
|---------|--------|------|
| **util-genai** | `util/opentelemetry-util-genai` | GenAI 工具库，提供通用的 span 处理、token 计算等功能 |
| **distro** | `loongsuite-distro` | 发行版入口，提供 `loongsuite-bootstrap` 和 `loongsuite-instrument` 命令 |
| **GenAI instrumentations** | `instrumentation-genai/*` | 来自上游的 GenAI 插桩，如 OpenAI、VertexAI 等 |
| **LoongSuite instrumentations** | `instrumentation-loongsuite/*` | LoongSuite 自研插桩，如 DashScope、AgentScope、MCP 等 |
| **标准 instrumentations** | `instrumentation/*` | 标准微服务插桩（Flask、Django、Redis 等），由上游发布 |

### 1.3 发布渠道

LoongSuite 采用**双轨发布策略**：

| 发布后包名 | 发布目标 | 来源 | 说明 |
|-----------|----------|------|------|
| `loongsuite-otel-util-genai` | **PyPI** | `util/opentelemetry-util-genai` | 重命名后发布 |
| `loongsuite-distro` | **PyPI** | `loongsuite-distro` | 引导器 |
| `loongsuite-instrumentation-*`（`instrumentation-loongsuite/*`） | **PyPI** | 各插件目录 | 与 release 版本对齐的独立 wheel（`agno` / `mcp` / `dify` 暂不上 PyPI，见构建脚本 FIXME） |
| `loongsuite-instrumentation-*` | **GitHub Release** | `instrumentation-genai/*` + `instrumentation-loongsuite/*` | 仍打入 tar.gz，供 bootstrap 离线安装 |
| `opentelemetry-instrumentation-*` | **PyPI (上游)** | `instrumentation/*` | 由上游 OpenTelemetry 发布 |

> **包名迁移说明：**`loongsuite-util-genai` 是历史发行名，仍可供旧环境使用；
> 新版本的 LoongSuite GenAI 工具库更新发布到 `loongsuite-otel-util-genai`。
> 本次迁移只改变 PyPI distribution 名，Python import 仍保持
> `opentelemetry.util.genai`。

**依赖关系图：**

```
用户环境
├── loongsuite-distro (PyPI)
│   ├── provides: loongsuite-bootstrap, loongsuite-instrument
│   └── depends: opentelemetry-api, opentelemetry-sdk
├── loongsuite-otel-util-genai (PyPI)
│   └── GenAI 通用工具库
├── loongsuite-instrumentation-*（PyPI：除 agno/mcp/dify 外其余 `instrumentation-loongsuite/*`；tar 内仍含全部 loongsuite 插桩）
│   ├── loongsuite-instrumentation-dashscope
│   ├── loongsuite-instrumentation-vertexai (renamed from opentelemetry-*)
│   └── ... (依赖 loongsuite-otel-util-genai)
└── opentelemetry-instrumentation-* (PyPI 上游)
    ├── opentelemetry-instrumentation-flask
    ├── opentelemetry-instrumentation-redis
    └── ...
```

---

## 2. 发布原理

### 2.1 发布架构

发布流程由统一脚本 `scripts/loongsuite/loongsuite_release.sh` 驱动，同时支持本地执行和 GitHub Actions 调用：

```
                    ┌─────────────────────────────────────┐
                    │       Release Trigger               │
                    │  (本地 CLI / GitHub workflow_dispatch)│
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────┐
                    │     loongsuite_release.sh           │
                    │     (统一发布脚本)                    │
                    └───────────────┬─────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
 创建 release/{version} 分支    构建包 (PyPI + GitHub tar)    收集/归档 changelog
 生成 bootstrap_gen.py          验证产物                      提交 & 推送
                                    │                               │
                            ┌───────▼────────┐              ┌───────▼────────┐
                            │  dist-pypi/    │              │ dist/release-  │
                            │  *.whl         │              │ notes.md       │
                            └───────┬────────┘              └───────┬────────┘
                                    │                               │
                            ┌───────▼────────┐              ┌───────▼────────┐
                            │     PyPI       │              │ GitHub Release │
                            └────────────────┘              └────────────────┘
```

### 2.2 核心脚本

| 脚本 | 作用 |
|------|------|
| `scripts/loongsuite/loongsuite_release.sh` | **统一发布脚本**，支持 `--dry-run` 验证和正式发布两种模式 |
| `scripts/loongsuite/collect_loongsuite_changelog.py` | changelog 收集、归档、模块版本升级（`--collect` / `--archive` / `--bump-dev`） |
| `scripts/loongsuite/generate_loongsuite_bootstrap.py` | 生成 `bootstrap_gen.py`，定义包名映射和版本 |
| `scripts/loongsuite/build_loongsuite_package.py` | 构建 wheel 包，处理包名重命名和依赖替换 |
| `.github/workflows/loongsuite-release.yml` | GitHub Actions 发布工作流（调用统一脚本） |

### 2.3 构建流程详解

#### Step 1: 生成 bootstrap_gen.py

```bash
python scripts/loongsuite/generate_loongsuite_bootstrap.py \
  --upstream-version 0.60b1 \
  --loongsuite-version 0.1.0
```

**处理逻辑：**
- 扫描 `instrumentation/`、`instrumentation-genai/`、`instrumentation-loongsuite/` 目录
- 对 `instrumentation-genai/opentelemetry-*` 包进行重命名（→ `loongsuite-*`）
- 生成包名到版本的映射表

#### Step 2: 构建 PyPI 包

```bash
python scripts/loongsuite/build_loongsuite_package.py --build-pypi \
  --version 0.1.0 --util-genai-version 0.1.0
```

**处理逻辑：**
- 构建 `util/opentelemetry-util-genai` → 输出 `loongsuite_otel_util_genai-*.whl`
  - 使用 TOML 解析修改 `pyproject.toml` 中的 `name` 字段
- 构建 `loongsuite-distro` → 输出 `loongsuite_distro-*.whl`
- 遍历 `instrumentation-loongsuite/*` → 输出 `loongsuite_instrumentation_*-*.whl`（`agno` / `mcp` / `dify` 在 `loongsuite_pypi_manifest.py` 中硬编码跳过 PyPI，FIXME 待测全后放开；依赖中的 `opentelemetry-util-genai` 临时替换为 `loongsuite-otel-util-genai`；另受 `loongsuite-build-config.json` 的 `skip_packages` 约束）；`--collect` 生成的 release notes 含本次上传 PyPI 的发行版名列表（与 manifest 一致）

#### Step 3: 构建 GitHub Release 包

```bash
python scripts/loongsuite/build_loongsuite_package.py --build-github-release \
  --version 0.1.0 --util-genai-version 0.1.0
```

**处理逻辑：**
- 遍历 `instrumentation-genai/` 目录：
  - **规则 1**：`opentelemetry-*` 前缀的包重命名为 `loongsuite-*`
  - **规则 2**：动态检测依赖，将 `opentelemetry-util-genai` 替换为 `loongsuite-otel-util-genai`
- 遍历 `instrumentation-loongsuite/` 目录：
  - 仅应用依赖替换规则
- 所有 `.whl` 打包为 `loongsuite-python-{version}.tar.gz`

**规则匹配（无需硬编码包名）：**

```python
# 重命名规则：instrumentation-genai/ 下的 opentelemetry-* 包
def should_rename_package(package_dir: Path) -> bool:
    return "instrumentation-genai" in str(package_dir) and \
           package_dir.name.startswith("opentelemetry-")

# 依赖替换规则：检测 pyproject.toml 中是否包含 opentelemetry-util-genai
def depends_on_util_genai(pyproject_path: Path) -> bool:
    content = pyproject_path.read_text()
    return "opentelemetry-util-genai" in content
```

---

## 3. 终端用户指南

### 3.1 快速开始

```bash
# 1. 安装 loongsuite-distro (从 PyPI)
pip install loongsuite-distro

# 2. 安装所有 instrumentations
loongsuite-bootstrap -a install --version 0.1.0

# 3. 运行你的应用（自动注入探针）
loongsuite-instrument python app.py
```

### 3.2 安装选项

#### 完整安装

```bash
# 安装所有可用的 instrumentations
loongsuite-bootstrap -a install --version 0.1.0
```

#### 按需安装 (推荐)

```bash
# 只安装当前环境中已安装库对应的 instrumentations
loongsuite-bootstrap -a install --version 0.1.0 --auto-detect
```

#### 白名单安装

```bash
# 创建白名单
cat > whitelist.txt << EOF
loongsuite-instrumentation-dashscope
loongsuite-instrumentation-langchain
opentelemetry-instrumentation-flask
opentelemetry-instrumentation-redis
EOF

# 只安装白名单中的包
loongsuite-bootstrap -a install --version 0.1.0 --whitelist whitelist.txt
```

### 3.3 使用探针

#### 方式 1: 命令行自动注入

```bash
# 自动加载所有已安装的 instrumentations
loongsuite-instrument python app.py

# 指定 exporter
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
loongsuite-instrument python app.py
```

#### 方式 2: 代码中手动集成

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 配置 TracerProvider
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# 手动启用特定 instrumentation
from opentelemetry.instrumentation.dashscope import DashScopeInstrumentor
DashScopeInstrumentor().instrument()
```

### 安装流程说明

`loongsuite-bootstrap -a install` 执行两阶段安装：

```
Phase 1: 从 GitHub Release tar.gz 安装 loongsuite-* 包
  └── pip install --find-links <extracted_dir> loongsuite-instrumentation-*

Phase 2: 从 PyPI 安装 opentelemetry-* 包
  └── pip install opentelemetry-instrumentation-flask==0.60b1 ...
```

---

## 4. 开发者指南

### 4.1 环境准备

开发环境**不需要**进行包名重命名，因为所有代码都在本地，可以直接使用 `opentelemetry-util-genai` 等原始包名。

```bash
# 克隆仓库
git clone https://github.com/alibaba/loongsuite-python.git
cd loongsuite-python

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装开发依赖
pip install -r dev-requirements.txt
```

### 4.2 本地开发

#### 安装核心包（editable 模式）

```bash
# 安装 opentelemetry 核心包（从上游）
pip install opentelemetry-api opentelemetry-sdk opentelemetry-semantic-conventions

# 安装本地 util-genai（使用原始包名，无需重命名）
pip install -e ./util/opentelemetry-util-genai

# 安装你要开发的 instrumentation
pip install -e ./instrumentation-loongsuite/loongsuite-instrumentation-dashscope

# 安装 loongsuite-distro（用于测试 bootstrap）
pip install -e ./loongsuite-distro
```

#### 开发新的 instrumentation

```bash
# 复制模板
cp -r instrumentation-loongsuite/_template \
      instrumentation-loongsuite/loongsuite-instrumentation-mylib

# 修改 pyproject.toml 中的包名、依赖等
# 实现 __init__.py 中的 Instrumentor 类
```

### 4.3 运行测试

#### 使用 tox（推荐）

```bash
# 激活 conda 环境（如果使用 conda）
conda activate loongsuite

# 运行特定模块的测试
tox -c tox-loongsuite.ini -e py312-test-loongsuite-instrumentation-dashscope-latest

# 运行 lint
tox -c tox-loongsuite.ini -e lint-loongsuite-instrumentation-dashscope
```

#### 直接运行 pytest

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 安装被测模块
pip install -e ./instrumentation-loongsuite/loongsuite-instrumentation-dashscope
pip install -r ./instrumentation-loongsuite/loongsuite-instrumentation-dashscope/tests/requirements.latest.txt

# 运行测试
pytest instrumentation-loongsuite/loongsuite-instrumentation-dashscope/tests/
```

#### 测试环境配置参考

参考 `tox-loongsuite.ini` 中的配置，每个模块可以有两套测试依赖：

- `requirements.oldest.txt`: 最低支持版本的依赖
- `requirements.latest.txt`: 最新版本的依赖

---

## 5. 维护者指南：发布新版本

### 5.1 版本号策略

发布时需要指定两个版本号：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--loongsuite-version` | LoongSuite 包版本 | `0.1.0`, `0.2.0b0` |
| `--upstream-version` | 上游 OpenTelemetry 包版本 | `0.60b1`, `0.61b0` |

**应用规则：**
- `loongsuite-version` → `loongsuite-otel-util-genai`, `loongsuite-distro`, `loongsuite-instrumentation-*`
- `upstream-version` → `bootstrap_gen.py` 中的 `opentelemetry-instrumentation-*` 版本

### 5.2 本地验证 (Dry Run)

```bash
# 完整验证（含安装测试）
./scripts/loongsuite/loongsuite_release.sh \
  -l 0.1.0 -u 0.60b1 --dry-run

# 快速验证（跳过安装测试）
./scripts/loongsuite/loongsuite_release.sh \
  -l 0.1.0 -u 0.60b1 --dry-run --skip-install
```

Dry Run 模式**不会**创建分支、归档 changelog、提交代码或创建 GitHub Release，仅执行构建和验证。

**验证步骤：**

| 步骤 | 说明 | 产物 |
|------|------|------|
| 1 | 安装构建依赖 | - |
| 3 | 生成 bootstrap_gen.py | `loongsuite-distro/src/.../bootstrap_gen.py` |
| 4 | 构建 PyPI 包 | `dist-pypi/*.whl` |
| 5 | 构建 GitHub Release 包 | `dist/*.tar.gz` |
| 6 | 验证 tar 内容 | - |
| 7 | 收集 changelog 生成 release notes | `dist/release-notes.md` |
| 10 | 安装验证（可选） | 临时 venv 中测试 |

**验证要点：**

- `loongsuite-otel-util-genai`、`loongsuite-distro` 以及 `instrumentation-loongsuite` 中参与 PyPI 构建的 `loongsuite-instrumentation-*` wheel 在 `dist-pypi/` 中（`agno` / `mcp` / `dify` 当前不产出 PyPI wheel）
- `loongsuite-otel-util-genai` 不在 `tar.gz` 中
- `loongsuite-instrumentation-*`（含 genai 重命名包与 loongsuite 插件）在 `tar.gz` 中
- `opentelemetry-util-genai` 不在任何产物中（避免冲突）
- 安装后依赖关系正确

### 5.3 正式发布

发布使用统一脚本 `scripts/loongsuite/loongsuite_release.sh`，支持本地和 CI 两种方式。

**发布流程会自动完成：**

1. 从 main 创建 `release/{version}` 分支
2. 生成 `bootstrap_gen.py`
3. 构建 PyPI 包和 GitHub Release tar.gz
4. 收集 changelog 生成 release notes
5. 归档 changelog（将 `Unreleased` 替换为 `Version X.Y.Z (YYYY-MM-DD)`）
6. 提交并推送 release 分支
7. 创建 GitHub Release
8. 创建 post-release PR 到 main（归档 changelog + 升级模块版本为下一个 `.dev`）

#### 方式 1: 本地执行（推荐）

```bash
# 完整发布
./scripts/loongsuite/loongsuite_release.sh \
  -l 0.1.0 -u 0.60b1

# 跳过 GitHub Release（没有 gh CLI 时）
./scripts/loongsuite/loongsuite_release.sh \
  -l 0.1.0 -u 0.60b1 --skip-github-release
```

#### 方式 2: GitHub Actions 手动触发

1. 进入 GitHub 仓库 → **Actions** → **LoongSuite Release**
2. 点击 **Run workflow**
3. 填写参数：
   - `loongsuite_version`: `0.1.0`
   - `upstream_version`: `0.60b1`
   - `skip_pypi`: 测试时可勾选
4. 执行

CI 工作流会调用同一个脚本完成构建和分支管理，然后在独立 job 中发布 PyPI、创建 GitHub Release，并自动创建 post-release PR 到 main。

#### 方式 3: Tag 触发

```bash
git tag v0.1.0
git push origin v0.1.0
```

#### 本地执行 vs CI 执行对比

| 步骤 | 本地 (`loongsuite_release.sh`) | CI (`loongsuite-release.yml`) |
|------|------|------|
| 创建 release 分支 | 脚本内 `git checkout -b` | 同上 |
| 构建包 | 脚本内调用 | 同上 |
| 收集/归档 changelog | 脚本内调用 Python 脚本 | 同上 |
| Commit + Push | 脚本内 `git commit` + `git push` | 同上 |
| GitHub Release | 脚本内 `gh release create`（可 skip） | 独立 job |
| PyPI publish | 不执行（本地不做） | 独立 job 通过 OIDC/Token |
| Post-release PR | 脚本内创建 PR（需 `gh` CLI） | 独立 job，自动创建 |

#### PyPI / Test PyPI 发布配置

**发布到生产 PyPI（二选一）：**

1. **API Token**：在 GitHub 仓库 Settings → Secrets → Actions 中添加：
   - `PYPI_API_TOKEN`：从 [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/) 创建

2. **OIDC Trusted Publishing**（推荐）：
   - PyPI 项目设置 → Publishing → Add a new pending publisher
   - Owner: `alibaba`，Repository: `loongsuite-python`
   - Workflow: `loongsuite-release.yml`，Environment: `pypi`
   - 在 GitHub 仓库中创建 Environment `pypi`（Settings → Environments）

**发布到 Test PyPI（测试用）：**

1. 在 [test.pypi.org/manage/account/token/](https://test.pypi.org/manage/account/token/) 创建 API Token
2. 在 GitHub Secrets 中添加：`TEST_PYPI_TOKEN`（值为 `pypi-xxx`）
3. 手动触发 workflow 时，将 `publish_target` 选为 **testpypi**

**重要说明：**

- `dist-pypi/` 中的 `loongsuite_otel_util_genai-*.whl`、`loongsuite_distro-*.whl` 以及 `loongsuite_instrumentation_*.whl`（`instrumentation-loongsuite` 中当前参与 PyPI 构建的插件；`agno` / `mcp` / `dify` 暂排除）会上传到 PyPI；每个新插件首次发布前需在 PyPI 上完成项目/Trusted Publisher 配置
- `loongsuite-python-*.tar.gz` 仅用于 GitHub Release，**禁止**上传到 PyPI

### 5.4 Post-Release PR

发布完成后，脚本会自动创建一个 PR 到 `main` 分支，完成两件事：

1. **归档 Changelog**：将各 changelog 文件中的 `Unreleased` 部分标记为已发布的版本号和日期
2. **升级模块版本**：将 `instrumentation-loongsuite/` 下所有模块的 `version.py` 以及 `loongsuite-distro/src/loongsuite/distro/version.py` 升级为下一个开发版本

**版本升级规则：** 发布 `0.1.0` → 模块版本改为 `0.2.0.dev`（minor 版本 +1，后缀 `.dev`）

**受影响的文件：**
- `CHANGELOG-loongsuite.md`、`util/opentelemetry-util-genai/CHANGELOG-loongsuite.md`、`instrumentation-loongsuite/*/CHANGELOG.md`
- `instrumentation-loongsuite/*/src/**/version.py`
- `loongsuite-distro/src/loongsuite/distro/version.py`

**本地执行时**，脚本通过 `gh pr create` 自动创建 PR（需要 `gh` CLI）。  
**CI 执行时**，`post-release-pr` job 独立完成 PR 创建。

如果需要跳过此步骤：

```bash
./scripts/loongsuite/loongsuite_release.sh \
  -l 0.1.0 -u 0.60b1 --skip-post-release-pr
```

#### 发布检查清单

- [ ] 本地 dry run 通过 (`--dry-run`)
- [ ] `CHANGELOG-loongsuite.md` 等 changelog 已更新
- [ ] 版本号格式正确（不带 `v` 前缀）
- [ ] `upstream_version` 与当前上游稳定版本匹配
- [ ] PyPI 权限已配置（CI 发布时需要）
- [ ] 发布完成后检查 post-release PR 已创建并合入 main

---

## 6. 维护者指南：同步上游代码

### 6.1 同步原理

本项目 Fork 自 [opentelemetry-python-contrib](https://github.com/open-telemetry/opentelemetry-python-contrib)，需要定期同步上游的更新。

**同步策略：**

```
upstream/main ──────────────────────────────────────► 上游主分支
                    │
                    │ git fetch upstream
                    │ git merge upstream/main
                    ▼
origin/main ───────────────────────────────────────► 我们的主分支
    │
    │ feature branches
    ▼
origin/feature/* ──────────────────────────────────► 功能分支
```

**需要注意的目录：**

| 目录 | 同步策略 |
|------|----------|
| `instrumentation/` | 完全同步上游 |
| `instrumentation-genai/` | 完全同步上游 |
| `util/opentelemetry-util-genai/` | 同步上游，**保留我们的扩展** |
| `instrumentation-loongsuite/` | **我们独有**，不受上游影响 |
| `loongsuite-distro/` | **我们独有**，不受上游影响 |
| `scripts/loongsuite/generate_loongsuite_*.py` | **我们独有** |
| `scripts/loongsuite/build_loongsuite_*.py` | **我们独有** |

### 6.2 同步步骤

```bash
# 1. 添加上游远程（如果未添加）
git remote add upstream https://github.com/open-telemetry/opentelemetry-python-contrib.git

# 2. 获取上游更新
git fetch upstream

# 3. 切换到主分支
git checkout main

# 4. 合并上游更新
git merge upstream/main

# 5. 解决冲突（如有）
# ...

# 6. 推送到我们的仓库
git push origin main
```

### 6.3 冲突处理

**常见冲突场景：**

1. **`util/opentelemetry-util-genai/` 冲突**
   - 我们对这个模块有扩展
   - 需要手动合并，保留我们的扩展代码

2. **`scripts/` 目录冲突**
   - 上游的 `scripts/generate_instrumentation_bootstrap.py` 等可能更新
   - 我们的 `scripts/loongsuite/generate_loongsuite_*.py` 依赖它们
   - 需要检查 API 兼容性

3. **`pyproject.toml` 冲突**
   - 上游可能更新依赖版本
   - 需要验证兼容性

**冲突解决后验证：**

```bash
# 运行测试确保兼容性
tox -c tox-loongsuite.ini -e py312-test-loongsuite-instrumentation-dashscope-latest

# 运行 dry run 确保发布流程正常
./scripts/loongsuite/loongsuite_release.sh -l 0.1.0 -u 0.60b1 --dry-run --skip-install
```

---

## 7. 故障排查

### 构建问题

**问题**: `hatch version` 失败
```bash
# 解决: 安装 hatch
pip install hatch
```

**问题**: 构建时找不到依赖
```bash
# 解决: 安装构建依赖
pip install -r pkg-requirements.txt
```

**问题**: tomlkit 相关错误
```bash
# 解决: 安装 tomlkit
pip install tomlkit
```

### 安装问题

**问题**: `loongsuite-otel-util-genai` 找不到
```bash
# 原因: PyPI 包未正确构建
# 解决: 检查 dry run step 3 的输出，确认包名为 loongsuite_otel_util_genai
```

**问题**: `opentelemetry-util-genai` 和 `loongsuite-otel-util-genai` 冲突
```bash
# 解决: 卸载旧包
pip uninstall opentelemetry-util-genai
pip install loongsuite-otel-util-genai
```

### 发布问题

**问题**: PyPI 发布 403 Forbidden
```bash
# 解决: 检查 OIDC trusted publishing 配置或 API token
```

**问题**: 版本号已存在
```bash
# 解决: PyPI 不允许覆盖版本，使用新版本号
```

---

## 8. 相关文件索引

| 文件 | 说明 |
|------|------|
| `scripts/loongsuite/loongsuite_release.sh` | **统一发布脚本**（本地 + CI 共用，支持 `--dry-run`） |
| `scripts/loongsuite/collect_loongsuite_changelog.py` | changelog 收集（`--collect`）、归档（`--archive`）、版本升级（`--bump-dev`） |
| `scripts/loongsuite/build_loongsuite_package.py` | 构建脚本，处理包名重命名和依赖替换 |
| `scripts/loongsuite/generate_loongsuite_bootstrap.py` | 生成 bootstrap_gen.py |
| `.github/workflows/loongsuite-release.yml` | GitHub Actions 发布工作流（调用统一脚本） |
| `loongsuite-distro/src/loongsuite/distro/bootstrap.py` | Bootstrap 安装逻辑 |
| `loongsuite-distro/src/loongsuite/distro/bootstrap_gen.py` | 生成的包名映射配置 |
| `tox-loongsuite.ini` | 测试配置 |
| `pkg-requirements.txt` | 构建依赖 |
| `CHANGELOG-loongsuite.md` | 根目录变更日志 |
| `util/opentelemetry-util-genai/CHANGELOG-loongsuite.md` | util-genai 变更日志 |
| `instrumentation-loongsuite/*/CHANGELOG.md` | 各 instrumentation 包变更日志 |
