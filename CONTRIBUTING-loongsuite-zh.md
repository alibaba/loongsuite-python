# 贡献代码到 loongsuite-python

为了得到更快速的响应，强烈建议您进入 DingTalk SIG 进行讨论。

LoongSuite Python SIG:

<img src="docs/_assets/img/loongsuite-python-sig-dingtalk.jpg" height="150">

## 目录

- [贡献代码到 loongsuite-python](#贡献代码到-loongsuite-python)
  - [目录](#目录)
  - [开发](#开发)
    - [虚拟环境](#虚拟环境)
    - [故障排除](#故障排除)
    - [Benchmark 测试](#benchmark-测试)
  - [PR](#PR)
    - [如何创建 PR](#如何创建-pr)
    - [如何接收评论](#如何接收评论)
    - [如何获得 PR Review](#如何获得-pr-review)
    - [如何让 PR 被合并](#如何让-pr-被合并)
  - [设计选择](#设计选择)
    - [关注功能，而非结构合规性](#关注功能而非结构合规性)
  - [本地运行测试](#本地运行测试)
    - [针对不同的 Core 仓库分支/提交进行测试](#针对不同的-core-仓库分支提交进行测试)
  - [风格指南](#风格指南)
  - [Instrumentation 指南](#instrumentation-指南)
    - [更新支持的 instrumentation 包版本](#更新支持的-instrumentation-包版本)
  - [GenAI Instrumentation 指南](#genai-instrumentation-指南)
    - [参与](#参与)
  - [对贡献者的期望](#对贡献者的期望)
  - [更新支持的 Python 版本](#更新支持的-python-版本)
    - [提升 Python 基线版本](#提升-python-基线版本)
    - [添加对新 Python 版本的支持](#添加对新-python-版本的支持)

## 开发

本项目使用 [tox](https://tox.readthedocs.io) 来自动化
开发的某些方面，包括针对多个 Python 版本的测试。
要安装 `tox`，请运行：

```sh
pip install tox
```

您也可以使用 `uv` 支持运行 tox。默认情况下 [tox.ini](./tox.ini) 会自动使用 `tox-uv` 创建一个已配置的 tox 环境，但您可以在主机级别安装它：

```sh
pip install tox-uv
```

您可以使用以下参数运行 `tox`：

* `tox` 运行所有现有的 tox 命令，包括所有包在多个 Python 版本下的单元测试
* `tox -e docs` 重新生成所有文档
* `tox -e py312-test-instrumentation-aiopg` 在特定 Python 版本下运行 aiopg instrumentation 单元测试
* `tox -c tox-loongsuite.ini -e py312-loongsuite-instrumentation-agentscope-1` 在特定 Python 版本下运行 instrumentation-loongsuite 模块下的 agentscope instrumentation 单元测试
* `tox -e spellcheck` 对所有代码运行拼写检查
* `tox -e lint-some-package` 对 `some-package` 运行 lint 检查
* `tox -e generate-workflows` 如果 tox 环境已更新，运行创建新的 CI workflow
* `tox -e ruff` 对整个代码库运行 ruff linter 和格式化检查
* `tox -e precommit` 运行所有 `pre-commit` 操作

运行 `tox -e ruff` 时会执行 `ruff check` 和 `ruff format`。我们强烈建议您通过将其安装为 git hooks 来配置 [pre-commit](https://pre-commit.com/)，以便在每次提交前自动运行 `ruff` 和 `rstcheck`。您只需在环境中[安装 pre-commit](https://pre-commit.com/#install)：

```console
pip install pre-commit -c dev-requirements.txt
```

并在 git 仓库内运行此命令：

```console
pre-commit install
```

请查看
[`tox.ini`](https://github.com/loongsuite/opentelemetry-python-contrib/blob/main/tox.ini)
[`tox-loongsuite.ini`](https://github.com/loongsuite/opentelemetry-python-contrib/blob/main/tox-loongsuite.ini)
以了解有关可用 tox 命令的更多详细信息。

### 虚拟环境

您也可以创建单个虚拟环境，以便更轻松地运行本地测试。

为此，您需要安装 [`uv`](https://docs.astral.sh/uv/getting-started/installation/)。

安装 `uv` 后，您可以运行以下命令：

```sh
uv sync
```

这将在 `.venv` 目录中创建虚拟环境并安装所有必要的依赖项。

### 故障排除

某些包可能需要安装额外的系统级依赖项。例如，您可能需要安装 `libpq-dev` 来运行 postgresql 客户端库 instrumentation 测试，或安装 `libsnappy-dev` 来运行 prometheus 导出器测试。如果遇到构建错误，请检查您尝试运行测试的包的安装说明。

对于 `docs` 构建，您可能需要根据需要安装 `mysql-client` 和其他必需的依赖项。确保本地设置中使用的 Python 版本与 [CI](./.github/workflows/) 中使用的版本匹配，以在构建文档时保持兼容性。

### Benchmark 测试

某些包有 Benchmark 测试。要运行它们，请运行 `tox -f benchmark`。Benchmark 测试使用 `pytest-benchmark`，它们会将结果表输出到控制台。

要编写 Benchmark 测试，只需使用 [pytest benchmark fixture](https://pytest-benchmark.readthedocs.io/en/latest/usage.html#usage)，如下所示：

```python
def test_simple_start_span(benchmark):
    def benchmark_start_as_current_span(span_name, attribute_num):
        span = tracer.start_span(
            span_name,
            attributes={"count": attribute_num},
        )
        span.end()

    benchmark(benchmark_start_as_current_span, "benchmarkedSpan", 42)
```

确保测试文件位于
包的 `benchmarks/` 文件夹下，并且路径对应于
它正在测试的包中的文件。确保文件名以
`test_benchmark_` 开头。（例如 `propagator/opentelemetry-propagator-aws-xray/benchmarks/trace/propagation/test_benchmark_aws_xray_propagator.py`）

## PR

### 如何创建 PR

欢迎所有人通过 GitHub Pull Request(PR) 为 `loongsuite-python` 贡献代码。

要创建新的 PR，请在 GitHub 上 fork 项目并克隆上游仓库：

```sh
git clone https://github.com/alibaba/loongsuite-python.git
cd loongsuite-python
```

将您的 fork 添加为 origin：

```sh
git remote add fork https://github.com/YOUR_GITHUB_USERNAME/loongsuite-python.git
```

确保您已安装所有支持的 Python 版本，首次安装 `tox`：

```sh
pip install tox tox-uv
```

在仓库根目录运行测试（这将运行所有 tox 环境，可能需要一些时间）：

```sh
tox
```

检出新分支，进行修改并将分支推送到您的 fork：

```sh
git checkout -b feature
```

编辑文件后，暂存当前目录中的更改：

```sh
git add .
```

然后运行以下命令提交更改：

```sh
git commit
git push fork feature
```

针对 `loongsuite-python` 仓库打开 PR。

### 如何接收评论

* 如果 PR 尚未准备好审查，请在标题中添加 `[WIP]`，将其标记为
  `work-in-progress`，或将其标记为 [`draft`](https://github.blog/2019-02-14-introducing-draft-pull-requests/)。
* 在请求审查之前，确保测试和 lint 在本地通过。
* 确保已签署 CLA 且 CI 通过。

### 如何获得 PR Review

请在 PR 中 @ 对应组件的贡献者，如果您找不到合适的 Reviewer，则可以联系 Maintainer 解决。

为了得到更快速的响应，强烈建议您进入 DingTalk SIG 进行讨论。请查看[贡献代码到 loongsuite-python](#贡献代码到-loongsuite-python)

> [!TIP]
> 即使您是新手，您的审查也很重要——对项目很有价值。请随时参与任何开放的 PR：检查文档、运行测试、提出问题，或在看起来不错时给出 +1。任何人都可以审查 PR 并帮助它们被合并。每条评论都推动项目向前发展，所以如果您有专业知识来审查 PR，请不要犹豫。

### 如何让 PR 被合并

当满足以下条件时，PR 被视为**准备合并**：

* 它已收到来自 Maintainer 的批准。
* 主要反馈已解决。
* 它已开放 Review 至少一个工作日。这给人们合理的 Review 时间。
* 微小更改（拼写错误、格式、文档等）不必等待一天。
* 紧急修复可以例外，只要已积极沟通。
* 如果对行为有任何影响，则在相应的代码库变更日志中添加变更日志条目。例如，文档条目不需要，但小的错误条目需要。

任何 Maintainer 都可以在 PR **准备合并**后合并它。

## 设计选择

与其他 LoongSuite 探针一样，loongsuite-python 遵循
[opentelemetry-specification](https://github.com/open-telemetry/opentelemetry-specification)。

推荐阅读 [库指南](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/library-guidelines.md)。

### 关注功能，而非结构合规性

OpenTelemetry 是一个不断发展的规范，其中需求和
用例是明确的，但满足这些用例的方法并不明确。

因此，贡献应该提供符合规范的功能和行为，
但接口和结构是灵活的。

最好让贡献遵循语言的惯用法
而不是符合规范中的特定 API 名称或参数模式。

有关更深入的讨论，请参阅：https://github.com/open-telemetry/opentelemetry-specification/issues/165

## 本地运行测试

1. 转到您的 Python Agent 仓库目录。`git clone git@github.com:alibaba/loongsuite-python.git && cd loongsuite-python`。
2. 确保您已安装 `tox`。`pip install tox`。
3. 运行 `tox` 不带任何参数以运行所有包的测试。阅读更多关于 [tox](https://tox.readthedocs.io/en/latest/) 的信息。

由于执行依赖项安装的预步骤，某些测试可能很慢。为了帮助解决这个问题，您可以先运行一次 tox，然后使用 toxdir 中先前安装的依赖项运行测试，如下所示：

1. 首次运行（例如，opentelemetry-instrumentation-aiopg）
```console
tox -e py312-test-instrumentation-aiopg
```
2. 再次运行测试而不执行预步骤：
```console
.tox/py312-test-instrumentation-aiopg/bin/pytest instrumentation/opentelemetry-instrumentation-aiopg
```

### 针对不同的 Core 仓库分支/提交进行测试

某些 tox 目标通过 pip 从 [OpenTelemetry Python Core 仓库](https://github.com/open-telemetry/opentelemetry-python) 安装包。安装的包的版本在本地运行 tox 时默认为该仓库的 main 分支。可以通过在运行 tox 之前设置环境变量来安装标记有特定 git 提交哈希的包，如下例所示：

```sh
CORE_REPO_SHA=c49ad57bfe35cfc69bfa863d74058ca9bec55fc3 tox
```

持续集成会根据[此处](https://github.com/alibaba/loongsuite-python/blob/main/.github/workflows/test_0.yml#L14)的配置覆盖该环境变量。

## 风格指南

* 文档字符串应遵循 [Google Python 风格指南](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)，如 [Sphinx](http://www.sphinx-doc.org/en/master/index.html) 中的 [napoleon 扩展](http://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#google-vs-numpy) 扩展所指定的那样。

## Instrumentation 指南

以下是在实现新 instrumentation 或处理特定 instrumentation 时需要注意的事项清单。作为社区，我们的目标之一是尽可能保持 instrumentation 的实现细节相似，以便于测试和功能对等。也推荐尽可能地抽象通用功能。

- 遵循语义约定
  - Instrumentation 应遵循[此处](https://github.com/open-telemetry/semantic-conventions/tree/main/docs)定义的语义约定。
  - 为了确保一致性，我们鼓励与 [STABLE](https://opentelemetry.io/docs/specs/otel/document-status/#lifecycle-status) 语义约定（如果可用）对齐的贡献。这种方法有助于我们避免潜在的混淆，并减少支持多个过时版本的语义约定的需要。但是，我们仍然愿意考虑有充分理由的例外情况。
  - 与过时的 HTTP 语义约定相关的贡献（在成为[稳定](https://github.com/open-telemetry/semantic-conventions/tree/v1.23.0)之前的约定）可能会被劝阻，因为它们增加了复杂性和误解的可能性。
- 包含一个在 [Pypi](https://pypi.org/) 中尚未被占用的名称。如果所需名称已被占用，请联系维护者。
- 继承自 [BaseInstrumentor](https://github.com/alibaba/loongsuite-python/blob/2518a4ac07cb62ad6587dd8f6cbb5f8663a7e179/opentelemetry-instrumentation/src/opentelemetry/instrumentation/instrumentor.py#L35)
- 支持自动 instrumentation
  - [在此处](./instrumentation-loongsuite/)新增一个新的 instrumentation 包，命名必须以 loongsuite-instrumentation 开头
  - 添加入口点（例如 <https://github.com/alibaba/loongsuite-python/blob/2518a4ac07cb62ad6587dd8f6cbb5f8663a7e179/instrumentation/opentelemetry-instrumentation-requests/pyproject.toml#L44>）
  - 添加新的 instrumentation 包后运行 `python scripts/generate_instrumentation_bootstrap.py`。
- 在其他 instrumentation 中通用且可以抽象的功能[在此处](https://github.com/alibaba/loongsuite-python/tree/main/opentelemetry-instrumentation/src/opentelemetry/instrumentation)，如果需要修改其中的内容，建议优先提交 issue 或 draft PR，Maintainer 会协助你贡献到上游的 OpenTelemetry 社区。
- HTTP instrumentation 的请求/响应 [hooks](https://github.com/alibaba/loongsuite-python/issues/408)
- `suppress_instrumentation` 功能
  - 例如 <https://github.com/alibaba/loongsuite-python/blob/2518a4ac07cb62ad6587dd8f6cbb5f8663a7e179/opentelemetry-instrumentation/src/opentelemetry/instrumentation/utils.py#L191>
- 抑制传播功能
  - https://github.com/alibaba/loongsuite-python/issues/344 了解更多上下文
- `exclude_urls` 功能
  - 例如 <https://github.com/alibaba/loongsuite-python/blob/2518a4ac07cb62ad6587dd8f6cbb5f8663a7e179/instrumentation/opentelemetry-instrumentation-flask/src/opentelemetry/instrumentation/flask/__init__.py#L327>
- `url_filter` 功能
  - 例如 <https://github.com/alibaba/loongsuite-python/blob/2518a4ac07cb62ad6587dd8f6cbb5f8663a7e179/instrumentation/opentelemetry-instrumentation-aiohttp-client/src/opentelemetry/instrumentation/aiohttp_client/__init__.py#L268>
- 在非采样 span 上的 `is_recording()` 优化
  - 例如 <https://github.com/alibaba/loongsuite-python/blob/2518a4ac07cb62ad6587dd8f6cbb5f8663a7e179/instrumentation/opentelemetry-instrumentation-requests/src/opentelemetry/instrumentation/requests/__init__.py#L234>
- 适当的错误处理
  - 例如 <https://github.com/alibaba/loongsuite-python/blob/2518a4ac07cb62ad6587dd8f6cbb5f8663a7e179/instrumentation/opentelemetry-instrumentation-requests/src/opentelemetry/instrumentation/requests/__init__.py#L220>
- 隔离同步和异步测试
  - 对于同步测试，典型的测试用例类继承自 `opentelemetry.test.test_base.TestBase`。但是，如果您想编写异步测试，测试用例类还应继承自 `IsolatedAsyncioTestCase`。将异步测试添加到公共测试类可能导致测试通过但实际上没有运行，这可能会产生误导。
  - 例如 <https://github.com/alibaba/loongsuite-python/blob/60fb936b7e5371b3e5587074906c49fb873cbd76/instrumentation/opentelemetry-instrumentation-grpc/tests/test_aio_server_interceptor.py#L84>
- 大多数 instrumentation 具有相同的版本。如果您要开发新的 instrumentation，它可能具有 `X.Y.dev` 版本，并依赖于 `opentelemetry-instrumentation` 和 `opentelemetry-semantic-conventions` 的[兼容版本](https://peps.python.org/pep-0440/#compatible-release)。这意味着您可能需要从 git 安装此仓库的 instrumentation 依赖项和核心仓库的依赖项。
- 文档
  - 添加新 instrumentation 时，请记住在 `docs/instrumentation/` 中添加名为 `<instrumentation>/<instrumentation>.rst` 的条目，以便从索引中引用 instrumentation 文档。您可以使用[此处](./_template/autodoc_entry.rst)提供的条目模板
- 测试
  - 添加新 instrumentation 时，请记住更新 `tox.ini`，在 `envlist`、`command_pre` 和 `commands` 部分添加适当的规则

### 更新支持的 instrumentation 包版本

- 导航到 **instrumentation 包目录：**
  - 通过修改 `[project.optional-dependencies]` 部分中的 `instruments` 或 `instruments-any` 条目来更新 **`pyproject.toml`** 文件，使用新的版本约束
  - 在 instrumentation **`package.py`** 文件中使用新的版本约束更新 `_instruments` 或 `_instruments_any` 变量
- 在**项目目录的根目录**，运行 `tox -e generate` 以重新生成必要的文件

请注意，`instruments-any` 是一个可选字段，可以代替或补充 `instruments` 使用。虽然 `instruments` 是依赖项列表，_所有_ 这些依赖项都是 instrumentation 所期望的，但 `instruments-any` 是 _任何_ 但不是全部都是期望的列表。例如，以下条目需要 `util` 和 `common` 加上 `foo` 或 `bar` 中的任何一个，以便进行 instrumentation：
```
[project.optional-dependencies]
instruments = [
  "util ~= 1.0"
  "common ~= 2.0"
]
instruments-any = [
  "foo ~= 3.0"
  "bar ~= 4.0"
]
```

<!-- 有关 instruments-any 的详细信息，请参阅 https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3610 -->

如果您要添加对 instrumentation 包新版本的支持，请按照以下附加步骤操作：

- 在 **instrumentation 包目录：** 添加新的 test-requirements.txt 文件，其中包含测试所需的相应包版本
- 在**项目目录的根目录**：在 [tox.ini](./tox.ini) 中为包版本添加新的测试环境条目，并运行 `tox -e generate-workflows` 以相应地重新生成新的 workflow。在同一 [tox.ini](./tox.ini) 文件中，搜索 `opentelemetry-instrumentation-{package}/test-requirements` 并添加一行指向您在上一步中创建的新 test-requirements.txt，以便 tox 可以安装正确的依赖项。

示例 PR：[#2976](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/2976)、[#2845](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/2845)

## GenAI Instrumentation 指南

与[生成式 AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) 系统相关的 Instrumentation 同样放置在 [instrumentation-loongsuite](./instrumentation-loongsuite) 文件夹中。本节涵盖与这些 instrumentation 相关的贡献。请注意，[instrumentation 指南](#instrumentation-指南)和[对贡献者的期望](#对贡献者的期望)仍然适用。

### 参与

* Review PR：如果您希望在新 PR 中标记为与这些 instrumentation 相关的审查者，请提交 PR 以将您的 GitHub 用户名添加到 [component_owners.yml](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/.github/component_owners.yml) 中相应的 instrumentation 文件夹下。

* Approve PR：任意来源的 approve 都是允许的，您可以通过 approve 来推动 PR 的合并进程，但在合并事实达成之前，仍需 Maintainer 之一进行 approve。

* 跟踪和创建问题：要跟踪与生成式 AI 相关的问题，请在创建或搜索问题时过滤或添加标签 [genai](https://github.com/alibaba/loongsuite-python/issues?q=is%3Aissue%20state%3Aopen%20label%3Agenai)。如果您没有看到与您想要贡献的 instrumentation 相关的问题，请创建一个新的跟踪问题，以便社区了解其进展。

## 对贡献者的期望

LoongSuite 是一个开源社区，因此非常鼓励任何对项目感兴趣的人做出贡献。话虽如此，即使在拉取请求合并后，对贡献者也有一定程度的期望。LoongSuite Python 社区期望贡献者对他们贡献的 instrumentation 保持一定程度的支持和兴趣，这是为了确保 instrumentation 不会变得过时，并且仍然按照原始贡献者的意图工作。某些 instrumentation 还涉及社区当前成员不太熟悉的库，因此有必要依赖原始贡献方的专业知识。

## 更新支持的 Python 版本

### 提升 Python 基线版本

更新最低支持的 Python 版本时，请记住：

- 从 `pyproject.toml` trove 分类器中删除该版本
- 从 `tox.ini` 中删除该版本
- 使用 `tox -e generate-workflows` 相应地更新 github workflow
- 搜索 `sys.version_info` 的使用并删除不支持版本的代码
- 在 `.pylintrc` 中提升 `py-version` 以进行 Python 版本相关的检查

### 添加对新 Python 版本的支持

添加对新 Python 版本的支持时，请记住：

- 在 `tox.ini` 中添加该版本
- 在 `pyproject.toml` trove 分类器中添加该版本
- 使用 `tox -e generate-workflows` 相应地更新 github workflow；lint 和 benchmarks 使用最新支持的版本
- 更新 `.pre-commit-config.yaml`
- 更新文档中的 tox 示例

