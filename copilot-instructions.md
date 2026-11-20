# LoongSuite Python Instrumentation Code Review Guidelines

This repository uses OpenTelemetry to collect telemetry from Python applications. Apply the following guidelines when reviewing code and suggesting changes.

## 1. Prefer framework extension points

Use the framework's own extension mechanisms where possible, such as ASGI middleware. Avoid monkey-patching individual methods because a method available in one framework version may disappear in the next.

## 2. Avoid adding third-party dependencies

Reuse existing dependencies where possible. For example, use `aiohttp` if it already provides the functionality that would otherwise require adding `httpx`. As a general rule, vendor the necessary source code if a third-party library is needed. The `requests` library is an exception.

## 3. Preserve application behavior

Instrumentation must preserve the application's behavior. If the application would raise an exception, instrumentation must allow it to propagate.

## 4. Keep instrumentation lightweight

Minimize CPU overhead. Do not use `deepcopy`. On the request path, do not log at `INFO` or higher. Use `logger.debug` for diagnostics so frequent log messages incur minimal CPU and I/O overhead when debug logging is disabled.

## 5. Check for resource leaks

### 5.1 Reference cycles and finalizers

Reference counting alone cannot reclaim cycles. Python relies on its cyclic garbage collector to reclaim them. Python 2 and early Python 3 releases could leave cycles containing objects with `__del__` uncollected. Python 3.4 improved this behavior. Where cycles are possible, avoid defining `__del__` or break the references explicitly.

### 5.2 Unbounded globals and module-level caches

Global containers and caches can retain objects for the lifetime of the process. Check for hand-written caches without an eviction policy and unbounded uses of `functools.lru_cache`. Specify a `maxsize` or clear the cache periodically.

### 5.3 Closures retaining outer variables

A closure can unintentionally keep a large object alive. Refactor the closure to avoid capturing the object or explicitly delete the reference when it is no longer needed.

### 5.4 Memory management in third-party C extensions

Python's garbage collector cannot release native memory that a C extension fails to free. This can occur in extensions used by NumPy, PyTorch, or OpenCV. Process memory may keep growing even when `tracemalloc` and `pympler` show no corresponding growth.

Check the extension's documentation and available updates. Use `objgraph`, `memory_profiler`, or a system tool such as Valgrind to investigate.

### 5.5 Unclosed files, connections, and threads

Unreleased resources can retain memory or system handles. For example:

```python
f = open("huge_file.txt")
# If f.close() is omitted, the file handle can remain open.
```

Use a `with` statement to manage resources that support it.

### 5.6 Retained exception tracebacks

A traceback holds stack frames, which in turn hold references to local variables. Retaining the traceback can keep an entire scope alive.

```python
import sys

exc_info = None
try:
    raise ValueError("oops")
except Exception:
    exc_info = sys.exc_info()  # Includes the traceback and its frames.
```

Avoid storing `exc_info` long term. Clear it after use or store the string returned by `traceback.format_exc()`.

## 6. Threads and scheduled tasks

Do not create threads or scheduled tasks. If one is essential, assess its resource cost carefully.

## 7. Check metric attribute cardinality

Check how many distinct values each metric attribute can take. High-cardinality attributes can produce an excessive number of time series. Pay particular attention to `endpoint`, `destId`, and `rpc`.

## 8. Limit instrumentation to the target framework

A plugin must instrument only classes belonging to its target framework. For example, CrewAI depends on LiteLLM. Instrumenting LiteLLM classes belongs outside the CrewAI plugin.

## 9. Preserve exception tracebacks

Use a bare `raise` when re-raising an exception to preserve its traceback. Avoid `raise e`, which changes the traceback at the re-raise site.

## 10. Thread safety

### 10.1 Protect singleton creation with a lock

Concurrent callers can create multiple instances if the check and initialization are unprotected. Guard singleton creation with a lock.

### 10.2 Protect shared global state

Concurrent writes to global variables can cause races. Protect shared reads and writes with a lock, or encapsulate configuration in a singleton.

## 11. Imports

### 11.1 Import order and placement

Follow PEP 8. Group standard-library imports first, third-party imports second, and local imports last. Separate groups with a blank line.

Keep imports at the top of the file, following Ruff rule `PLC0415`. If a local import is necessary, explain why in a comment and add the inline suppression `# noqa: PLC0415`.

### 11.2 Avoid duplicate imports

Check that each module is imported only once per file.

### 11.3 Optional dependencies

Use lazy imports or a guarded `try`/`except` import for optional libraries. The instrumentation module should still load when the target framework is absent.

## 12. Exception handling and diagnostics

### 12.1 Avoid overly broad exception handlers

Do not catch `BaseException`. It also catches `KeyboardInterrupt` and `SystemExit`. Use `Exception` when a general exception handler is appropriate.

### 12.2 Use a logger for diagnostics

Use the logger instead of `print`.

## 13. Python version compatibility

Avoid syntax that requires a Python version newer than 3.8.

## 14. Code formatting

End each Python file with a newline. Separate top-level functions and classes with two blank lines. Use spaces around assignment operators. Follow existing enum naming conventions and choose descriptive variable names.

## 15. Unit test coverage

Cover new functionality with tests for core behavior, boundary conditions, and error paths. Tests should verify the observable effect of the change. For a new plugin, verify the complete trace and the emitted metrics. Exercise realistic workloads and keep mocking to a minimum.

## 16. Dictionary and cache memory management

### 16.1 Bound dictionary growth

Set a maximum size, expire entries periodically, or use `WeakValueDictionary` where appropriate so the garbage collector can reclaim unused values.

## 17. Spelling

Check for spelling errors.

## 18. Verify that methods exist

After refactoring, check that every method call still resolves. Renamed or removed methods can otherwise cause an `AttributeError` at runtime.

## 19. Match instrumentation setup and teardown

`uninstrument` must release resources acquired by `instrument` and undo setup operations in reverse order.

## 20. Comments

For key methods, consider including an input/output example. Remove leftover development notes, comments that add no information, and commented-out code.

## 21. Pull requests

Use a Conventional Commits title and link the PR to a work item.

## 22. Chinese and English terminology

Apply these conventions in all comments and documentation. Translate the noun “探针” as **instrumentation** (avoid “probe”) and the verb “埋点” as **instrument**.
