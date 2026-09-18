# RTP-LLM (fork: ningweikang/rtp-llm, PR #25 aclgraph adaptation)

Alibaba RTP-LLM is a production LLM inference engine (C++ core + Python bindings,
built with Bazel). This checkout is on branch `pr-25` (fetched from
`refs/pull/25/head`), which adds Ascend ACL Graph support on top of `main`.

## Repo layout

- `rtp_llm/cpp/` — C++ engine core (bazel `cc_library` targets)
  - `rtp_llm/cpp/ascend_graph/` — **PR #25 feature**: Ascend ACL Graph runner
    (`ascend_graph_runner.cc/h`), device shims (`ascend_graph_device_shims.cc/h`),
    utils. Mirrors `cuda_graph` but wraps `c10_npu::NPUGraph`; compiles to no-op
    stubs on non-Ascend platforms so it is safe to depend on unconditionally.
  - `rtp_llm/cpp/cuda_graph/` — CUDA Graph equivalent (base classes reused by ascend_graph)
  - `rtp_llm/cpp/pybind/` — C++/Python bindings (`PyWrappedModel.h` touched by PR #25)
  - `rtp_llm/cpp/cache/` — KV cache management
  - `rtp_llm/cpp/models/` — model implementations
- `rtp_llm/models/` — Python model definitions
  - `rtp_llm/models/factory/attention/ascend_impl/ascend_decode.py` — **PR #25**: Ascend decode
    implementation (+201 lines, FIA graph forward)
- `rtp_llm/ops/` — Python op wrappers
- `deps/` — pip requirements (`requirements_*_ascend.txt` touched by PR #25)
- `docs/` — sphinx docs (built HTML under `docs/build/en/`)
- `benchmark/`, `example/`, `docker/`, `tools/`, `scripts/`

## Build

- Bazel 6.4.0 via bazelisk (see `.bazeliskrc`; Huawei Cloud mirror configured).
- Typical targets:
  - `bazel build //rtp_llm/cpp/ascend_graph:ascend_graph` (Ascend only)
  - `bazel build //rtp_llm:sdk` — full Python SDK (needs CUDA + deps installed,
    pip deps in `deps/requirements_torch_gpu_cuda12.txt`)
  - `bazel test //rtp_llm/test:generate_config_test` — example bazel py_test
- Platform selection via `@arch_config` (see `def.bzl`, `arch_config/`).
- Prebuilt stubs live in `rtp_llm/ops`; when regenerated, copy only modified
  parts from `bazel-bin/stubs`.

## Lint / format

- `.pre-commit-config.yaml`: black + isort (python), clang-format v19 (C/C++/CUDA).
  `rtp_llm/ops` and `3rdparty` are excluded; `rtp_llm/cpp/cutlass/` excluded from clang-format.
- `.clang-format` at repo root for C++ style.

## Tests

- Bazel `py_test`/`cc_test` targets (GPU `exec_properties` are set per target).
- Python unit tests live in `rtp_llm/test/`, C++ tests in `rtp_llm/cpp/test/`.

## Conventions / notes

- C++ copts come from `load("//:def.bzl", "copts")`.
- PR #25 pattern: new device modules should compile to no-op stubs on other
  platforms (see `ascend_graph_device_shims.cc` guard style) so they can be
  depended on unconditionally.
- `deps/http.bzl` and `requirements_*_ascend.txt` were updated by PR #25 —
  check Bazel's http_archive caching when those change.
- Do not run tests from inside the repo root if the wheel-installed `rtp_llm`
  package shadows the local one (README_cn.md FAQ #2).
