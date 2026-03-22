import importlib
import sys
import uuid

import pytest

from .helpers import assert_torch_error


def test_pipeline_requires_meta_kernel_error():
    from candle._dispatch import pipeline
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry

    op_name = f"pipeline_no_meta_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}() -> Any")
    registry.register_kernel(op_name, DispatchKey.CPU, lambda: None)

    def mt():
        with pipeline.pipeline_context():
            dispatch(op_name, "cpu")

    def th():
        raise RuntimeError(f"pipeline requires meta kernel for op {op_name}")

    assert_torch_error(mt, th)


def test_pipeline_last_error_structured_payload():
    from candle._dispatch import pipeline
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry

    op_name = f"pipeline_error_payload_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}() -> Any")
    def _meta():
        from candle._backends.meta.infer import TensorSpec
        from candle._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom():
        raise RuntimeError("boom-from-kernel")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    with pipeline.pipeline_context(debug_enabled=True) as pipe:
        dispatch(op_name, "cpu")
        try:
            pipe.flush()
        except RuntimeError:
            pass

        err = pipe.last_error()
        assert err is not None
        payload = err.to_dict()
        assert payload["op_name"] == op_name
        assert payload["phase"] == "submit"
        assert payload["backend"] in {"cpu", "npu", "meta"}
        assert isinstance(payload["op_seq"], int)
        assert "callsite" in payload
        assert "read_set" in payload
        assert "write_set" in payload
        assert "alias_set" in payload
        assert "version_plan" in payload
        assert payload["read_set"] == []
        assert payload["write_set"] == []
        assert "dependency_edges" in payload
        assert "runtime_code" in payload
        assert "suppressed_errors" in payload


def test_pipeline_error_debug_interfaces():
    from candle._dispatch import pipeline
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry

    op_name = f"pipeline_error_debug_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}() -> Any")
    def _meta():
        from candle._backends.meta.infer import TensorSpec
        from candle._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom():
        raise RuntimeError("boom-for-debug")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    with pipeline.pipeline_context(debug_enabled=True) as pipe:
        dispatch(op_name, "cpu")
        assert pipe.pending_count() == 1
        try:
            pipe.flush()
        except RuntimeError:
            pass

        short = pipe.format_error("short")
        full = pipe.format_error("full")
        dump = pipe.debug_dump(failed_only=True)

        assert "boom-for-debug" in short
        assert "op_name" in full
        assert dump["last_error"] is not None
        assert len(dump["entries"]) == 1


def test_pipeline_error_id_is_deterministic_for_same_site():
    from candle._dispatch import pipeline
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry

    op_name = f"pipeline_error_stable_id_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}() -> Any")
    def _meta():
        from candle._backends.meta.infer import TensorSpec
        from candle._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom():
        raise RuntimeError("stable-id-error")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    ids = []
    for _ in range(2):
        with pipeline.pipeline_context(debug_enabled=True) as pipe:
            dispatch(op_name, "cpu")
            try:
                pipe.flush()
            except RuntimeError:
                pass
            ids.append(pipe.last_error().to_dict()["error_id"])

    assert ids[0] == ids[1]


def test_pipeline_error_payload_captures_mutating_alias_set():
    from candle._dispatch import pipeline
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry

    op_name = f"pipeline_error_mutate_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}(Tensor(a!) self) -> Tensor")

    def _meta(x):
        from candle._backends.meta.infer import TensorSpec
        from candle._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom(x):
        raise RuntimeError("mutate-boom")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    import candle as torch

    x = torch.tensor([1.0])
    with pipeline.pipeline_context(debug_enabled=True) as pipe:
        dispatch(op_name, "cpu", x)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        payload = pipe.last_error().to_dict()
        assert payload["write_set"] == ["self"]
        assert payload["alias_set"] == "a"
        assert payload["version_plan"].get("a") == 1


def test_pipeline_version_plan_counts_multiple_mutations():
    from candle._dispatch import pipeline
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry
    import candle as torch

    op_name = f"pipeline_error_multi_mutate_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}(Tensor(a!) a, Tensor(a!) b) -> Tensor")

    def _meta(a, b):
        from candle._backends.meta.infer import TensorSpec
        from candle._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op_name, DispatchKey.Meta, _meta)

    def _boom(a, b):
        raise RuntimeError("multi-mutate-boom")

    registry.register_kernel(op_name, DispatchKey.CPU, _boom)

    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    with pipeline.pipeline_context(debug_enabled=True) as pipe:
        dispatch(op_name, "cpu", a, b)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        payload = pipe.last_error().to_dict()
        assert payload["version_plan"].get("a") == 2


def test_pipeline_dependency_edges_capture_write_write():
    from candle._dispatch import pipeline
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry
    import candle as torch

    op1 = f"pipeline_dep_ww_ok_{uuid.uuid4().hex}"
    op2 = f"pipeline_dep_ww_fail_{uuid.uuid4().hex}"

    registry.register_schema(op1, f"{op1}(Tensor! self) -> Tensor")
    registry.register_schema(op2, f"{op2}(Tensor! self) -> Tensor")

    def _meta(x):
        from candle._backends.meta.infer import TensorSpec
        from candle._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op1, DispatchKey.Meta, _meta)
    registry.register_kernel(op2, DispatchKey.Meta, _meta)

    def _ok(x):
        return x

    def _boom(x):
        raise RuntimeError("ww-dep-boom")

    registry.register_kernel(op1, DispatchKey.CPU, _ok)
    registry.register_kernel(op2, DispatchKey.CPU, _boom)

    x = torch.tensor([1.0])
    with pipeline.pipeline_context(debug_enabled=True) as pipe:
        dispatch(op1, "cpu", x)
        dispatch(op2, "cpu", x)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        deps = pipe.last_error().to_dict()["dependency_edges"]
        assert any(
            edge["from"] == 0 and edge["to"] == 1 and "write->write" in edge["reason"]
            for edge in deps
        )


def test_pipeline_dependency_edges_capture_alias_writes():
    from candle._dispatch import pipeline
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry
    import candle as torch

    op1 = f"pipeline_dep_write_ok_{uuid.uuid4().hex}"
    op2 = f"pipeline_dep_write_fail_{uuid.uuid4().hex}"

    registry.register_schema(op1, f"{op1}(Tensor(a!) self) -> Tensor")
    registry.register_schema(op2, f"{op2}(Tensor(a!) self) -> Tensor")

    def _meta(x):
        from candle._backends.meta.infer import TensorSpec
        from candle._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op1, DispatchKey.Meta, _meta)
    registry.register_kernel(op2, DispatchKey.Meta, _meta)

    def _ok(x):
        return x

    def _boom(x):
        raise RuntimeError("dep-boom")

    registry.register_kernel(op1, DispatchKey.CPU, _ok)
    registry.register_kernel(op2, DispatchKey.CPU, _boom)

    x = torch.tensor([1.0])
    with pipeline.pipeline_context(debug_enabled=True) as pipe:
        dispatch(op1, "cpu", x)
        dispatch(op2, "cpu", x)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        payload = pipe.last_error().to_dict()
        deps = payload["dependency_edges"]
        assert deps
        assert deps[0]["from"] == 0
        assert deps[0]["to"] == 1
        assert "alias_set:a" in deps[0]["reason"]


def test_pipeline_dependency_edges_capture_tensor_rw():
    from candle._dispatch import pipeline
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry
    import candle as torch

    op1 = f"pipeline_dep_rw_ok_{uuid.uuid4().hex}"
    op2 = f"pipeline_dep_rw_fail_{uuid.uuid4().hex}"

    registry.register_schema(op1, f"{op1}(Tensor! self) -> Tensor")
    registry.register_schema(op2, f"{op2}(Tensor self) -> Tensor")

    def _meta(x):
        from candle._backends.meta.infer import TensorSpec
        from candle._dtype import float32

        return TensorSpec(shape=(1,), stride=(1,), dtype=float32)

    registry.register_kernel(op1, DispatchKey.Meta, _meta)
    registry.register_kernel(op2, DispatchKey.Meta, _meta)

    def _ok(x):
        return x

    def _boom(x):
        raise RuntimeError("rw-dep-boom")

    registry.register_kernel(op1, DispatchKey.CPU, _ok)
    registry.register_kernel(op2, DispatchKey.CPU, _boom)

    x = torch.tensor([1.0])
    with pipeline.pipeline_context(debug_enabled=True) as pipe:
        dispatch(op1, "cpu", x)
        dispatch(op2, "cpu", x)
        try:
            pipe.flush()
        except RuntimeError:
            pass

        deps = pipe.last_error().to_dict()["dependency_edges"]
        assert any(
            edge["from"] == 0 and edge["to"] == 1 and "write->read" in edge["reason"]
            for edge in deps
        )


def test_dispatcher_module_exports_cython_dispatch_entrypoints():
    """Contract: candle._dispatch.dispatcher must expose the Cython entrypoints
    (cy_dispatch_full, cy_dispatch_with_keyset_fast) as public module-level
    attributes so callers can introspect whether the accelerated path is
    active.  Currently the dispatcher only tries the import inside a
    try/except and never re-exports the names, so this test is expected RED
    until the export surface is made explicit.
    """
    from candle._dispatch import dispatcher

    assert hasattr(dispatcher, "cy_dispatch_full"), (
        "candle._dispatch.dispatcher must export cy_dispatch_full at module "
        "level so consumers can detect whether the Cython hot-path is active."
    )
    assert hasattr(dispatcher, "cy_dispatch_with_keyset_fast"), (
        "candle._dispatch.dispatcher must export cy_dispatch_with_keyset_fast "
        "at module level."
    )
    assert callable(dispatcher.cy_dispatch_full), (
        "candle._dispatch.dispatcher.cy_dispatch_full must be callable."
    )
    assert callable(dispatcher.cy_dispatch_with_keyset_fast), (
        "candle._dispatch.dispatcher.cy_dispatch_with_keyset_fast must be callable."
    )


def test_runtime_core_import_failure_is_actionable(monkeypatch):
    """Contract: when candle._cython._dispatcher_core is absent the dispatcher
    must raise an ImportError whose message explicitly names the missing
    extension and tells the user how to build it (e.g. 'pip install -e .[cython]'
    or 'python setup.py build_ext --inplace').  A silent fallback or a raw
    low-level ImportError with no guidance is not acceptable once runtime-core
    policy is codified.

    This test is expected RED until the dispatcher wraps the hard-import with
    an actionable error message.
    """
    monkeypatch.setitem(sys.modules, "candle._cython._dispatcher_core", None)
    monkeypatch.delitem(sys.modules, "candle._dispatch.dispatcher", raising=False)

    try:
        importlib.import_module("candle._dispatch.dispatcher")
    except ImportError as exc:
        msg = str(exc)
        assert "_dispatcher_core" in msg, (
            "ImportError must name the missing extension '_dispatcher_core', "
            f"got: {msg!r}"
        )
        assert any(
            hint in msg
            for hint in ("build_ext", "pip install", "setup.py", "[cython]")
        ), (
            "ImportError must include a build hint so the user knows how to "
            f"recover, got: {msg!r}"
        )
        return

    pytest.fail(
        "When _dispatcher_core is absent the dispatcher must raise an "
        "actionable ImportError, but import_module() completed without "
        "raising. The dispatcher must not silently fall back to the "
        "pure-Python path."
    )
