from candle._dispatch.registry import registry
from candle._dispatch.schema import OpSchema


def test_schema_parses_view_alias_on_input_and_return():
    schema = OpSchema("view(Tensor(a) input, int[] shape) -> Tensor(a)")
    assert schema.params[0].alias_set == "a"
    assert schema.returns[0].alias_set == "a"


def test_core_view_like_schemas_expose_alias_sets():
    for name in ("view", "reshape", "transpose"):
        entry = registry.get(f"aten::{name}")
        assert entry.schema_obj is not None
        assert entry.schema_obj.params[0].alias_set == "a"
        assert entry.schema_obj.returns[0].alias_set == "a"


def test_core_inplace_schemas_expose_return_alias_sets():
    for name in ("add_", "mul_", "relu_", "zero_"):
        entry = registry.get(f"aten::{name}")
        assert entry.schema_obj is not None
        assert entry.schema_obj.params[0].alias_set == "a"
        assert entry.schema_obj.returns[0].alias_set == "a"


def test_dispatch_mutating_args_include_aliasless_mutation():
    from candle._dispatch.dispatcher import _mutating_args

    schema = OpSchema("foo(Tensor(a!) x, Tensor(!) y, Tensor z) -> Tensor")
    mutated = _mutating_args(schema, (1, 2, 3), {})
    assert mutated == [1, 2]


def test_functionalize_writeback_supports_aliasless_mutation():
    import uuid

    import candle as torch
    from candle._dispatch.dispatcher import dispatch
    from candle._dispatch.functionalize import functionalize_context
    from candle._dispatch.keys import DispatchKey

    token = uuid.uuid4().hex
    mutate_name = f"aliasless_fill_{token}_"
    functional_name = mutate_name[:-1]

    registry.register_schema(mutate_name, f"{mutate_name}(Tensor(!) self) -> Tensor")
    registry.register_schema(functional_name, f"{functional_name}(Tensor input) -> Tensor")
    registry.register_kernel(
        functional_name,
        DispatchKey.CPU,
        lambda x: torch.full(x.shape, 5.0, dtype=x.dtype),
    )

    x = torch.tensor([1.0, 2.0])
    before = x._version_counter.value

    with functionalize_context():
        out = dispatch(mutate_name, x.device.type, x)

    assert out is x
    assert x.tolist() == [5.0, 5.0]
    assert x._version_counter.value == before + 1
