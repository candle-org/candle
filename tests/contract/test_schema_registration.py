import candle as torch
from candle._dispatch.registry import registry


def test_inplace_schema_registered():
    entry = registry.get("aten::add_")
    assert entry.schema_obj is not None
    assert any(param.mutates for param in entry.schema_obj.params)


def test_mul_schema_registered():
    entry = registry.get("aten::mul")
    assert entry.schema_obj is not None
    assert entry.schema == "mul(Tensor input, Tensor other) -> Tensor"
