from candle._backends.npu import ops_soc


def test_soc_policy_profile_mapping_contains_expected_profiles():
    assert "allclose" in ops_soc.fallback_ops("910a")
    assert "std" in ops_soc.fallback_ops("910b")
    assert ops_soc.fallback_ops("310p") == frozenset()
    assert "remainder" in ops_soc.fallback_ops("310b")


def test_soc_policy_case_insensitive_profile_name():
    assert ops_soc.use_fallback("remainder", profile="310B")
    assert not ops_soc.use_fallback("remainder", profile="910A")


def test_soc_policy_unknown_profile_returns_safe_default():
    assert ops_soc.fallback_ops("unknown") == frozenset()
    assert not ops_soc.use_fallback("uniform_", profile="unknown")


def test_soc_capability_table_routes_smallop_arange_for_310b_only():
    assert ops_soc.use_smallop_arange_1d(profile="310b")
    assert not ops_soc.use_smallop_arange_1d(profile="910a")
    assert not ops_soc.use_smallop_arange_1d(profile="910b")
    assert not ops_soc.use_smallop_arange_1d(profile="310p")


def test_soc_capability_table_routes_smallop_linspace_for_310b_910a_and_910b():
    assert ops_soc.use_smallop_linspace(profile="310b")
    assert ops_soc.use_smallop_linspace(profile="910a")
    assert ops_soc.use_smallop_linspace(profile="910b")
    assert not ops_soc.use_smallop_linspace(profile="310p")


def test_soc_capability_table_routes_safe_int64_index_compare_for_310b_only():
    assert ops_soc.capability("use_safe_int64_index_compare", profile="310b")
    assert not ops_soc.capability("use_safe_int64_index_compare", profile="910a")
    assert not ops_soc.capability("use_safe_int64_index_compare", profile="910b")
    assert not ops_soc.capability("use_safe_int64_index_compare", profile="310p")


def test_soc_capability_table_routes_aclgraph_support_for_910b_only():
    assert not ops_soc.aclgraph_supported(profile="910a")
    assert ops_soc.aclgraph_supported(profile="910b")
    assert not ops_soc.aclgraph_supported(profile="310b")
    assert not ops_soc.aclgraph_supported(profile="310p")


def test_soc_capability_unknown_profile_uses_default_value():
    assert not ops_soc.capability("use_smallop_arange_1d", profile="unknown")
    assert ops_soc.capability("missing_key", profile="unknown", default=True)


def test_soc_910b_fallback_ops_cover_expected_watchlist_set():
    expected = {
        "std",
        "nansum",
        "instance_norm",
        "avg_pool2d",
        "adaptive_avg_pool2d",
        "upsample_nearest1d",
        "einsum",
        "isinf",
        "im2col",
    }
    got = set(ops_soc.fallback_ops("910b"))
    assert got == expected


def test_soc_910a_fallback_ops_cover_expected_watchlist_set():
    expected = {
        "allclose",
        "isinf",
        "frac",
        "gather",
        "matmul",
        "addmm",
        "mv",
        "repeat_interleave_tensor",
    }
    got = set(ops_soc.fallback_ops("910a"))
    assert got == expected


def test_soc_310b_fallback_ops_cover_expected_watchlist_set():
    expected = {
        "allclose",
        "isinf",
        "dot",
        "matmul",
        "addmm",
        "mv",
        "remainder",
        "where",
        "atan2",
        "lerp",
        "dropout",
        "softplus",
        "isclose",
        "flip",
        "argsort",
        "sort",
        "topk",
        "diag",
        "gather",
        "take_along_dim",
        "layer_norm",
        "mish",
        "batch_norm",
        "avg_pool2d",
        "adaptive_avg_pool2d",
        "einsum",
        "allclose",
    }
    got = set(ops_soc.fallback_ops("310b"))
    assert got == expected
