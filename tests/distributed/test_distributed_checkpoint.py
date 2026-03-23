"""Tests for distributed checkpoint shard-aware save/load (Task 9).

Scope:
 - DDP save/load round-trip via legacy get_state_dict/set_state_dict
 - DDP save/load round-trip via DCP save/load with FileSystem backend
 - FSDP-aware get_state_dict / set_state_dict (full and sharded type)
 - set_optimizer_state_dict (gap: previously missing)
 - Stateful protocol: state_dict / load_state_dict interface
 - DCP save/load with Stateful objects
 - Per-rank shard metadata is captured in DCP Metadata

All tests run without HCCL/NCCL (no_dist=True or rank-mock patterns).
"""
import os
import sys
import types
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import candle
import candle.nn as nn
import candle.distributed.checkpoint as dcp
from candle.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    save as legacy_save,
    load as legacy_load,
)
from candle.distributed.checkpoint.stateful import Stateful


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)


class _DDPWrapper:
    """Minimal DDP-like wrapper exposing .module."""
    def __init__(self, module):
        self.module = module

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd, strict=True):
        return self.module.load_state_dict(sd, strict=strict)


class _StubOptimizer:
    def __init__(self, params):
        self.param_groups = [{'params': list(params), 'lr': 0.01}]
        self.state = {}

    def state_dict(self):
        return {
            'state': {},
            'param_groups': [{'lr': pg['lr']} for pg in self.param_groups],
        }

    def load_state_dict(self, sd):
        for pg, saved_pg in zip(self.param_groups, sd.get('param_groups', [])):
            pg.update({k: v for k, v in saved_pg.items() if k != 'params'})


# ---------------------------------------------------------------------------
# 1. Legacy DDP get_state_dict / set_state_dict
# ---------------------------------------------------------------------------

class TestLegacyDDPStateDict:
    def test_plain_model_returns_four_keys(self):
        m = _SimpleModel()
        sd, opt_sd = get_state_dict(m)
        assert set(sd.keys()) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}
        assert opt_sd is None

    def test_with_optimizer_returns_optim_sd(self):
        m = _SimpleModel()
        opt = _StubOptimizer(m.parameters())
        _, opt_sd = get_state_dict(m, opt)
        assert 'param_groups' in opt_sd

    def test_ddp_wrapper_unwrapped(self):
        m = _SimpleModel()
        ddp = _DDPWrapper(m)
        sd, _ = get_state_dict(ddp)
        assert set(sd.keys()) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}

    def test_set_state_dict_restores_weights(self):
        a = _SimpleModel()
        b = _SimpleModel()
        sd_a, _ = get_state_dict(a)
        result = set_state_dict(b, model_state_dict=sd_a)
        assert result['loaded_keys_count'] == 4
        for key in sd_a:
            np.testing.assert_array_equal(
                sd_a[key].detach().numpy(),
                b.state_dict()[key].detach().numpy(),
            )

    def test_set_state_dict_restores_optimizer(self):
        m = _SimpleModel()
        opt = _StubOptimizer(m.parameters())
        _, opt_sd = get_state_dict(m, opt)
        result = set_state_dict(m, opt, optim_state_dict=opt_sd)
        assert 'restored_optimizer_state_keys_count' in result

    def test_rank0_only_non_zero_returns_none(self):
        m = _SimpleModel()
        result = get_state_dict(m, rank0_only=True, rank=1)
        assert result == (None, None)

    def test_rank0_only_rank0_returns_real_sd(self):
        m = _SimpleModel()
        sd, _ = get_state_dict(m, rank0_only=True, rank=0)
        assert len(sd) == 4


# ---------------------------------------------------------------------------
# 2. Legacy save / load file round-trip
# ---------------------------------------------------------------------------

class TestLegacySaveLoad:
    def test_roundtrip_plain_model(self, tmp_path):
        a = _SimpleModel()
        path = str(tmp_path / 'ckpt.pt')
        legacy_save(path, a)

        b = _SimpleModel()
        legacy_load(path, b)

        for key in a.state_dict():
            np.testing.assert_array_equal(
                a.state_dict()[key].detach().numpy(),
                b.state_dict()[key].detach().numpy(),
            )

    def test_roundtrip_with_optimizer(self, tmp_path):
        m = _SimpleModel()
        opt = _StubOptimizer(m.parameters())
        path = str(tmp_path / 'ckpt.pt')
        legacy_save(path, m, opt)

        m2 = _SimpleModel()
        opt2 = _StubOptimizer(m2.parameters())
        result = legacy_load(path, m2, opt2)
        assert result['loaded_keys_count'] == 4

    def test_rank0_only_nonzero_writes_nothing(self, tmp_path):
        m = _SimpleModel()
        path = str(tmp_path / 'should_not_exist.pt')
        result = legacy_save(path, m, rank0_only=True, rank=1)
        assert result is None
        assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# 3. DCP save / load round-trip (no_dist=True)
# ---------------------------------------------------------------------------

class TestDCPSaveLoad:
    def test_dcp_roundtrip_plain_tensors(self, tmp_path):
        m = _SimpleModel()
        src = {k: v.detach() for k, v in m.state_dict().items()}
        ckpt = str(tmp_path / 'ckpt')
        dcp.save(src, checkpoint_id=ckpt, no_dist=True)

        m2 = _SimpleModel()
        dst = {k: v.detach() for k, v in m2.state_dict().items()}
        dcp.load(dst, checkpoint_id=ckpt, no_dist=True)

        for key in src:
            np.testing.assert_array_almost_equal(
                src[key].numpy(), dst[key].numpy(), decimal=6
            )

    def test_dcp_metadata_file_written(self, tmp_path):
        m = _SimpleModel()
        src = {k: v.detach() for k, v in m.state_dict().items()}
        ckpt = str(tmp_path / 'ckpt')
        dcp.save(src, checkpoint_id=ckpt, no_dist=True)
        assert os.path.exists(os.path.join(ckpt, '.metadata'))

    def test_dcp_per_rank_distcp_file_written(self, tmp_path):
        m = _SimpleModel()
        src = {k: v.detach() for k, v in m.state_dict().items()}
        ckpt = str(tmp_path / 'ckpt')
        dcp.save(src, checkpoint_id=ckpt, no_dist=True)
        distcp_files = [f for f in os.listdir(ckpt) if f.endswith('.distcp')]
        assert len(distcp_files) >= 1

    def test_dcp_metadata_tensor_sizes(self, tmp_path):
        m = _SimpleModel()
        src = {k: v.detach() for k, v in m.state_dict().items()}
        ckpt = str(tmp_path / 'ckpt')
        meta = dcp.save(src, checkpoint_id=ckpt, no_dist=True)
        assert meta is not None
        assert tuple(meta.state_dict_metadata['fc1.weight'].size) == (8, 4)
        assert tuple(meta.state_dict_metadata['fc1.bias'].size) == (8,)
        assert tuple(meta.state_dict_metadata['fc2.weight'].size) == (2, 8)

    def test_dcp_metadata_has_chunks(self, tmp_path):
        """Each tensor in metadata must have at least one chunk."""
        m = _SimpleModel()
        src = {k: v.detach() for k, v in m.state_dict().items()}
        ckpt = str(tmp_path / 'ckpt')
        meta = dcp.save(src, checkpoint_id=ckpt, no_dist=True)
        for fqn, tsm in meta.state_dict_metadata.items():
            if hasattr(tsm, 'chunks'):
                assert len(tsm.chunks) >= 1, f"{fqn}: no chunks"
                assert hasattr(tsm.chunks[0], 'offsets')
                assert hasattr(tsm.chunks[0], 'sizes')

    def test_dcp_chunk_offsets_zero_for_single_rank(self, tmp_path):
        """For a single rank, all chunk offsets should be (0, 0, ...)."""
        m = _SimpleModel()
        src = {k: v.detach() for k, v in m.state_dict().items()}
        ckpt = str(tmp_path / 'ckpt')
        meta = dcp.save(src, checkpoint_id=ckpt, no_dist=True)
        for fqn, tsm in meta.state_dict_metadata.items():
            if hasattr(tsm, 'chunks') and tsm.chunks:
                offsets = tsm.chunks[0].offsets
                assert all(o == 0 for o in offsets), (
                    f"{fqn}: expected zero offsets for rank-0 single-rank save, got {offsets}"
                )

    def test_dcp_overwrite_zeros(self, tmp_path):
        """DCP load overwrites all-zero tensors with saved values."""
        m = _SimpleModel()
        src = {k: v.detach() for k, v in m.state_dict().items()}
        ckpt = str(tmp_path / 'ckpt')
        dcp.save(src, checkpoint_id=ckpt, no_dist=True)

        dst = {k: candle.zeros_like(v) for k, v in m.state_dict().items()}
        # Pre-condition: values differ
        assert not np.allclose(src['fc1.weight'].numpy(), dst['fc1.weight'].numpy())

        dcp.load(dst, checkpoint_id=ckpt, no_dist=True)
        for key in src:
            np.testing.assert_array_almost_equal(
                src[key].numpy(), dst[key].numpy(), decimal=6
            )


# ---------------------------------------------------------------------------
# 4. Legacy get/set_state_dict FSDP-awareness
# ---------------------------------------------------------------------------

class TestLegacyFSDPAwareness:
    """get/set_state_dict detect _fsdp_state and delegate to FSDP helpers."""

    def _fsdp_model(self):
        m = _SimpleModel()
        m._fsdp_state = types.SimpleNamespace(param_group=None)
        return m

    def test_fsdp_detected_does_not_raise(self):
        m = self._fsdp_model()
        sd, _ = get_state_dict(m)
        assert isinstance(sd, dict)

    def test_fsdp_type_full_returns_all_keys(self):
        m = self._fsdp_model()
        sd, _ = get_state_dict(m, fsdp_type='full')
        assert set(sd.keys()) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}

    def test_fsdp_type_sharded_returns_all_keys(self):
        m = self._fsdp_model()
        sd, _ = get_state_dict(m, fsdp_type='sharded')
        assert set(sd.keys()) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}

    def test_fsdp_set_state_dict_full(self):
        a = _SimpleModel()
        sd_a = {k: v.detach() for k, v in a.state_dict().items()}
        b = self._fsdp_model()
        result = set_state_dict(b, model_state_dict=sd_a, fsdp_type='full')
        assert result['loaded_keys_count'] == 4

    def test_fsdp_set_state_dict_sharded(self):
        a = _SimpleModel()
        sd_a = {k: v.detach() for k, v in a.state_dict().items()}
        b = self._fsdp_model()
        result = set_state_dict(b, model_state_dict=sd_a, fsdp_type='sharded')
        assert result['loaded_keys_count'] == 4


# ---------------------------------------------------------------------------
# 5. set_optimizer_state_dict (gap fill)
# ---------------------------------------------------------------------------

class TestSetOptimizerStateDict:
    def test_importable(self):
        from candle.distributed._composable.fsdp._fsdp_state_dict import (
            set_optimizer_state_dict,
        )
        assert callable(set_optimizer_state_dict)

    def test_restores_lr(self):
        from candle.distributed._composable.fsdp._fsdp_state_dict import (
            get_optimizer_state_dict,
            set_optimizer_state_dict,
        )
        m = _SimpleModel()
        opt = _StubOptimizer(m.parameters())
        opt.param_groups[0]['lr'] = 0.001
        saved = get_optimizer_state_dict(m, opt)
        assert saved['param_groups'][0]['lr'] == 0.001

        opt.param_groups[0]['lr'] = 0.9
        set_optimizer_state_dict(m, opt, saved)
        assert opt.param_groups[0]['lr'] == 0.001

    def test_type_full(self):
        from candle.distributed._composable.fsdp._fsdp_state_dict import (
            get_optimizer_state_dict, set_optimizer_state_dict,
        )
        m = _SimpleModel()
        opt = _StubOptimizer(m.parameters())
        sd = get_optimizer_state_dict(m, opt, type='full')
        set_optimizer_state_dict(m, opt, sd, type='full')  # must not raise

    def test_type_sharded(self):
        from candle.distributed._composable.fsdp._fsdp_state_dict import (
            get_optimizer_state_dict, set_optimizer_state_dict,
        )
        m = _SimpleModel()
        opt = _StubOptimizer(m.parameters())
        sd = get_optimizer_state_dict(m, opt, type='sharded')
        set_optimizer_state_dict(m, opt, sd, type='sharded')  # must not raise


# ---------------------------------------------------------------------------
# 6. Stateful protocol
# ---------------------------------------------------------------------------

class TestStatefulProtocol:
    def test_has_state_dict_method(self):
        assert hasattr(Stateful, 'state_dict')

    def test_has_load_state_dict_method(self):
        assert hasattr(Stateful, 'load_state_dict')

    def test_concrete_subclass_roundtrip(self):
        class _MyState(Stateful):
            def __init__(self):
                self.step = 0

            def state_dict(self):
                return {'step': self.step}

            def load_state_dict(self, sd):
                self.step = sd['step']

        obj = _MyState()
        obj.step = 42
        sd = obj.state_dict()
        assert sd == {'step': 42}

        obj2 = _MyState()
        obj2.load_state_dict(sd)
        assert obj2.step == 42

    def test_stateful_is_abstract_base(self):
        """Stateful cannot be directly instantiated if methods are abstract.
        If it is just a mixin, plain instantiation should work (either is fine)."""
        # Either it is abstract (raises TypeError) or it can be instantiated.
        try:
            s = Stateful()
        except TypeError:
            pass  # Abstract - OK
        else:
            # Plain mixin - methods should exist
            assert hasattr(s, 'state_dict')
            assert hasattr(s, 'load_state_dict')


# ---------------------------------------------------------------------------
# 7. DCP save/load with Stateful objects
# ---------------------------------------------------------------------------

class TestDCPWithStateful:
    """DCP save/load must call state_dict() / load_state_dict() on Stateful
    values inside the state dict mapping."""

    def _make_stateful_model(self):
        m = _SimpleModel()

        class _StatefulModule(Stateful):
            def __init__(self, module):
                self._m = module

            def state_dict(self):
                return {k: v.detach() for k, v in self._m.state_dict().items()}

            def load_state_dict(self, sd):
                self._m.load_state_dict(sd)

        return m, _StatefulModule(m)

    def test_dcp_calls_state_dict_on_stateful(self, tmp_path):
        """DCP save extracts state from Stateful objects before writing."""
        m, stateful = self._make_stateful_model()
        ckpt = str(tmp_path / 'ckpt_stateful')
        # Pass a plain dict containing a Stateful value
        dcp.save({'model': stateful}, checkpoint_id=ckpt, no_dist=True)
        assert os.path.exists(os.path.join(ckpt, '.metadata'))

    def test_dcp_calls_load_state_dict_on_stateful(self, tmp_path):
        """DCP load calls load_state_dict on Stateful values."""
        m_src, stateful_src = self._make_stateful_model()
        ckpt = str(tmp_path / 'ckpt_stateful_load')
        dcp.save({'model': stateful_src}, checkpoint_id=ckpt, no_dist=True)

        m_dst, stateful_dst = self._make_stateful_model()
        # Zero out dst
        for p in m_dst.parameters():
            p.data.fill_(0)

        dcp.load({'model': stateful_dst}, checkpoint_id=ckpt, no_dist=True)

        # After load, dst should match src
        for key in m_src.state_dict():
            np.testing.assert_array_almost_equal(
                m_src.state_dict()[key].numpy(),
                m_dst.state_dict()[key].numpy(),
                decimal=6,
            )
