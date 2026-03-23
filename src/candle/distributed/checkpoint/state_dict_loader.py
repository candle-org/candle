"""DCP load entry point.

Mirrors ``torch.distributed.checkpoint.state_dict_loader``.
"""

from .filesystem import FileSystemReader
from .planner import DefaultLoadPlanner
from .stateful import Stateful


def _extract_stateful(state_dict):
    """Separate Stateful objects from plain tensors in *state_dict*.

    Returns:
        flat_dict: mapping of ``{fqn: tensor}`` with Stateful values expanded
            into sub-keys using ``prefix/sub_key`` notation.
        stateful_map: mapping of ``{prefix: Stateful}`` for post-load restore.
    """
    flat_dict = {}
    stateful_map = {}
    for key, value in state_dict.items():
        if isinstance(value, Stateful):
            stateful_map[key] = value
            # Allocate placeholder tensors from the Stateful's current state
            inner = value.state_dict()
            if isinstance(inner, dict):
                for sub_key, sub_val in inner.items():
                    flat_dict[f"{key}/{sub_key}"] = sub_val
            else:
                flat_dict[key] = inner
        else:
            flat_dict[key] = value
    return flat_dict, stateful_map


def _restore_stateful(flat_dict, stateful_map):
    """Call load_state_dict on each Stateful using the loaded flat_dict values."""
    for prefix, stateful_obj in stateful_map.items():
        # Gather all sub-keys that belong to this stateful prefix
        sub_state = {}
        for fqn, val in flat_dict.items():
            if fqn.startswith(f"{prefix}/"):
                sub_key = fqn[len(prefix) + 1:]
                sub_state[sub_key] = val
            elif fqn == prefix:
                # Single-value stateful
                sub_state = val
                break
        stateful_obj.load_state_dict(sub_state)


def _dist_available():
    """Return True if distributed is initialized."""
    try:
        from .. import is_initialized  # pylint: disable=import-outside-toplevel
        return is_initialized()
    except Exception:  # pylint: disable=broad-except
        return False


def _get_dist_info(process_group):
    """Return (rank, world_size) from the process group or defaults."""
    try:
        from .. import get_rank, get_world_size  # pylint: disable=import-outside-toplevel
        return get_rank(process_group), get_world_size(process_group)
    except Exception:  # pylint: disable=broad-except
        return 0, 1


def _coordinate_load_plans(use_dist, local_plan, planner, rank, world_size, process_group):
    """Gather local plans, create global plan, return this rank's plan."""
    if use_dist:
        from .. import (  # pylint: disable=import-outside-toplevel
            all_gather_object, broadcast_object_list,
        )
        all_plans = [None] * world_size
        all_gather_object(all_plans, local_plan, group=process_group)

        global_plans = planner.create_global_plan(all_plans) if rank == 0 else None

        bcast = [global_plans]
        broadcast_object_list(bcast, src=0, group=process_group)
        return bcast[0][rank]

    return planner.create_global_plan([local_plan])[0]


def load(
    state_dict,
    *,
    checkpoint_id=None,
    storage_reader=None,
    planner=None,
    process_group=None,
    no_dist=False,
):
    """Load a DCP checkpoint into *state_dict* in-place.

    Args:
        state_dict: mutable mapping of ``{fqn: tensor}`` to populate.
        checkpoint_id: path or identifier for the checkpoint.
        storage_reader: a :class:`StorageReader` instance.
            Defaults to ``FileSystemReader(checkpoint_id)``.
        planner: a :class:`LoadPlanner` instance.
            Defaults to :class:`DefaultLoadPlanner`.
        process_group: distributed process group (or ``None`` for default).
        no_dist: if True, skip all distributed coordination.
    """
    use_dist = (not no_dist) and _dist_available()
    rank, world_size = _get_dist_info(process_group) if use_dist else (0, 1)
    is_coordinator = rank == 0

    if storage_reader is None:
        if checkpoint_id is None:
            raise ValueError("checkpoint_id is required when storage_reader is not provided")
        storage_reader = FileSystemReader(checkpoint_id)

    if planner is None:
        planner = DefaultLoadPlanner()

    # Expand Stateful values; keep a map so we can call load_state_dict after.
    # For plain tensors, flat_dict holds the same tensor objects as state_dict.
    flat_dict, stateful_map = _extract_stateful(state_dict)

    # Reset & read metadata
    storage_reader.reset(checkpoint_id)
    metadata = storage_reader.read_metadata() if is_coordinator else None

    if use_dist:
        from .. import broadcast_object_list  # pylint: disable=import-outside-toplevel
        bcast = [metadata]
        broadcast_object_list(bcast, src=0, group=process_group)
        metadata = bcast[0]

    # Set up reader and planner, create plans
    storage_reader.set_up_storage_reader(metadata, is_coordinator)
    planner.set_up_planner(flat_dict, metadata, is_coordinator)
    local_plan = storage_reader.prepare_local_plan(planner.create_local_plan())

    my_plan = _coordinate_load_plans(
        use_dist, local_plan, planner, rank, world_size, process_group,
    )

    # Finalize plan and read data
    final_plan = storage_reader.prepare_global_plan(planner.finish_plan(my_plan))
    storage_reader.read_data(final_plan, planner).wait()

    # Copy loaded flat_dict values back into the original state_dict.
    # commit_tensor replaces flat_dict entries with newly-read tensor objects;
    # for non-Stateful keys the caller expects in-place update of state_dict.
    for fqn, val in flat_dict.items():
        # Only copy back top-level keys that exist in the original state_dict
        if fqn in state_dict and not isinstance(state_dict[fqn], Stateful):
            orig = state_dict[fqn]
            # Write in-place when shapes match, otherwise replace the reference
            if hasattr(orig, 'data') and hasattr(val, 'data') and orig.shape == val.shape:
                orig.data = val.data
            else:
                state_dict[fqn] = val

    # Restore Stateful objects from the now-populated flat_dict
    if stateful_map:
        _restore_stateful(flat_dict, stateful_map)

    if use_dist:
        from .. import barrier  # pylint: disable=import-outside-toplevel
        barrier(group=process_group)
