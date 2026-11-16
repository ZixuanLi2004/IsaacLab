# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-write validation of optional native solver joint-property capabilities."""

import numpy as np
import warp as wp


def _host(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _unsupported_friction_rows(solver):
    """None means every DOF; a set names unsupported parent-model owned rows.

    Solvers opt into rejection with ``supports_joint_friction = False``. Native
    coupled entries expose ownership separately from their proxy/view maps;
    only ``joint_qd_indices`` identifies the entry's actual parent-model DOFs.
    Solvers without this capability declaration preserve their existing API.
    """
    if getattr(solver, "supports_joint_friction", None) is False:
        return None
    rows = set()
    for entry in getattr(solver, "_entries", {}).values():
        if getattr(entry.solver, "supports_joint_friction", None) is False:
            rows.update(int(row) for row in _host(entry.joint_qd_indices).reshape(-1))
    return rows


def _view_dof_rows(articulation):
    view = articulation.root_view
    frequency = view.model.get_attribute_frequency("joint_friction")
    layout = view.frequency_layouts[frequency]
    rows = layout.get_absolute_indices(view.world_count, view.count_per_world)
    if (rows.shape != (view.world_count, view.count_per_world, articulation.num_joints)
            or view.world_count * view.count_per_world != articulation.num_instances):
        raise ValueError("Joint-friction capability validation requires one complete DOF row per instance.")
    public_to_backend = _host(articulation._joint_user_to_backend_map()).astype(int)
    return rows.reshape(articulation.num_instances, articulation.num_joints)[:, public_to_backend]


def validate_joint_friction_write(articulation, solver, value, *, joint_ids=None, env_ids=None,
                                 joint_mask=None, env_mask=None, masked=False):
    """Reject unsupported nonzero dry friction before staging or native writes.

    This is a host-side parameter-update boundary. For a solver declaring an
    unsupported domain, array-valued updates during CUDA capture are rejected
    explicitly because device values and selections cannot be read safely.
    Zero scalar writes remain valid and need no readback.
    """
    if isinstance(value, (int, float)) and value == 0:
        return
    entries = getattr(solver, "_entries", {}).values()
    if (getattr(solver, "supports_joint_friction", None) is not False
            and not any(getattr(entry.solver, "supports_joint_friction", None) is False for entry in entries)):
        return
    if wp.get_device(articulation.device).is_capturing:
        raise NotImplementedError("Joint-friction capability validation must run outside CUDA graph capture.")
    unsupported = _unsupported_friction_rows(solver)
    if masked:
        env_selection = (np.ones(articulation.num_instances, dtype=bool) if env_mask is None else _host(env_mask))
        joint_selection = (np.ones(articulation.num_joints, dtype=bool) if joint_mask is None else _host(joint_mask))
        if (env_selection.shape != (articulation.num_instances,)
                or joint_selection.shape != (articulation.num_joints,)
                or env_selection.dtype != np.bool_ or joint_selection.dtype != np.bool_):
            raise ValueError("Joint-friction masks must be boolean arrays with the complete instance/joint shape.")
        envs, joints = np.flatnonzero(env_selection), np.flatnonzero(joint_selection)
        values = _host(value)
        if values.ndim:
            if values.shape != (articulation.num_instances, articulation.num_joints):
                raise ValueError("Masked joint friction requires a complete instance-by-joint value array.")
            values = values[np.ix_(envs, joints)]
    else:
        envs = _host(articulation._resolve_env_ids(env_ids)).astype(int)
        joints = _host(articulation._resolve_joint_ids(joint_ids)).astype(int)
        values = _host(value)
        if values.ndim and values.shape != (len(envs), len(joints)):
            raise ValueError("Indexed joint friction requires one value per selected instance and joint.")
    values = np.broadcast_to(values, (len(envs), len(joints)))
    if not np.any(values != 0):
        return
    rows = _view_dof_rows(articulation)[np.ix_(envs, joints)]
    denied = np.ones(rows.shape, dtype=bool) if unsupported is None else np.isin(rows, list(unsupported))
    rejected = rows[denied & (values != 0)]
    if rejected.size:
        raise NotImplementedError(
            "The owning solver does not support nonzero dry joint friction for model DOFs "
            f"{np.unique(rejected).tolist()}; no joint-property buffers were changed."
        )
