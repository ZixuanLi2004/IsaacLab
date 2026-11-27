# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless regression tests for a retained Newton rigid object data container."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab_newton.physics import NewtonManager
from newton.selection import ArticulationView

from isaaclab.physics import PhysicsManager
from isaaclab.test.utils import test_devices


def _build_model(device, *, fixed=False):
    world = newton.ModelBuilder()
    for label in ("target", "neighbor"):
        body = world.add_link(label=label)
        world.add_shape_sphere(body, radius=0.1)
        joint = world.add_joint_fixed(-1, body) if fixed else world.add_joint_free(body)
        world.add_articulation([joint], label=label)
    builder = newton.ModelBuilder()
    for _ in range(2):
        builder.add_world(world)
    return builder.finalize(device=device)


def _set_state(model, state, x_positions, x_speeds):
    coords = state.joint_q.numpy().reshape(-1, 7)
    coords[:, 0] = x_positions
    speeds = state.joint_qd.numpy().reshape(-1, 6)
    speeds[:, 0] = x_speeds
    state.joint_q.assign(coords.ravel())
    state.joint_qd.assign(speeds.ravel())
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _rigid_writer(view, device):
    # Exercise the SDK writer and real Newton state without starting a dynamics solver.
    asset = object.__new__(RigidObject)
    asset._root_view = view
    asset._device = device
    asset._data = RigidObjectData(view, device)
    return asset


@pytest.mark.parametrize("device", test_devices())
def test_ready_rebind_refreshes_warm_caches_and_velocity_history(monkeypatch, device):
    model = _build_model(device)
    backend = SimpleNamespace(model=model, state_0=model.state())
    monkeypatch.setattr(NewtonManager, "backend", backend)
    view = ArticulationView(model, "target")
    data = RigidObjectData(view, device)
    data.update(0.0)
    _set_state(model, backend.state_0, [1, 0, 2, 0], [4, 0, 8, 0])
    data.update(0.5)
    np.testing.assert_allclose(data.root_com_pos_w.warp.numpy()[:, 0], [1, 2])
    np.testing.assert_allclose(data.root_link_lin_vel_w.warp.numpy()[:, 0], [4, 8])
    with pytest.warns(DeprecationWarning):
        np.testing.assert_allclose(data.root_state_w.warp.numpy()[:, 0], [1, 2])
    np.testing.assert_allclose(data.body_com_acc_w.warp.numpy()[:, 0, 0], [8, 16])

    replacement = model.state()
    _set_state(model, replacement, [10, 99, 20, 99], [40, 99, 80, 99])
    backend.state_0 = replacement
    data._create_simulation_bindings()
    np.testing.assert_allclose(data.root_com_pos_w.warp.numpy()[:, 0], [10, 20])
    np.testing.assert_allclose(data.root_link_lin_vel_w.warp.numpy()[:, 0], [40, 80])
    with pytest.warns(DeprecationWarning):
        np.testing.assert_allclose(data.root_state_w.warp.numpy()[:, [0, 7]], [[10, 40], [20, 80]])
    np.testing.assert_array_equal(data.body_com_acc_w.warp.numpy(), 0)

    _set_state(model, replacement, [10, 99, 20, 99], [41, 99, 82, 99])
    data.update(0.25)
    np.testing.assert_allclose(data.body_com_acc_w.warp.numpy()[:, 0, 0], [4, 8])
    data._create_simulation_bindings()  # Another READY with the same state preserves the last measurement.
    np.testing.assert_allclose(data.body_com_acc_w.warp.numpy()[:, 0, 0], [4, 8])

    # Keep the old data reference and replace the active model with one of identical layout.
    backend.model = _build_model(device)
    backend.state_0 = backend.model.state()
    for key in ("root_com_pos_w", "root_link_pose_w", "body_com_vel_w", "body_com_acc_w", "body_mass", "GRAVITY_VEC_W"):
        with pytest.raises(RuntimeError, match="different Newton model"):
            getattr(data, key)
    with pytest.raises(RuntimeError, match="different Newton model"):
        data._create_simulation_bindings()


@pytest.mark.parametrize("device", test_devices())
def test_singleton_view_uses_native_gravity_and_reset_rows(monkeypatch, device):
    prototype = newton.ModelBuilder()
    body = prototype.add_link(label="body")
    prototype.add_shape_sphere(body, radius=0.1)
    prototype.add_articulation([prototype.add_joint_free(body)], label="owner")
    builder = newton.ModelBuilder()
    builder.add_builder(prototype, label_prefix="/Global")
    for world in range(2):
        builder.add_world(prototype, label_prefix=f"/W{world}")
    model = builder.finalize(device=device)
    model.gravity.assign([[1, 0, 0], [0, -2, 0], [0, 0, -3]])
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(model=model, state_0=model.state()))
    monkeypatch.setattr(PhysicsManager, "_device", device)
    monkeypatch.setattr(NewtonManager, "kinematics_dirty", False)
    world_mask = wp.zeros(3, dtype=wp.bool, device=device)
    fk_mask = wp.zeros(model.articulation_count, dtype=wp.bool, device=device)
    monkeypatch.setattr(NewtonManager, "_world_reset_mask", world_mask)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", fk_mask)
    for label, world in (("/Global/owner", -1), ("/W1/owner", 1)):
        view = ArticulationView(model, label, homogeneous=False)
        assert RigidObject.supports_root_view(view)
        data = RigidObjectData(view, device)
        np.testing.assert_array_equal(data.GRAVITY_VEC_W.warp.numpy()[0], model.gravity.numpy()[world])
        world_mask.zero_()
        fk_mask.zero_()
        data._reset_pose()
        np.testing.assert_array_equal(np.flatnonzero(world_mask.numpy()), [2 if world == -1 else world])
        np.testing.assert_array_equal(np.flatnonzero(fk_mask.numpy()), [model.articulation_label.index(label)])


@pytest.mark.parametrize("device", test_devices())
def test_ready_rebind_keeps_fixed_body_velocity_shapes(monkeypatch, device):
    model = _build_model(device, fixed=True)
    backend = SimpleNamespace(model=model, state_0=model.state())
    monkeypatch.setattr(NewtonManager, "backend", backend)
    view = ArticulationView(model, "target")
    asset = _rigid_writer(view, device)
    data = asset.data
    for _ in range(2):
        asset.write_root_com_velocity_to_sim_index(
            root_velocity=wp.array([[7, 0, 0, 0, 0, 0]], dtype=wp.spatial_vectorf, device=device), env_ids=[1]
        )
        data.update(0.25)
        assert data.root_com_vel_w.warp.shape == (2,)
        assert data.body_com_vel_w.warp.shape == (2, 1)
        np.testing.assert_array_equal(data.body_com_acc_w.warp.numpy(), 0)
        backend.state_0 = model.state()
        data._create_simulation_bindings()


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("frame", ["com", "link"])
@pytest.mark.parametrize("selector", ["index", "mask"])
def test_partial_velocity_write_rebases_only_selected_acceleration(monkeypatch, device, frame, selector):
    model = _build_model(device)
    model.body_com.assign(np.tile([0.5, 0.0, 0.0], (model.body_count, 1)))
    state = model.state()
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(model=model, state_0=state))
    view = ArticulationView(model, "target")
    asset = _rigid_writer(view, device)
    asset.update(0.0)
    _set_state(model, state, [0] * 4, [4, 0, 8, 0])
    asset.update(0.5)
    write = getattr(asset, f"write_root_{frame}_velocity_to_sim_{selector}")
    velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    if selector == "index":
        write(root_velocity=wp.array([velocity], dtype=wp.spatial_vectorf, device=device), env_ids=[1])
    else:
        write(
            root_velocity=wp.array([velocity, velocity], dtype=wp.spatial_vectorf, device=device),
            env_mask=wp.array([False, True], dtype=wp.bool, device=device),
            skip_forward=True,
        )
    # The selected row starts a new history interval. The other row retains its last measurement.
    np.testing.assert_allclose(asset.data.body_com_acc_w.warp.numpy()[:, 0, 0], [8, 0])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    expected_com_y = 1.0 if frame == "link" else 0.0
    np.testing.assert_allclose(state.body_qd.numpy()[2], [0, expected_com_y, 0, 0, 0, 2])

    # Supply distinct physical motion during the following interval in both worlds.
    _set_state(model, state, [0] * 4, [6, 0, 0.5, 0])
    asset.update(0.25)
    np.testing.assert_allclose(asset.data.body_com_acc_w.warp.numpy()[:, 0], [[8, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]])
