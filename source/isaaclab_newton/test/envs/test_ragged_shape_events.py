# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton material and collider randomization with no rectangular shape axis."""

from types import SimpleNamespace
from unittest.mock import Mock

import newton
import pytest
import torch
import warp as wp
from isaaclab_newton import assets
from isaaclab_newton.envs.mdp.events import randomize_rigid_body_collider_offsets, randomize_rigid_body_material
from newton.selection import ArticulationView
from newton.solvers import SolverKamino

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.test.utils import test_devices

pytestmark = pytest.mark.unit


@pytest.fixture(params=test_devices())
def scene(request):
    device = request.param
    builder = newton.ModelBuilder()
    builder.add_shape_sphere(-1, radius=0.1, label="global")
    for world, count in enumerate((0, 2, 1)):
        source = newton.ModelBuilder()
        crate = source.add_link(label="crate")
        source.add_articulation([source.add_joint_free(crate)], label="crate")
        root = source.add_link(label="arm/root")
        tip = source.add_link(label="arm/tip")
        fixed = source.add_joint_fixed(-1, root)
        hinge = source.add_joint_revolute(root, tip)
        source.add_articulation([fixed, hinge], label="arm")
        # Import order interleaves the two assets' physical shapes.
        for shape in range(count):
            source.add_shape_sphere(crate, radius=0.1, label=f"crate/shape{shape}")
            source.add_shape_sphere(tip, radius=0.1, label=f"arm/tip/shape{shape}")
        # The arm has two shapes in each world but their owning links differ.
        for shape in range(2 - count):
            source.add_shape_sphere(root, radius=0.1, label=f"arm/root/shape{shape}")
        builder.add_world(source, label_prefix=f"w{world}")
    model = builder.finalize(device=device)
    manager = SimpleNamespace(get_model=lambda: model, add_model_change=Mock(), _solver=None)
    crate = SimpleNamespace(_root_view=ArticulationView(model, "*/crate", homogeneous=False))
    arm = Mock(spec=assets.Articulation)
    arm._root_view = ArticulationView(model, "*/arm", homogeneous=False)
    # Public body 0 selects the tip in this asset's backend ordering.
    arm.map_body_ids_to_backend.return_value = torch.tensor([1], device=device)
    env = SimpleNamespace(
        scene={"crate": crate, "arm": arm}, device=device, num_envs=3, sim=SimpleNamespace(physics_manager=manager)
    )
    return env, model


def test_material_and_offsets_update_only_selected_rigid_shapes(scene):
    env, model = scene
    asset_cfg = SceneEntityCfg("crate")
    params = dict(
        asset_cfg=asset_cfg,
        static_friction_range=(0.4, 0.4),
        dynamic_friction_range=(0.1, 0.1),
        restitution_range=(0.8, 0.8),
        num_buckets=1,
    )
    material = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    offsets = randomize_rigid_body_collider_offsets(
        EventTermCfg(func=randomize_rigid_body_collider_offsets, params={"asset_cfg": asset_cfg}), env
    )
    arrays = [
        wp.to_torch(getattr(model, name))
        for name in ("shape_material_mu", "shape_material_restitution", "shape_margin", "shape_gap")
    ]
    expected = [array.clone() for array in arrays]
    rows = [model.shape_label.index(f"w1/crate/shape{i}") for i in (0, 1)]
    for i, value in enumerate((0.4, 0.8, 0.2, 0.1)):
        expected[i][rows] = value
    # World 0 has no crate shapes. The world 1 subset must still write both.
    material(env, torch.tensor([1, 0], dtype=torch.int32, device=env.device), **params)
    offsets(env, slice(0, 2), asset_cfg, (0.2, 0.2), (0.3, 0.3))
    for actual, wanted in zip(arrays, expected, strict=True):
        torch.testing.assert_close(actual, wanted)
    empty = torch.empty(0, dtype=torch.int64, device=env.device)
    material(env, empty, **params)
    offsets(env, empty, asset_cfg, (0.7, 0.7), (0.8, 0.8))
    for actual, wanted in zip(arrays, expected, strict=True):
        torch.testing.assert_close(actual, wanted)
    # Reuse margins from the last call. Preserve material and unrelated gaps.
    offsets(env, None, asset_cfg, contact_offset_distribution_params=(0.15, 0.15))
    all_crates = rows + [model.shape_label.index("w2/crate/shape0")]
    expected[3][all_crates] = (0.15 - expected[2][all_crates]).clamp_min(0)
    for actual, wanted in zip(arrays, expected, strict=True):
        torch.testing.assert_close(actual, wanted)
    assert env.sim.physics_manager.add_model_change.call_args.args == (newton.ModelFlags.SHAPE_PROPERTIES,)


def test_kamino_shares_persistent_original_groups_for_selected_links(scene):
    env, model = scene
    env.sim.physics_manager._solver = Mock(spec=SolverKamino)
    cfg = SceneEntityCfg("arm", body_ids=[0])
    params = dict(
        asset_cfg=cfg,
        static_friction_range=(0.5, 0.9),
        dynamic_friction_range=(0.1, 0.1),
        restitution_range=(0.2, 0.6),
        num_buckets=1,
    )
    material = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    arm_offsets = randomize_rigid_body_collider_offsets(
        EventTermCfg(func=randomize_rigid_body_collider_offsets, params={"asset_cfg": cfg}), env
    )
    # Collider offsets apply to every link of the selected asset.
    margin = wp.to_torch(model.shape_margin)
    expected_margin = margin.clone()
    expected_margin[[model.shape_label.index(name) for name in ("w2/arm/root/shape0", "w2/arm/tip/shape0")]] = 0.3
    arm_offsets(env, slice(2, 3), cfg, rest_offset_distribution_params=(0.3, 0.3))
    torch.testing.assert_close(margin, expected_margin)
    friction = wp.to_torch(model.shape_material_mu)
    restitution = wp.to_torch(model.shape_material_restitution)
    rows = [model.shape_label.index(name) for name in ("w1/arm/tip/shape0", "w1/arm/tip/shape1", "w2/arm/tip/shape0")]
    friction[rows] = torch.tensor([0.1, 0.3, 0.1], device=env.device)
    original_friction, original_restitution = friction.clone(), restitution.clone()
    for _ in range(2):
        material(env, torch.tensor([0], device=env.device), **params)
        for values, original, bounds in (
            (friction, original_friction, (0.5, 0.9)),
            (restitution, original_restitution, (0.2, 0.6)),
        ):
            samples = values[rows]
            assert ((samples >= bounds[0]) & (samples <= bounds[1])).all()
            assert samples[0] == samples[2]
            assert samples[0] != samples[1]
            expected = original.clone()
            expected[rows] = samples
            torch.testing.assert_close(values, expected)
            # The original material groups remain distinct after current values coincide.
            values[rows] = 0.7


@pytest.mark.parametrize("selected", [["w0/crate", "w2/crate"], ["w2/crate"]], ids=["ragged", "singleton"])
def test_sparse_shapes_use_manager_environment_ids(scene, selected):
    env, model = scene
    env.scene["crate"]._root_view = ArticulationView(model, selected, homogeneous=False)
    asset_cfg = SceneEntityCfg("crate")
    params = dict(
        asset_cfg=asset_cfg,
        static_friction_range=(0.4, 0.4),
        dynamic_friction_range=(0.1, 0.1),
        restitution_range=(0.8, 0.8),
        num_buckets=1,
    )
    material = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    offsets = randomize_rigid_body_collider_offsets(
        EventTermCfg(func=randomize_rigid_body_collider_offsets, params={"asset_cfg": asset_cfg}), env
    )
    values = [
        wp.to_torch(getattr(model, name))
        for name in ("shape_material_mu", "shape_material_restitution", "shape_margin", "shape_gap")
    ]
    expected = [array.clone() for array in values]
    for world in (0, 1, 2):
        material(env, torch.tensor([world], device=env.device), **params)
        offsets(env, slice(world, world + 1), asset_cfg, (0.1, 0.1), (0.3, 0.3))
        if world == 2:
            shape = model.shape_label.index("w2/crate/shape0")
            for array, value in zip(expected, (0.4, 0.8, 0.1, 0.2), strict=True):
                array[shape] = value
        for actual, wanted in zip(values, expected, strict=True):
            torch.testing.assert_close(actual, wanted)
