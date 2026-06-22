from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda, WidowX250S
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat


@register_env("ColorMatching-v1", max_episode_steps=500)
class ColorMatchingEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a specific colored cube (Red, Green, Blue) and drop it into its
    corresponding colored shallow box (bin) on the table.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch", "widowx250s"]
    agent: Union[PandaWristCam, Panda, Fetch, WidowX250S]

    cube_half_size = 0.02
    bin_half_size = 0.05
    bin_thickness = 0.005

    box_slots = [
        ((-0.3, -0.45), (-0.2, -0.15)),
        ((-0.3, -0.15), (-0.2, 0.15)),
        ((-0.3, 0.15), (-0.2, 0.45)),
    ]
    bin_slots = [
        ((-0.1, -0.45), (0.05, -0.15)),
        ((-0.1, -0.15), (0.05, 0.15)),
        ((-0.1, 0.15), (0.05, 0.45)),
    ]

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.colors = {
            "red": [1.0, 0.0, 0.0, 1.0],
            "green": [0.0, 1.0, 0.0, 1.0],
            "blue": [0.0, 0.0, 1.0, 1.0],
        }
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        front_pose = sapien_utils.look_at(eye=[0.5, 0, 0.8], target=[-0.3, 0, 0.05])
        side_pose = sapien_utils.look_at(eye=[-0.3, 0.8, 0.5], target=[-0.3, 0, 0.05])
        return [
            CameraConfig("front_cam", front_pose, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("side_cam", side_pose, 128, 128, np.pi / 2, 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.6, 0.8], target=[-0.3, 0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubes = {}
        self.bins = {}

        for color_name, rgba in self.colors.items():
            self.cubes[color_name] = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=rgba,
                name=f"{color_name}_cube",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            )

            bin_rgba = [rgba[0] * 0.7, rgba[1] * 0.7, rgba[2] * 0.7, rgba[3]]
            self.bins[color_name] = actors.build_box(
                self.scene,
                half_sizes=[self.bin_half_size, self.bin_half_size, 0.005],
                color=bin_rgba,
                name=f"{color_name}_bin",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[0, 0, 0.005]),
            )

    def _random_pose_in_slot(self, b, slot, z, max_yaw):
        (x_lo, y_lo), (x_hi, y_hi) = slot
        xyz = torch.zeros((b, 3))
        xyz[..., 0] = x_lo + torch.rand(b) * (x_hi - x_lo)
        xyz[..., 1] = y_lo + torch.rand(b) * (y_hi - y_lo)
        xyz[..., 2] = z
        yaw = (torch.rand(b) * 2 - 1) * max_yaw
        quat = torch.tensor(
            [euler2quat(0, 0, float(yaw[j])) for j in range(b)],
            dtype=torch.float32,
        )
        return xyz, quat

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.pick_color = options.get("pick_color", "red")
            self.place_color = options.get("place_color", "red")

            color_names = list(self.colors.keys())
            n = len(color_names)

            cube_order = torch.stack([torch.randperm(n) for _ in range(b)])
            bin_order = torch.stack([torch.randperm(n) for _ in range(b)])

            for ci in range(n):
                c_xyz_all = torch.zeros((b, 3))
                c_quat_all = torch.zeros((b, 4))
                b_xyz_all = torch.zeros((b, 3))
                b_quat_all = torch.zeros((b, 4))

                for slot_idx in range(n):
                    cube_mask = cube_order[:, ci] == slot_idx
                    if cube_mask.any():
                        bc = int(cube_mask.sum())
                        xyz, quat = self._random_pose_in_slot(
                            bc,
                            self.box_slots[slot_idx],
                            self.cube_half_size,
                            np.pi / 6,
                        )
                        c_xyz_all[cube_mask] = xyz
                        c_quat_all[cube_mask] = quat

                    bin_mask = bin_order[:, ci] == slot_idx
                    if bin_mask.any():
                        bb = int(bin_mask.sum())
                        xyz, quat = self._random_pose_in_slot(
                            bb,
                            self.bin_slots[slot_idx],
                            self.bin_thickness,
                            np.pi / 6,
                        )
                        b_xyz_all[bin_mask] = xyz
                        b_quat_all[bin_mask] = quat

                self.cubes[color_names[ci]].set_pose(
                    Pose.create_from_pq(p=c_xyz_all, q=c_quat_all)
                )
                self.bins[color_names[ci]].set_pose(
                    Pose.create_from_pq(p=b_xyz_all, q=b_quat_all)
                )

    def evaluate(self) -> Dict[str, torch.Tensor]:
        cube_pos = self.cubes[self.pick_color].pose.p
        bin_pos = self.bins[self.place_color].pose.p
        horizontal_dist = torch.linalg.norm(
            cube_pos[..., :2] - bin_pos[..., :2], axis=1
        )
        on_bin_z = cube_pos[..., 2] <= 1.01 * (bin_pos[..., 2] + self.cube_half_size + self.bin_thickness)
        success = (horizontal_dist < self.bin_half_size) & on_bin_z
        return {"success": success}

    def _get_obs_extra(self, info: dict) -> Dict[str, torch.Tensor]:
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            for color_name in self.colors.keys():
                obs[f"{color_name}_cube_pos"] = self.cubes[color_name].pose.p
                obs[f"{color_name}_bin_pos"] = self.bins[color_name].pose.p
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return torch.zeros(self.num_envs, device=self.device)


class VectorizedScriptedPolicy:
    LIFT_HEIGHT = 0.25
    GRASP_STEPS = 25

    def __init__(
        self, num_envs: int, device: torch.device, pick_color="red", place_color="red"
    ):
        self.num_envs = num_envs
        self.device = device
        self.pick_color = pick_color
        self.place_color = place_color

        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.state_counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self):
        self.state.fill_(0)
        self.state_counter.fill_(0)

    def act(self, env) -> torch.Tensor:
        tcp_pos = env.agent.tcp.pose.p[..., :3]
        cube_pos = env.cubes[self.pick_color].pose.p[..., :3]
        bin_pos = env.bins[self.place_color].pose.p[..., :3]

        ee_action = torch.zeros((self.num_envs, 3), device=self.device)
        gripper = torch.ones(self.num_envs, device=self.device)

        # --- Stage 0: Hover above cube ---
        target_0 = cube_pos.clone()
        target_0[..., 2] += 0.08
        dist_0 = torch.linalg.norm(target_0 - tcp_pos, axis=1)
        mask_0 = self.state == 0
        ee_action[mask_0] = (target_0[mask_0] - tcp_pos[mask_0]) * 3.0
        self.state[mask_0 & (dist_0 < 0.015)] = 1

        # --- Stage 1: Descend onto cube ---
        target_1 = cube_pos.clone()
        target_1[..., 2] += 0.015
        dist_1 = torch.linalg.norm(target_1 - tcp_pos, axis=1)
        mask_1 = self.state == 1
        ee_action[mask_1] = (target_1[mask_1] - tcp_pos[mask_1]) * 2.0
        self.state[mask_1 & (dist_1 < 0.01)] = 2

        # --- Stage 2: Close gripper ---
        mask_2 = self.state == 2
        gripper[mask_2] = -1.0
        self.state_counter[mask_2] += 1
        advance_2 = mask_2 & (self.state_counter >= self.GRASP_STEPS)
        self.state[advance_2] = 3
        self.state_counter[advance_2] = 0

        # --- Stage 3: Lift to fixed absolute height ---
        target_3 = tcp_pos.clone()
        target_3[..., 2] = self.LIFT_HEIGHT
        dist_3 = torch.abs(tcp_pos[..., 2] - self.LIFT_HEIGHT)
        mask_3 = self.state == 3
        gripper[mask_3] = -1.0
        ee_action[mask_3] = (target_3[mask_3] - tcp_pos[mask_3]) * 5.0
        self.state[mask_3 & (dist_3 < 0.02)] = 4

        # --- Stage 4: Move horizontally above bin ---
        target_4 = bin_pos.clone()
        target_4[..., 2] = self.LIFT_HEIGHT
        dist_4 = torch.linalg.norm(target_4 - tcp_pos, axis=1)
        mask_4 = self.state == 4
        gripper[mask_4] = -1.0
        ee_action[mask_4] = (target_4[mask_4] - tcp_pos[mask_4]) * 4.0
        self.state[mask_4 & (dist_4 < 0.02)] = 5

        # --- Stage 5: Lower onto bin ---
        target_5 = bin_pos.clone()
        target_5[..., 2] += 0.05
        dist_5 = torch.linalg.norm(target_5 - tcp_pos, axis=1)
        mask_5 = self.state == 5
        gripper[mask_5] = -1.0
        ee_action[mask_5] = (target_5[mask_5] - tcp_pos[mask_5]) * 4.0
        self.state[mask_5 & (dist_5 < 0.02)] = 6

        # --- Stage 6: Release ---
        mask_6 = self.state == 6
        gripper[mask_6] = 1.0

        final_actions = torch.zeros((self.num_envs, 7), device=self.device)
        final_actions[..., :3] = torch.clip(ee_action, -0.1, 0.1)
        final_actions[..., 6] = gripper

        return final_actions
