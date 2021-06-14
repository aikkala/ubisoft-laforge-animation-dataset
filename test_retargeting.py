import gym
from lafan1 import extract
from scipy.spatial.transform import Rotation as R
import numpy as np

# Create env
env = gym.make('crouch_c1_env:crouch_c1_env-v1', load_experience_data=False)

# Extract trajectory from one file
filepath = 'lafan1/lafan1/aiming2_subject5.bvh'
traj = extract.read_bvh(filepath)

# Dict for transforming joint names to mujoco indices
j2id = env.sim.model._joint_name2id

#orients = []
#for i in range(0, traj.pos.shape[0]):
#  euler = R.from_quat(traj.quats[i, traj.bones.index("Spine1")]).as_euler('xyz', degrees=False)
#  rot = R.from_quat(traj.quats[i, traj.bones.index("RightFoot")])
#  axis = R.from_euler('xyz', env.model.jnt_axis[j2id["ankle_angle_r"], :], degrees=False)
#  euler = np.dot(rot.as_euler('xyz', degrees=False), axis.as_matrix())
#  orients.append(euler)
#import matplotlib.pyplot as pp
#orients = np.asarray(orients)
#pp.plot(orients)


# Convert from centimeters into meters
traj.pos[:, 0, :] /= 100

# Try to get closer to ground
traj.pos[:, 0, 1] -= 0.19

# Loop through poses and extract joint angles
for i in range(0, traj.pos.shape[0]):

  # Pelvis position
  env.sim.data.qpos[j2id["pelvis_tz"]] = -traj.pos[i, 0, 0]
  env.sim.data.qpos[j2id["pelvis_ty"]] = traj.pos[i, 0, 1]
  env.sim.data.qpos[j2id["pelvis_tx"]] = traj.pos[i, 0, 2]

  # Pelvis rotation
  euler = R.from_quat(traj.quats[i, traj.bones.index("Hips")]).as_euler('xyz', degrees=False)
  env.sim.data.qpos[j2id["pelvis_tilt"]] = -euler[1]
  env.sim.data.qpos[j2id["pelvis_list"]] = (np.pi/2) - euler[0]
  env.sim.data.qpos[j2id["pelvis_rotation"]] = euler[2] - (np.pi/2)

  # Hip rotation, right
  euler = R.from_quat(traj.quats[i, traj.bones.index("RightUpLeg")]).as_euler('xyz', degrees=False)
  env.sim.data.qpos[j2id["hip_flexion_r"]] = euler[0]
  env.sim.data.qpos[j2id["hip_adduction_r"]] = euler[1]
  env.sim.data.qpos[j2id["hip_rotation_r"]] = np.pi - euler[2]

  # Knee rotation, right
  euler = R.from_quat(traj.quats[i, traj.bones.index("RightLeg")]).as_euler('xyz', degrees=False)
  env.sim.data.qpos[j2id["knee_angle_r"]] = euler[0]*-1 - np.pi

  # Ankle rotation, right
  rot = R.from_quat(traj.quats[i, traj.bones.index("RightFoot")])
  axis = R.from_euler('xyz', env.model.jnt_axis[j2id["ankle_angle_r"], :], degrees=False)
  euler = np.dot(rot.as_euler('xyz', degrees=False), axis.as_matrix())
  env.sim.data.qpos[j2id["ankle_angle_r"]] = euler[1] + (np.pi/2)

  # Hip rotation, left
  euler = R.from_quat(traj.quats[i, traj.bones.index("LeftUpLeg")]).as_euler('xyz', degrees=False)
  env.sim.data.qpos[j2id["hip_flexion_l"]] = euler[0]
  env.sim.data.qpos[j2id["hip_adduction_l"]] = -euler[1]
  env.sim.data.qpos[j2id["hip_rotation_l"]] = -(np.pi - euler[2])

  # Knee rotation, left
  euler = R.from_quat(traj.quats[i, traj.bones.index("LeftLeg")]).as_euler('xyz', degrees=False)
  env.sim.data.qpos[j2id["knee_angle_l"]] = euler[0]*-1 - np.pi

  # Ankle rotation, left
  rot = R.from_quat(traj.quats[i, traj.bones.index("LeftFoot")])
  axis = R.from_euler('xyz', env.model.jnt_axis[j2id["ankle_angle_l"], :], degrees=False)
  euler = np.dot(rot.as_euler('xyz', degrees=False), axis.as_matrix())
  env.sim.data.qpos[j2id["ankle_angle_l"]] = euler[1] + (np.pi/2)

  # Lumbar rotation
  euler = R.from_quat(traj.quats[i, traj.bones.index("Spine1")]).as_euler('xyz', degrees=False)
  env.sim.data.qpos[j2id["lumbar_extension"]] = euler[0] - np.pi
  env.sim.data.qpos[j2id["lumbar_bending"]] = euler[1]
  env.sim.data.qpos[j2id["lumbar_rotation"]] = euler[2]

  # Render
  env.sim.forward()
  env.render()