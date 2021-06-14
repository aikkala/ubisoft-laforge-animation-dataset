import gym
from lafan1 import extract
from scipy.spatial.transform import Rotation as R
import numpy as np
import os


def set_pose(env, traj, t):

  qpos = env.sim.data.qpos.copy()

  # Pelvis position
  qpos[j2id["pelvis_tz"]] = -traj.pos[t, 0, 0]
  qpos[j2id["pelvis_ty"]] = traj.pos[t, 0, 1]
  qpos[j2id["pelvis_tx"]] = traj.pos[t, 0, 2]

  # Pelvis rotation
  euler = R.from_quat(traj.quats[t, traj.bones.index("Hips")]).as_euler('xyz', degrees=False)
  qpos[j2id["pelvis_tilt"]] = -euler[1]
  qpos[j2id["pelvis_list"]] = (np.pi / 2) - euler[0]
  qpos[j2id["pelvis_rotation"]] = euler[2] - (np.pi / 2)

  # Hip rotation, right
  euler = R.from_quat(traj.quats[t, traj.bones.index("RightUpLeg")]).as_euler('xyz', degrees=False)
  qpos[j2id["hip_flexion_r"]] = euler[0]
  qpos[j2id["hip_adduction_r"]] = euler[1]
  qpos[j2id["hip_rotation_r"]] = np.pi - euler[2]

  # Knee rotation, right
  euler = R.from_quat(traj.quats[t, traj.bones.index("RightLeg")]).as_euler('xyz', degrees=False)
  qpos[j2id["knee_angle_r"]] = euler[0] * -1 - np.pi

  # Ankle rotation, right
  rot = R.from_quat(traj.quats[t, traj.bones.index("RightFoot")])
  axis = R.from_euler('xyz', env.model.jnt_axis[j2id["ankle_angle_r"], :], degrees=False)
  euler = np.dot(rot.as_euler('xyz', degrees=False), axis.as_matrix())
  qpos[j2id["ankle_angle_r"]] = euler[1] + (np.pi / 2)

  # Hip rotation, left
  euler = R.from_quat(traj.quats[t, traj.bones.index("LeftUpLeg")]).as_euler('xyz', degrees=False)
  qpos[j2id["hip_flexion_l"]] = euler[0]
  qpos[j2id["hip_adduction_l"]] = -euler[1]
  qpos[j2id["hip_rotation_l"]] = -(np.pi - euler[2])

  # Knee rotation, left
  euler = R.from_quat(traj.quats[t, traj.bones.index("LeftLeg")]).as_euler('xyz', degrees=False)
  qpos[j2id["knee_angle_l"]] = euler[0] * -1 - np.pi

  # Ankle rotation, left
  rot = R.from_quat(traj.quats[t, traj.bones.index("LeftFoot")])
  axis = R.from_euler('xyz', env.model.jnt_axis[j2id["ankle_angle_l"], :], degrees=False)
  euler = np.dot(rot.as_euler('xyz', degrees=False), axis.as_matrix())
  qpos[j2id["ankle_angle_l"]] = euler[1] + (np.pi / 2)

  # Lumbar rotation
  euler = R.from_quat(traj.quats[t, traj.bones.index("Spine1")]).as_euler('xyz', degrees=False)
  qpos[j2id["lumbar_extension"]] = euler[0] - np.pi
  qpos[j2id["lumbar_bending"]] = euler[1]
  qpos[j2id["lumbar_rotation"]] = euler[2]

  # Initialise simulation
  env.initialise_simulation({"qpos": qpos})

def find_ground_level(env, mode="none"):

  # Get height before decreasing / increasing height
  pelvis_height = env.sim.data.qpos[1]
  qpos = env.sim.data.qpos.copy()

  # Then decrease / increase until ground level is found
  def in_contact(env):
    for c in env.sim.data.contact:
      if (c.geom1 == 0 or c.geom2 == 0) and not (c.geom1 == 0 and c.geom2 == 0):
        return True
    return False

  count = 0
  if mode in ["none", "increase"] and in_contact(env):
    # Increase until no more contact
    while in_contact(env):
      qpos[1] += 0.001
      #env.sim.forward()
      env.initialise_simulation({"qpos": qpos})
      count += 1
      if count == 1000:
        return False, 0
  elif mode in ["none", "decrease"] and not in_contact(env):
    # Decrease until contact
    while not in_contact(env):
      qpos[1] -= 0.001
      #env.sim.forward()
      env.initialise_simulation({"qpos": qpos})
      count += 1
      if count == 1000:
        return False, 0

    # Increase a little so there's no contact anymore
    qpos[1] += 0.001
    #env.sim.forward()
    env.initialise_simulation({"qpos": qpos})

  elif mode not in ["none", "increase", "decrease"]:
    raise NotImplementedError

  # Return decrease/increase
  return True, qpos[1] - pelvis_height

def extract_trajectory(filepath, env):

  # Extract trajectory from one file
  traj = extract.read_bvh(filepath)

  # Convert from centimeters into meters
  traj.pos[:, 0, :] /= 100

  # Set initial pose and find ground level; all trajectories should start in the same (upward standing) pose
  set_pose(env, traj, 0)
  success, diff = find_ground_level(env)

  # Return empty array if we couldn't find ground level
  if not success:
    return np.array([])

  # Try to get closer to ground
  traj.pos[:, 0, 1] += diff

  # Loop through poses and extract joint angles
  L = traj.pos.shape[0]
  sample_freq = 1/traj.frametime
  action_freq = 1/(env.frame_skip*env.model.opt.timestep)
  assert sample_freq > action_freq, "Trajectory must be sampled at a higher frequency, otherwise timesteps calculation below is incorrect"
  timesteps = np.arange(0, L, sample_freq/action_freq).astype(int)
  trajectory_qpos = np.zeros((len(timesteps), env.model.nq))
  trajectory_qvel = np.zeros_like(trajectory_qpos)
  for i, t in enumerate(timesteps):
    set_pose(env, traj, t)

    # Make sure model is not below or inside floor
    find_ground_level(env, mode="increase")

    # Wrap angles; modifies in place
    env.wrap_angles({"qpos": env.sim.data.qpos})

    # Estimate qvel
    if i > 0:
      trajectory_qvel[i-1] = (env.sim.data.qpos - previous_qpos)*(1/env.dt)
    previous_qpos = env.sim.data.qpos.copy()

    #env.render()
    trajectory_qpos[i] = env.sim.data.qpos.copy()

  return trajectory_qpos, trajectory_qvel

def get_trajectories(env, data_folder, prefixes):
  data = []

  all_files = os.listdir(data_folder)

  for prefix in prefixes:
    # List of files for this prefix
    files = [os.path.join(data_folder, file) for file in all_files if prefix in file]

    # Extract trajectory for all files
    for file in files:
      trajectory_qpos, trajectory_qvel = extract_trajectory(file, env)
      if trajectory_qpos.size == 0:
        continue
      else:

        # Let's do some preprocessing here
        trajectory = [{"qpos": qpos, "qvel": qvel} for qpos, qvel in zip(trajectory_qpos, trajectory_qvel)]
        trajectory_targets = env.calculate_targets(trajectory)
        env.set_trajectory(trajectory, trajectory_targets)
        data.append((env.trajectory, env.trajectory_targets))

  return data

if __name__ == "__main__":

  # Create env
  env = gym.make('crouch_c1_env:crouch_c1_env-v1', load_experience_data=False)

  # Data folder
  data_folder = '/home/aleksi/Workspace/ubisoft-laforge-animation-dataset/lafan1/lafan1'

  # Dict for transforming joint names to mujoco indices
  j2id = env.sim.model._joint_name2id

  # prefixes of tasks that will be processed
  train_prefixes = [
    "aiming", "dance", "fallAndGetUp", "fight", "fightAndSports", "jumps", "push",
    "pushAndFall", "pushAndStumble", "run", "sprint", "walk"
  ]
  test_prefixes = ["multipleActions"]

  # Get train data
  train_data = get_trajectories(env, data_folder, train_prefixes)

  # Then test data
  test_data = get_trajectories(env, data_folder, test_prefixes)

  # Save trajectories
  output_folder = '/home/aleksi/Workspace/gym_environments/opensim_converted/crouch_c1_env/envs'
  np.save(os.path.join(output_folder, 'laforge_train_trajectories'), np.asarray(train_data, dtype=object))
  np.save(os.path.join(output_folder, 'laforge_test_trajectories'), np.asarray(test_data, dtype=object))