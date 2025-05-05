import mujoco
import numpy as np

class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData, target_points):
    self.m = m
    self.d = d
    self.target_points = target_points
    
    self.init_qpos = d.qpos.copy()
    
    # Control gains 
    self.kp_default = 100
    self.kp = self.kp_default
    
    self.kd_default = 50
    self.kd = self.kd_default
    
    # To track points 
    self.current_target = None # (3,)
    self.current_idx = None # Index of current_target
    self.points_active = np.array([True] * 8) # Track active points (from PointManger)
    
    self.cur_point_steps = 0
    self.thresh = 0.010 # Check if reached point
    
    self.ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")

    self.damping = 0.001 # For Levenberg-Marquardt

  def jacobian(self):
    jacp = np.zeros((3, self.m.nv))
    jacr = np.zeros((3, self.m.nv))
    mujoco.mj_jac(self.m, self.d, jacp, jacr, self.d.xpos[self.ee_id], self.ee_id)
    return jacp

  def get_closest_point(self):
    '''
    Returns closest point to the EE (3,) right when 
    this function is called and the index of that point
    '''
    distances = np.linalg.norm(self.d.xpos[self.ee_id].copy() - self.target_points[:,].T, axis=1)

    # considers magnitude of torque required to plan next point
    torque_mags = np.zeros((8,))
    for i in range(8):
      torque_cmd =  self.get_torque_cmd(self.d.xpos[self.ee_id], self.target_points[:, i])
      torque_mags[i] = np.linalg.norm(torque_cmd)
    torque_mags = torque_mags  * 0.1
    distances = torque_mags + distances
    
    closest_index = np.argmin(np.where(self.points_active, distances, np.inf))
    closest = self.target_points[:, closest_index]

    return closest, closest_index

  def update_kp(self, target):
    '''
    Updates kp given the steps already taken and distance to the point
    '''
    distance = np.linalg.norm(self.d.xpos[self.ee_id] - target)
    if self.cur_point_steps < 300 or distance > 0.05:
      if distance > 1:
        self.kp = 500
        self.kd = self.kd_default * 0.8
      else:
        if np.sum(self.points_active) == 8:
          self.kp = 10
        else:
          self.kp = self.kp_default
          self.kd = self.kd_default
    else:
      if distance < 0.03 and self.cur_point_steps > 500:
        self.kp = 500
        self.kd = self.kd_default * 1.5 
      else:
        self.kp = 100

  def get_torque_cmd(self, current_pos, target=None):
    if target is None:
      target = self.current_target
    error = target - current_pos

    self.update_kp(target)

    J = self.jacobian()
    
    product = J @ J.T + self.damping * np.eye(3)

    if np.isclose(np.linalg.det(product), 0):
      inv_J = J.T @ np.linalg.pinv(product)
    else: 
      inv_J = J.T @ np.linalg.inv(product)

    xdot = self.kp * error
    qdot = inv_J @ xdot
    
    jtorque_cmd = (qdot - self.d.qvel) * self.kd + self.d.qfrc_bias

    return jtorque_cmd

  def CtrlUpdate(self):
    current_pos = self.d.xpos[self.ee_id]
    
    if self.current_target is None: # Initializes first target point
      self.current_target, self.current_idx = self.get_closest_point()
      
    # lavenberg-Marquardt algorithm
    error = self.current_target - current_pos

    if np.linalg.norm(error) < self.thresh:
      self.points_active[self.current_idx] = False
      self.current_target, self.current_idx = self.get_closest_point()
      print(f'Steps taken: {self.cur_point_steps}')
      self.cur_point_steps = 0
      self.kp = self.kp_default
    else:
      self.cur_point_steps += 1
    
    return self.get_torque_cmd(current_pos)


