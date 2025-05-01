import mujoco
import numpy as np

class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData, target_points):
    self.m = m
    self.d = d
    self.target_points = target_points
    
    self.init_qpos = d.qpos.copy()
    
    # Control gains 
    self.kp = 150.0
    self.kd = 10.0
    
    # To track points 
    self.current_target = None # (3,)
    self.current_idx = None # Index of current_target
    self.points_active = np.array([True] * 8) # Track active points (from PointManger)
    
    self.thresh = 0.01 # Check if reached point
    
    self.ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")

    self.damping = 0.03 # For Levenberg-Marquardt

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
    closest_index = np.argmin(np.where(self.points_active, distances, np.inf))
    closest = self.target_points[:, closest_index]
    return closest, closest_index

  def CtrlUpdate(self):
    current_pos = self.d.xpos[self.ee_id]
    
    if self.current_target is None: # Initializes first target point
      self.current_target, self.current_idx = self.get_closest_point()
    
    # lavenberg-Marquardt algorithm
    error = self.current_target - current_pos
    
    if np.linalg.norm(error) < self.thresh:
      self.points_active[self.current_idx] = False
      self.current_target, self.current_idx = self.get_closest_point()

    J = self.jacobian()
    
    product = J @ J.T + self.damping * np.eye(3)

    if np.isclose(np.linalg.det(product), 0):
      inv_J = J.T @ np.linalg.pinv(product)
    else: 
      inv_J = J.T @ np.linalg.inv(product)

    xdot = self.kp * error
    qdot = inv_J @ xdot
    
    jtorque_cmd = (qdot - self.d.qvel) * self.kp + self.d.qfrc_bias
    
    return jtorque_cmd



