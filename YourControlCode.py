import mujoco
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.interpolate import CubicSpline

class YourCtrl:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, targets: np.ndarray,
                 v_const=1.7, kp=190.0, kd=170.0, damping=0.01, min_dt=0.02):
        self.m, self.d = m, d

        # create self variables
        self.v_const, self.kp, self.kd = v_const, kp, kd
        self.damping, self.min_dt = damping, min_dt

        #mujoco ee setup
        self.ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")
        mujoco.mj_forward(m, d)
        cur = d.xpos[self.ee_id]

        # TSP ordering and rotate so first waypoint is closest
        order = self._solve_tsp(targets)
        wps   = [targets[:,i] for i in order]
        k     = int(np.argmin([np.linalg.norm(cur-p) for p in wps]))
        seq   = [cur] + wps[k:]+wps[:k]

        #create and alloc time for splines
        P     = np.stack(seq, axis=1)               
        dists = np.linalg.norm(np.diff(P, axis=1), axis=0)
        times = np.maximum(dists/self.v_const, self.min_dt)
        times[0] *= 2.0                                    #double time for first point for stability
        self.times   = np.concatenate(([0.], times.cumsum()))
        self.splines = [CubicSpline(self.times, P[i], bc_type='clamped')
                        for i in range(3)]
        self.start_time = float(d.time)

    #greedy tsp solver
    def _solve_tsp(self, pts):
        
        pts_list = pts.T.tolist()
        N = len(pts_list)

        tour = [0]
        remaining = set(range(1, N))
        while remaining:
            last = tour[-1]
            # pick closest unvisited
            nxt = min(remaining,
                      key=lambda j: np.linalg.norm(np.array(pts_list[last]) - np.array(pts_list[j])))
            tour.append(nxt)
            remaining.remove(nxt)

        return tour

    def CtrlUpdate(self):
        t = np.clip(self.d.time - self.start_time, 0, self.times[-1])

        #uses splines for pos/velocity
        x_des = np.array([s(t) for s in self.splines])
        v_des = np.array([s.derivative()(t) for s in self.splines])

        e     = x_des - self.d.xpos[self.ee_id]
        xdot  = v_des + self.kp * e

        Jp    = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, Jp, np.zeros_like(Jp),
                      self.d.xpos[self.ee_id], self.ee_id)
        
        J_pinv = Jp.T @ np.linalg.pinv(Jp @ Jp.T + self.damping*np.eye(3))

        qdot_des = J_pinv @ xdot
        return self.kd*(qdot_des - self.d.qvel) + self.d.qfrc_bias
