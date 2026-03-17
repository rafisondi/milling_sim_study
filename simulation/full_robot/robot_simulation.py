import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from eraser_of_matter import milling_workpiece
from milling_data import MillingPath, load_workpiece_vertices_from_pickle


class Settings:
    pass

class SettingsRealParameter(Settings): # 1 - 3 COMPLIANT (REAL PARAMETERS)
    def __init__(self):
        self.URDF_PATH = URDF_PATH_V3
        self.JOINT_STATE = np.array([False, False, False, True, True, True])
        self.K_m_diag = np.array([1.15e6, 3.76e6, 1.29e6, np.NAN, np.NAN, np.NAN]) 
        self.D_m_diag = np.array([2.82e3, 0.61e3, 1.87e3, np.NAN, np.NAN, np.NAN])

class Robot:
    def __init__(self, settings: Settings): 

        self.model = pin.buildModelFromUrdf(str(Path(settings.URDF_PATH)))
        self.data = self.model.createData()
        self.n = self.model.njoints - 1

        print("Full gravity motion object:\n", self.model.gravity)
        JOINT_STATE = settings.JOINT_STATE #np.array([False, False, False, True, True, True])
        self.idx_flex = np.where(~JOINT_STATE)[0]
        self.idx_rigid = np.where(JOINT_STATE)[0]
        self.K_m_diag = settings.K_m_diag   #np.array([1, 1, 1, 0, 0, 0])
        self.D_m_diag = settings.D_m_diag   #np.array([1, 1, 1, 0, 0, 0])
        self.K_m_diag_flex = settings.K_m_diag[self.idx_flex].reshape(-1, 1)   #np.array([1, 1, 1, 0, 0, 0])
        self.D_m_diag_flex = settings.D_m_diag[self.idx_flex].reshape(-1, 1)   #np.array([1, 1, 1, 0, 0, 0])

    def frame_placement(self, q, frame_name):
        q = np.asarray(q, dtype=float).reshape(-1)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        frame_id = self.model.getFrameId(frame_name)
        return self.data.oMf[frame_id].copy()

    def fkine(self, q, ee_name):
        T_ee = self.frame_placement(q, ee_name)

        T_matrix = np.vstack([
            np.hstack([T_ee.rotation, T_ee.translation.reshape(3,1)]),
            np.array([0, 0, 0, 1])
        ])
        return T_matrix

    def frame_pose_error(self, target_pose, current_pose):
        pose_error = target_pose.actInv(current_pose)
        return pin.log6(pose_error).vector

    def damped_ik_step(self, J, twist_error, damp):
        normal_matrix = J @ J.T + damp * np.eye(J.shape[0])
        return -J.T @ np.linalg.solve(normal_matrix, twist_error)

    def frame_pose(self, q, frame_name):
        return self.frame_placement(q, frame_name)

    def frame_jacobian(self, q, frame_name, reference_frame=pin.ReferenceFrame.LOCAL):
        frame_id = self.model.getFrameId(frame_name)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return pin.computeFrameJacobian(self.model, self.data, q, frame_id, reference_frame)

    def clik(
        self,
        target_pose,
        ee_name,
        q0,
        eps=1e-6,
        max_iters=500,
        damp=1e-8,
        step_size=1.0,
    ):
        q = np.asarray(q0, dtype=float).copy().reshape(-1)

        for _ in range(max_iters):
            current_pose = self.frame_pose(q, ee_name)
            twist_error = self.frame_pose_error(target_pose, current_pose)

            if np.linalg.norm(twist_error) < eps:
                return q

            J = self.frame_jacobian(q, ee_name, reference_frame=pin.ReferenceFrame.LOCAL)
            dq = self.damped_ik_step(J, twist_error, damp)
            q = pin.integrate(self.model, q, step_size * dq)

        raise RuntimeError(f"IK did not converge for frame '{ee_name}' after {max_iters} iterations.")
    
    def jacobian(self, q, ee_name):
        ee_id = self.model.getFrameId(ee_name)
        pin.computeJointJacobians(self.model, self.data, q)
        return pin.computeFrameJacobian(self.model, self.data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def jacobian_dot(self, q, ee_name):
        ee_id = self.model.getFrameId(ee_name)
        pin.computeJointJacobians(self.model, self.data, q)
        return pin.getFrameJacobianTimeVariation(self.model, self.data, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def inertia_matrix(self, q):
        return pin.crba(self.model, self.data, q)  # Composite Rigid Body Algorithm

    def bias(self, q, v):
        bias = pin.rnea(self.model, self.data, q, v, np.zeros(self.model.nv))  # no acceleration
        return bias.reshape(-1, 1)
    
    def M(self, q):
    # M(q)
        return pin.crba(self.model, self.data, q)

    def g(self, q):
    # g(q)
        return pin.computeGeneralizedGravity(self.model, self.data, q).reshape(-1, 1)

    def C(self, q, v):
    # C(q,v)
        return pin.computeCoriolisMatrix(self.model, self.data, q, v)

        

    def forward_dynamics(self, q_full, qD_full, tau_ext, tau_u):

        bias = self.bias(q_full, qD_full)[self.idx_flex, :]                     # Coriolis + gravity vector
        inertia_matrix = self.inertia_matrix(q_full)[np.ix_(self.idx_flex, self.idx_flex)]
            
        rhs_q =  -bias + tau_ext + tau_u    #-bias[self.idx_flex,] + tau_ext + tau_u          
        qDD_flex = np.linalg.solve(inertia_matrix, rhs_q)
    
        return qDD_flex
    
    
class Environment():
    def get_f_ext(self):
        return self.f_ext
    
class EnvironmentMilling(Environment):
    def __init__(
        self,
        robot: Robot,
        trajectory=None,
        workpiece_pickle: str | Path | None = None,
        axial_cutting_depth_mm: float = 5.0,
        ee_name: str = "TCP",
        samples_per_period: int = 180,
        spindle_spin: int = -1,
        number_of_teeth: int = 8,
    ):
        self.t = 0
        self.robot = robot
        self.f_ext = np.zeros((6, 1))
        self.ee_name = ee_name
        self.trajectory = trajectory
        self.axial_cutting_depth_mm = float(axial_cutting_depth_mm)
        self.spindle_spin = int(spindle_spin)
        self.number_of_teeth = int(number_of_teeth)
        self.samples_per_period = int(samples_per_period)
        self.omega = 0.0
        self.milling_process = None
        self.T_workpiece_base = None

        if trajectory is not None and getattr(trajectory, "dt", 0.0) > 0.0:
            rotation_period_s = trajectory.dt * max(self.samples_per_period, 1)
            self.omega = self.spindle_spin * 2.0 * np.pi / max(rotation_period_s, 1e-12)

        if trajectory is not None:
            self.T_workpiece_base = np.linalg.inv(trajectory.T_base_workpiece)

        if workpiece_pickle is not None:
            workpiece_vertices = load_workpiece_vertices_from_pickle(workpiece_pickle)
            self.milling_process = milling_workpiece(
                workpiece_vertices,
                axial_cutting_depth=self.axial_cutting_depth_mm,
            )
            self.milling_process.number_of_teeth = self.number_of_teeth
            self.milling_process.axial_cutting_depth = self.axial_cutting_depth_mm
        
    def compute_tau_ext(self, q_full):
        if self.milling_process is None or self.T_workpiece_base is None:
            self.f_ext = np.zeros((6, 1))
            return np.zeros((len(self.robot.idx_flex), 1))

        tcp_pose = self.robot.frame_placement(np.asarray(q_full, dtype=float).reshape(-1), self.ee_name)
        tcp_position_base = tcp_pose.translation
        tcp_position_base_aug = np.append(tcp_position_base, 1.0)
        tcp_position_workpiece = self.T_workpiece_base @ tcp_position_base_aug
        tool_center_mm = tcp_position_workpiece[:2] * 1000.0
        spindle_angle = self.t * self.omega

        self.milling_process.erase_step(
            tool_center_mm,
            spindle_angle,
            direction=self.spindle_spin,
            t=self.t,
        )

        if self.t <= 2.0 * max(getattr(self.trajectory, "dt", 0.0), 1e-12):
            milling_force_wp = np.zeros((3, 1))
        else:
            milling_force_wp = np.asarray(
                self.milling_process.total_milling_force[:3],
                dtype=float,
            ).reshape(3, 1)

        rotation_base_workpiece = self.trajectory.T_base_workpiece[:3, :3]
        milling_force_base = rotation_base_workpiece @ milling_force_wp
        self.f_ext = np.vstack([milling_force_base, np.zeros((3, 1))])

        J = self.robot.jacobian(np.asarray(q_full, dtype=float).reshape(-1), self.ee_name)
        tau_full = J.T @ self.f_ext
        return tau_full[self.robot.idx_flex, :]
    
    def set_t(self, t):
        self.t = t
    


class Controller():
    def get_ctrl_freq(self):
        return self.ctrl_freq


class ComplianceCompensation(Controller):
    def __init__(
        self,
        ctrl_freq: float,
        robot_model: Robot,
        f_cutoff: float = None,
        compliance_error_ratio: float = 0.0,
        input_delay_samples: int = 0,
        input_noise: bool = False,
        fts_t_std: float = 0.01,
        fts_f_std: float = 0.01,
        reconstr_filter: bool = False,
    ):
        self.ctrl_freq = ctrl_freq
        self.robot_model = robot_model
        self.f_cutoff = f_cutoff
        self.input_delay_samples = input_delay_samples
        self.input_noise = input_noise
        self.fts_t_std = fts_t_std
        self.fts_f_std = fts_f_std
        self.reconstr_filter = reconstr_filter

        K_m_diag_flex = self.robot_model.K_m_diag_flex.flatten()
        self.static_compliance_model_error = (
            np.random.choice([-1, 1], size=len(K_m_diag_flex))
            * (compliance_error_ratio * np.random.rand(len(K_m_diag_flex)))
            * K_m_diag_flex
        )
        self.K = np.diag(K_m_diag_flex + self.static_compliance_model_error)
        self.K_inv = np.linalg.inv(self.K)

    def compute_theta_fb(self, theta: np.ndarray, force_cartesian: np.ndarray):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        force_cartesian = np.asarray(force_cartesian, dtype=float).reshape(-1, 1)
        j = self.robot_model.jacobian(theta, "TCP")[:, self.robot_model.idx_flex]

        theta_fb = np.zeros(theta.shape, dtype=float)
        theta_fb[self.robot_model.idx_flex] -= (self.K_inv @ j.transpose() @ force_cartesian).flatten()
        return theta_fb


class Trajectory():
    pass

class TrajectoryMilling_Simple(Trajectory):
    def __init__(
        self,
        robot: Robot,
        trajectory_df: pd.DataFrame,
        theta_init=None,
        ee_name: str = "TCP",
        axial_cutting_depth_mm: float = 5.0,
        tcp_orientation_base=None,
        start_at_rest: bool = True,
        workpiece_rotation_base=None,
        smoothing_window: int = 21,
        smoothing_polyorder: int = 3,
    ):
        self.robot = robot
        self.ee_name = ee_name
        self.noj = robot.n
        self.start_at_rest = start_at_rest
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        self.trajectory_df = self._validate_trajectory_df(trajectory_df)
        self.theta_init = self._resolve_theta_init(theta_init)
        self.initial_tcp_pose = self.robot.frame_placement(self.theta_init, self.ee_name)
        self.t_eval = self._build_time_grid()
        self.dt = self._resolve_dt()
        self.trajectory_local = self._build_local_trajectory()
        self.T_base_workpiece = self._build_workpiece_transform(
            axial_cutting_depth_mm=axial_cutting_depth_mm,
            workpiece_rotation_base=workpiece_rotation_base,
        )
        self.desired_tcp_orientation_base = self._resolve_tcp_orientation(tcp_orientation_base)
        self.tcp_positions_base = self._build_tcp_positions_base()
        self.theta = self._solve_ik_sequence()
        # self.theta = self._smooth_joint_trajectory(self.theta)
        self.thetaD = self._compute_joint_velocity_profile()

    def _solve_ik_sequence(self):
        theta = np.zeros((len(self.trajectory_local), self.noj))
        theta_previous = self.theta_init.copy()
        theta[0, :] = theta_previous

        for i, p_base in enumerate(self.tcp_positions_base[1:], start=1):
            target_pose = pin.SE3(self.desired_tcp_orientation_base, p_base)
            theta_previous = self.robot.clik(
                target_pose=target_pose,
                ee_name=self.ee_name,
                q0=theta_previous,
                eps=1e-8,
            )
            theta[i, :] = theta_previous

        return theta
    def _validate_trajectory_df(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        trajectory_df = trajectory_df.reset_index(drop=True).copy()
        required_cols = {"x", "y"}
        missing_cols = required_cols.difference(trajectory_df.columns)
        if missing_cols:
            raise ValueError(f"trajectory_df missing required columns: {sorted(missing_cols)}")
        return trajectory_df

    def _resolve_theta_init(self, theta_init):
        if theta_init is None:
            theta_init = np.array([
                -90.0 * np.pi / 180.0,
                 30.0 * np.pi / 180.0,
                 95.0 * np.pi / 180.0,
                  0.0,
                  0.0,
                  0.0,
            ])
        return np.asarray(theta_init, dtype=float).reshape(-1)

    def _build_time_grid(self):
        if "t" in self.trajectory_df.columns:
            return self.trajectory_df["t"].to_numpy(dtype=float)
        return np.arange(len(self.trajectory_df), dtype=float)

    def _resolve_dt(self):
        if len(self.t_eval) < 2:
            return 0.0
        return float(self.t_eval[1] - self.t_eval[0])

    def _build_local_trajectory(self):
        trajectory_local_xy_m = self.trajectory_df[["x", "y"]].to_numpy(dtype=float) / 1000.0
        trajectory_local_z_m = np.zeros((len(trajectory_local_xy_m), 1))
        return np.hstack([trajectory_local_xy_m, trajectory_local_z_m])

    def _build_workpiece_transform(self, axial_cutting_depth_mm: float, workpiece_rotation_base):
        T_base_TCP0 = self.robot.fkine(self.theta_init, self.ee_name)
        if workpiece_rotation_base is None:
            workpiece_rotation_base = np.eye(3)
        workpiece_rotation = np.asarray(workpiece_rotation_base, dtype=float)

        T_base_workpiece = np.eye(4)
        T_base_workpiece[:3, :3] = workpiece_rotation
        T_base_workpiece[:3, 3] = T_base_TCP0[:3, 3]
        T_base_workpiece[2, 3] += axial_cutting_depth_mm / 1000.0
        T_base_workpiece[:3, 3] -= workpiece_rotation @ self.trajectory_local[0]
        return T_base_workpiece

    def _resolve_tcp_orientation(self, tcp_orientation_base):
        if tcp_orientation_base is None:
            tcp_orientation_base = self.initial_tcp_pose.rotation.copy()
        return np.asarray(tcp_orientation_base, dtype=float)

    def _build_tcp_positions_base(self):
        tcp_positions_base = np.zeros((len(self.trajectory_local), 3))
        tcp_positions_base[0, :] = self.initial_tcp_pose.translation.copy()

        for i, p_local in enumerate(self.trajectory_local[1:], start=1):
            p_aug = np.append(p_local, 1.0)
            tcp_positions_base[i, :] = (self.T_base_workpiece @ p_aug)[:3]

        return tcp_positions_base

    # def _smooth_joint_trajectory(self, theta):
    #     n_samples = theta.shape[0]
    #     if n_samples < 5:
    #         return theta

    #     window = min(int(self.smoothing_window), n_samples)
    #     if window % 2 == 0:
    #         window -= 1
    #     min_window = self.smoothing_polyorder + 2
    #     if min_window % 2 == 0:
    #         min_window += 1
    #     if window < min_window:
    #         return theta

    #     theta_smooth = savgol_filter(
    #         theta,
    #         window_length=window,
    #         polyorder=self.smoothing_polyorder,
    #         axis=0,
    #         mode="interp",
    #     )
    #     theta_smooth[0, :] = theta[0, :]
    #     theta_smooth[-1, :] = theta[-1, :]
    #     return theta_smooth

    def _compute_joint_velocity_profile(self):
        if len(self.t_eval) < 2:
            thetaD = np.zeros_like(self.theta)
        else:
            thetaD = np.gradient(self.theta, self.t_eval, axis=0)
            if self.start_at_rest:
                thetaD[0, :] = 0.0
                thetaD[-1, :] = 0.0
        return thetaD

    def _compute_joint_acceleration_profile(self):
        if len(self.t_eval) < 2:
            thetaDD = np.zeros_like(self.theta)
        else:
            thetaDD = np.gradient(self.thetaD, self.t_eval, axis=0)
            if self.start_at_rest:
                thetaDD[0, :] = 0.0
                thetaDD[-1, :] = 0.0
        return thetaDD

    # Alternative feed-task implementation kept for reference only.
    #
    # def _task_twist(self, q, idx):
    #     current_pose = self.robot.frame_placement(q, self.ee_name)
    #     position_error = self.tcp_positions_base[idx, :] - current_pose.translation
    #     feed_speed_base = self.trajectory_df["feed_mm_s"].to_numpy(dtype=float) / 1000.0
    #     path_tangent_base = np.zeros_like(self.tcp_positions_base)
    #     path_tangent_base[1:-1, :] = self.tcp_positions_base[2:, :] - self.tcp_positions_base[:-2, :]
    #     path_tangent_base[0, :] = self.tcp_positions_base[1, :] - self.tcp_positions_base[0, :]
    #     path_tangent_base[-1, :] = self.tcp_positions_base[-1, :] - self.tcp_positions_base[-2, :]
    #     norms = np.linalg.norm(path_tangent_base, axis=1, keepdims=True)
    #     norms[norms < 1e-12] = 1.0
    #     path_tangent_base /= norms
    #     feedforward_velocity = feed_speed_base[idx] * path_tangent_base[idx, :]
    #     linear_velocity = feedforward_velocity.reshape(3, 1) + 5.0 * position_error.reshape(3, 1)
    #     current_tool_z = current_pose.rotation[:, 2]
    #     desired_tool_z = self.desired_tcp_orientation_base[:, 2]
    #     angular_velocity = 3.0 * np.cross(current_tool_z, desired_tool_z).reshape(3, 1)
    #     return np.vstack([linear_velocity, angular_velocity])
    #
    # def _solve_resolved_rate_sequence(self):
    #     theta = np.zeros((len(self.trajectory_local), self.noj))
    #     thetaD = np.zeros_like(theta)
    #     theta[0, :] = self.theta_init.copy()
    #
    #     for i in range(len(self.trajectory_local) - 1):
    #         q_i = theta[i, :].copy()
    #         J_task = self.robot.jacobian(q_i, self.ee_name)
    #         task_twist = self._task_twist(q_i, i)
    #         normal_matrix = J_task @ J_task.T + 1e-6 * np.eye(J_task.shape[0])
    #         qdot = (J_task.T @ np.linalg.solve(normal_matrix, task_twist)).reshape(-1)
    #         thetaD[i, :] = qdot
    #         theta[i + 1, :] = pin.integrate(self.robot.model, q_i, qdot * self.dt)
    #
    #     if len(theta) > 1:
    #         thetaD[-1, :] = thetaD[-2, :]
    #
    #     self.theta = self._smooth_joint_trajectory(theta)
    #     self.thetaD = np.gradient(self.theta, self.t_eval, axis=0)
    #     self.thetaDD = np.gradient(self.thetaD, self.t_eval, axis=0)

class SettingsRealParameter(): 
    def __init__(self):
        self.URDF_PATH = URDF_PATH_V3
        self.JOINT_STATE = np.array([False, False, False, True, True, True])
        self.K_m_diag = np.array([1.15e6, 3.76e6, 1.29e6, np.NAN, np.NAN, np.NAN]) 
        self.D_m_diag = np.array([2.82e3, 0.61e3, 1.87e3, np.NAN, np.NAN, np.NAN])
        
        
class Solver:
    def __init__(
        self,
        robot: Robot,
        environment: Environment,
        trajectory: Trajectory,
        controller: Controller | None = None,
    ):
        self.robot = robot
        self.environment = environment
        self.t_eval = trajectory.t_eval 
        self.theta = trajectory.theta
        self.thetaD = trajectory.thetaD
        self.t_end = trajectory.t_eval[-1]
        self.reference_system = 'TCP'
        self.Kp_flex = self.robot.K_m_diag_flex
        self.controller = controller

    def _rk4_step(self, f, t, dt, x, theta, thetaD, tau_ext, f_ext):
        k1 = f(t, x, theta, thetaD, tau_ext, f_ext)
        k2 = f(t + dt/2, x + dt/2 * k1, theta, thetaD, tau_ext, f_ext)
        k3 = f(t + dt/2, x + dt/2 * k2, theta, thetaD, tau_ext, f_ext)
        k4 = f(t + dt, x + dt * k3, theta, thetaD, tau_ext, f_ext)
        
        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    
    @staticmethod
    def _assemble_q_full(robot: Robot, q_flex, theta, n):
        q_full = np.full((robot.n, 1), np.nan)
        q_full[robot.idx_flex, :] = q_flex
        q_full[robot.idx_rigid] = theta[robot.idx_rigid].reshape(-1, 1)
        return q_full
     
    @staticmethod
    def _compute_q_flex(x_flex, n):
        return x_flex[:n].reshape(-1, 1)

    def _initial_flexible_state(self):
        theta0 = self.theta[0, :]
        thetaD0 = self.thetaD[0, :]
        q_flex0 = theta0[self.robot.idx_flex].reshape(-1, 1)
        qD_flex0 = thetaD0[self.robot.idx_flex].reshape(-1, 1)
        return np.vstack([q_flex0, qD_flex0]).flatten()

    def _compensated_theta(self, theta, f_ext):
        theta_cmd = np.asarray(theta, dtype=float).reshape(-1).copy()
        if self.controller is None:
            return theta_cmd
        theta_cmd += self.controller.compute_theta_fb(theta_cmd, f_ext)
        return theta_cmd

    def _ode_system(self, t, x_flex, theta, thetaD, tau_ext, f_ext):
            n = self.n_flex
            q_flex = self._compute_q_flex(x_flex, n)
            qD_flex = x_flex[n:2*n].reshape(-1, 1)
            theta_cmd = self._compensated_theta(theta, f_ext)

            q_full = self._assemble_q_full(self.robot, q_flex, theta_cmd, n)

            qD_full = np.full((self.robot.n, 1), np.nan)               
            qD_full[self.robot.idx_flex, :] = qD_flex.reshape(-1, 1)
            qD_full[self.robot.idx_rigid, :] = thetaD[self.robot.idx_rigid].reshape(-1, 1)

            tau_flex_joint = (
                - self.robot.K_m_diag_flex * (q_flex - theta_cmd[self.robot.idx_flex].reshape(-1, 1))
                - self.robot.D_m_diag_flex * (qD_flex - thetaD[self.robot.idx_flex].reshape(-1, 1))
            ).reshape(-1, 1)
   
            qDD_flex = self.robot.forward_dynamics(q_full, qD_full, tau_ext, tau_u=tau_flex_joint)

            return np.vstack([qD_flex, qDD_flex]).flatten()

    def _calc_fkine_vec(self, q_full, ee_name):
        T = self.robot.fkine(q_full, ee_name)
        rot = R.from_matrix(T[:3, :3])
        return np.hstack([T[:3, -1], rot.as_euler('xyz', degrees=False) ])

    def solve(self):
        
        noj = self.robot.n
        dt = self.t_eval[1] - self.t_eval[0]
        len_t_eval = len(self.t_eval)

        # Define logs
        x_flex = np.zeros([len(self.t_eval), 2*len(self.robot.idx_flex)])
        x0_flex = self._initial_flexible_state()
        x_flex[0, :] = x0_flex

        self.x_full = np.full((len(self.t_eval), 2*noj), np.nan)
        self.f_ext = np.full([len(self.t_eval), 6], np.nan)
        self.tau_ext = np.full([len(self.t_eval), len(self.robot.idx_flex)], np.nan)
        self.fkine_vec = np.full([len(self.t_eval), 6], np.nan)

        # Main loop
        self.n_flex = x_flex.shape[1] // 2 # number of elements for flexible variables
        for i, t in enumerate(self.t_eval):

            if i < len_t_eval - 1:
                self.environment.set_t(t)

                # external interaction
                q_flex = self._compute_q_flex(x_flex=x_flex[i, :], n=self.n_flex)
                theta_cmd = self._compensated_theta(self.theta[i, :], self.environment.get_f_ext())
                q_full = self._assemble_q_full(self.robot, q_flex=q_flex, theta=theta_cmd, n=self.n_flex)
                tau_ext = self.environment.compute_tau_ext(q_full) ## HERE YOU INTERACT WITH THE ROBOT AT THE TCP
                f_ext = self.environment.get_f_ext().copy()
                theta_cmd = self._compensated_theta(self.theta[i, :], f_ext)
                q_full = self._assemble_q_full(self.robot, q_flex=q_flex, theta=theta_cmd, n=self.n_flex)

                # additional logging
                self.f_ext[i, :] = f_ext.flatten()
                self.tau_ext[i, :] = tau_ext.flatten()
                self.fkine_vec[i, :] = self._calc_fkine_vec(q_full, self.reference_system) ## HERE YOU SEE THE RESPONSE ON THE TCP

                # integration
                x_flex[i+1, :] = self._rk4_step(
                    self._ode_system,
                    t,
                    dt,
                    x_flex[i, :],
                    self.theta[i, :],
                    self.thetaD[i, :],
                    tau_ext,
                    f_ext,
                )

                if np.isnan(x_flex[i+1, :]).any():
                    raise Exception("NaN detected")

            # final logging step
            self.f_ext[-1, :] = self.environment.get_f_ext().flatten()
            self.tau_ext[-1, :] = tau_ext.flatten()
            self.fkine_vec[-1, :] = self._calc_fkine_vec(q_full, self.reference_system)

            if i % 100 == 0:
                print(f"{t/self.t_end*100:.2f}%")

        # Assemble x_flex + x_rigid -> x_full
        indices_x_flex = np.unique([self.robot.idx_flex, noj+self.robot.idx_flex])
        indices_x_rigid = np.unique([self.robot.idx_rigid, noj+self.robot.idx_rigid])
        self.x_full[:, indices_x_flex] = x_flex
        self.x_full[:, indices_x_rigid] = np.hstack([self.theta[:, self.robot.idx_rigid], self.thetaD[:, self.robot.idx_rigid]])
                    
    def get_qs(self):
        n = self.x_full.shape[1] // 2 # 4 are the number of outputs -> q, theta, qD, thetaD
        q           = self.x_full[:, 0:n]
        qD          = self.x_full[:, n:2*n]
        return q, qD
    
    def get_fkine_vec(self):
        return self.fkine_vec
    
    def get_tau_ext(self):
        return self.tau_ext
    
    def get_f_ext(self):
        return self.f_ext




# ====================================
# Funcs
# ====================================

def load_precomputed_feed_curve(csv_path: str | Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    required_cols = {"x_support", "feed_curve"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Feed-profile CSV missing required columns: {sorted(missing_cols)}")

    feed_curve = df[["x_support", "feed_curve"]].to_numpy(dtype=float)
    order = np.argsort(feed_curve[:, 0], kind="stable")
    feed_curve = feed_curve[order]

    unique_support_mask = np.ones(len(feed_curve), dtype=bool)
    unique_support_mask[1:] = np.diff(feed_curve[:, 0]) > 0.0
    feed_curve = feed_curve[unique_support_mask]
    return feed_curve
        

def generate_xy_trajectory(
    path_csv,
    dt_resample,
    feed_curve,
    *,
    start_pos_on_path_mm: float = 0.0,
    final_pos_on_path_mm: float | None = None,
):
    path_df = pd.read_csv(path_csv, index_col=0)
    x = path_df["x"].to_numpy()
    y = path_df["y"].to_numpy()

    # Path arc length
    s_path = np.concatenate((
        [0.0],
        np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    ))

    # Feed profile v(s)
    s_feed = feed_curve[:, 0]
    v_feed = feed_curve[:, 1]

    # remove zero/negative feed values
    valid = v_feed > 0
    s_feed = s_feed[valid]
    v_feed = v_feed[valid]

    s_start = max(float(start_pos_on_path_mm), 0.0)
    path_end_mm = float(s_path[-1])
    if final_pos_on_path_mm is None:
        s_stop = min(float(s_feed[-1]), path_end_mm)
    else:
        s_stop = min(float(final_pos_on_path_mm), float(s_feed[-1]), path_end_mm)
    if s_stop < s_start:
        raise ValueError(
            f"Requested trajectory interval is invalid: start={s_start:.3f} mm, stop={s_stop:.3f} mm."
        )

    support_mask = (s_feed >= s_start) & (s_feed <= s_stop)
    s_feed_window = s_feed[support_mask]
    v_feed_window = v_feed[support_mask]

    endpoint_support = np.array([s_start, s_stop], dtype=float)
    endpoint_feed = np.interp(endpoint_support, s_feed, v_feed)
    s_feed_window = np.concatenate([endpoint_support[:1], s_feed_window, endpoint_support[1:]])
    v_feed_window = np.concatenate([endpoint_feed[:1], v_feed_window, endpoint_feed[1:]])

    unique_support_mask = np.ones(len(s_feed_window), dtype=bool)
    unique_support_mask[1:] = np.diff(s_feed_window) > 0.0
    s_feed_window = s_feed_window[unique_support_mask]
    v_feed_window = v_feed_window[unique_support_mask]

    # cumulative time
    t_feed = np.zeros_like(s_feed_window)
    if len(t_feed) > 1:
        t_feed[1:] = np.cumsum(np.diff(s_feed_window) / v_feed_window[:-1])

    # uniform sampled 
    t_uniform = np.arange(0, t_feed[-1] + dt_resample, dt_resample)
    s_uniform = np.interp(t_uniform, t_feed, s_feed_window)
    x_uniform = np.interp(s_uniform, s_path, x)
    y_uniform = np.interp(s_uniform, s_path, y)
    v_uniform = np.interp(s_uniform, s_feed_window, v_feed_window)

    return pd.DataFrame({
        "t": t_uniform,
        "s": s_uniform,
        "x": x_uniform,
        "y": y_uniform,
        "feed_mm_s": v_uniform,
    })




def print_initial_tcp_state(robot: Robot, q0, ee_name: str = "TCP"):
    tcp_pose = robot.frame_placement(q0, ee_name)
    tcp_position = tcp_pose.translation
    tcp_rpy = R.from_matrix(tcp_pose.rotation).as_euler("xyz", degrees=True)

    print("\n=== Initial Robot State ===")
    print("q0 [rad]:", np.asarray(q0, dtype=float))
    print("TCP position [m]:", tcp_position)
    print("TCP orientation matrix:")
    print(tcp_pose.rotation)
    print("TCP orientation xyz [deg]:", tcp_rpy)


def plot_ik_fk_vs_desired(trajectory: Trajectory, robot: Robot, ee_name: str = "TCP", *, show=True):
    fk_positions_base = np.zeros_like(trajectory.tcp_positions_base)

    for i, q in enumerate(trajectory.theta):
        fk_positions_base[i, :] = robot.frame_placement(q, ee_name).translation

    desired_xy_mm = trajectory.tcp_positions_base[:, :2] * 1000.0
    fk_xy_mm = fk_positions_base[:, :2] * 1000.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        desired_xy_mm[:, 0],
        desired_xy_mm[:, 1],
        linewidth=1.8,
        label="Desired trajectory",
    )
    ax.plot(
        fk_xy_mm[:, 0],
        fk_xy_mm[:, 1],
        "--",
        linewidth=1.4,
        label="FK of IK trajectory",
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Desired TCP path vs FK of IK solution")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_joint_trajectory_diagnostics(trajectory: Trajectory, *, show=True):
    t_eval = np.asarray(trajectory.t_eval, dtype=float)
    theta = np.asarray(trajectory.theta, dtype=float)
    thetaD = np.asarray(trajectory.thetaD, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for j in range(theta.shape[1]):
        axes[0].plot(t_eval, theta[:, j], linewidth=1.2, label=f"q{j+1}")
        axes[1].plot(t_eval, thetaD[:, j], linewidth=1.2, label=f"qD{j+1}")

    axes[0].set_ylabel("Joint position [rad]")
    axes[0].set_title("Joint trajectory generated by IK")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3)

    axes[1].set_ylabel("Joint velocity [rad/s]")
    axes[1].set_xlabel("t [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes

def plot_tcp_tracking_2d(trajectory: Trajectory, fkine_vec, *, show=True):
    desired_xy_mm = trajectory.tcp_positions_base[:, :2] * 1000.0
    actual_xy_mm = fkine_vec[:, :2] * 1000.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        desired_xy_mm[:, 0],
        desired_xy_mm[:, 1],
        linewidth=1.8,
        label="Driven trajectory",
    )
    ax.plot(
        actual_xy_mm[:, 0],
        actual_xy_mm[:, 1],
        "--",
        linewidth=1.4,
        label="Current trajectory",
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("TCP trajectory tracking in base XY")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_tcp_tracking_error(trajectory: Trajectory, fkine_vec, t_eval, *, show=True):
    desired_xy_mm = trajectory.tcp_positions_base[:, :2] * 1000.0
    actual_xy_mm = fkine_vec[:, :2] * 1000.0
    tracking_error_xy_mm = actual_xy_mm - desired_xy_mm
    tracking_error_norm_mm = np.linalg.norm(tracking_error_xy_mm, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t_eval, tracking_error_xy_mm[:, 0], linewidth=1.2)
    axes[0].set_ylabel("e_x [mm]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_eval, tracking_error_xy_mm[:, 1], linewidth=1.2)
    axes[1].set_ylabel("e_y [mm]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_eval, tracking_error_norm_mm, linewidth=1.4)
    axes[2].set_ylabel("|e| [mm]")
    axes[2].set_xlabel("t [s]")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("TCP tracking error in base XY")
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes

if __name__ == '__main__':
    dt_ik = 1e-4
    
    # Chafli geometry/path assets
    URDF_PATH_V3 =  'data/urdf/v2/Staeubli-Huynh 1.urdf'
    FEED_PROFILE_CSV = "optimized_feed/optimization_offline_cc/Optimal_feed_curve.csv"
    MILLING_PATH_CSV = "data/paths/Workpiece_long_with_start_milling_path.csv"
    WORKPIECE_PICKLE = "data/paths/Workpiece_long_with_start.pickle"
    AXIAL_CUTTING_DEPTH_MM = 5.0
    START_POS_ON_PATH_MM = 150
    FINAL_POS_ON_PATH_MM = 300
    
    feedrate_optimized = load_precomputed_feed_curve(FEED_PROFILE_CSV)
    milling_path_df = pd.read_csv(MILLING_PATH_CSV, index_col=0)
    
    trajectory_df = generate_xy_trajectory(
        MILLING_PATH_CSV,
        dt_ik,
        feedrate_optimized,
        start_pos_on_path_mm=START_POS_ON_PATH_MM,
        final_pos_on_path_mm=FINAL_POS_ON_PATH_MM,
    )
    trajecotry_xy = trajectory_df[["x", "y"]].to_numpy()
    
    
    settings = SettingsRealParameter()
    robot = Robot(settings)
    trajectory = TrajectoryMilling_Simple(
        robot=robot,
        trajectory_df=trajectory_df,
        axial_cutting_depth_mm=AXIAL_CUTTING_DEPTH_MM,
    )
    environment = EnvironmentMilling(
        robot,
        trajectory=trajectory,
        workpiece_pickle=WORKPIECE_PICKLE,
        axial_cutting_depth_mm=AXIAL_CUTTING_DEPTH_MM,
    )

    print_initial_tcp_state(robot, trajectory.theta_init, ee_name="TCP")
    plot_joint_trajectory_diagnostics(trajectory)
    # plot_ik_fk_vs_desired(trajectory, robot, ee_name="TCP")
    
    
            
    controller = ComplianceCompensation(
        ctrl_freq=1.0 / max(dt_ik, 1e-12),
        robot_model=robot,
    )

    solver = Solver(robot, environment, trajectory, controller=controller)
    solver.solve()
    q, qD = solver.get_qs()
    fkine_vec = solver.get_fkine_vec()
    tau_ext = solver.get_tau_ext()
    f_ext = solver.get_f_ext()
    plot_tcp_tracking_2d(trajectory, fkine_vec)
    plot_tcp_tracking_error(trajectory, fkine_vec, trajectory.t_eval)
