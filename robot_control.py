# -*- coding: utf-8 -*-
"""
"""

from marco_nest_utils import utils

__author__ = 'marco'

import numpy as np
from scipy.integrate import odeint
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import time

from nest_multiscale.nest_multiscale import generate_poisson_trains, evaluate_fir, calculate_instantaneous_fr, set_poisson_fr


class cereb_control():
    def __init__(self, nest_, Cereb_class, robot, desired_trajectory, desired_traj_vel=None,
                 cortical_input=None, control_tau=None):
        self.nest_ = nest_
        self.Cereb_class = Cereb_class

        self.des_trajec = desired_trajectory  # function of time
        self.des_trajec_vel = desired_traj_vel
        self.CTX_pop_list = self.get_CTX_pops()

        self.starting_state = [des_tr(0) for des_tr in self.des_trajec] + [des_tr_v(0) for des_tr_v in self.des_trajec_vel]

        self.T = None           # will be set once linked to simulation handler
        self.max_traj = None    # will be updated later
        self.min_traj = None
        self.max_traj_vel = None
        self.min_traj_vel = None

        self.robot = robot
        self.n_joints = len(robot.q)
        assert self.n_joints == len(desired_trajectory), 'Should provide as many trajectory as robot joints'

        self.u_R_dim = None   # total number of moments
        self.y_R_dim = self.u_R_dim     # total number of error signals to be sent to IO

        self.rng = np.random.default_rng(round(time.time() * 1000))

        self.pop_list_to_robot = None
        self.sd_list_R = None

        self.cortical_input = cortical_input    # the signal delivered from the cortex
        self.control_tau = control_tau          # the tau sequence which will generate the desired trajectory


    def start(self, Sim_time, T_sample):
        self.pop_list_to_robot = self.cereb_control_setup(Sim_time, T_sample)
        self.sd_list_R = utils.attach_spikedetector(self.nest_, self.pop_list_to_robot)
        self.cereb_control_buffers()

        # robot
        self.robot.set_robot_T(T_sample)
        self.robot.n_joints = len(self.robot.q)


    def before_loop(self):
        self.robot.reset_state(self.starting_state)
        self.tau_old = np.zeros((self.prev_steps, 2 * self.n_joints))
        # io_old = np.zeros((prev_steps_io, 2 * self.n_joints))
        # q_val = [0.]  # visualizer buffer


    def beginning_loop(self, trial_time, total_time):
        # set future RBF input
        if self.cortical_input is None:
            self.generate_RBF_activity(trajectory_time=trial_time, simulation_time=total_time)
        else:
            self.generate_RBF_activity(trajectory_time=trial_time, simulation_time=total_time,
                                       ctx_input=self.cortical_input[int(trial_time)] * 200.)


    def ending_loop(self, trial_time, total_time):
        # a) Update robot position
        # use tau of (previous step!) to update robot position
        tau_pos = self.tau[1::2]  # to take odd elements:   [1::2]
        tau_neg = self.tau[0::2]  # to take even elements:  [0::2]

        if self.n_joints == 1: k = 0.4
        if self.n_joints == 2: k = 0.5
        if self.control_tau is not None:
            if self.cortical_input is None:
                self.robot.update_state(        # only apply the control sequence + tau pos and neg difference
                    (tau_pos - tau_neg) * k + self.control_tau(trial_time) * 1)
            else:
                self.robot.update_state(        # apply the control sequence modulated by cortical input
                    (tau_pos - tau_neg) * k +   # + tau pos and neg difference
                    self.control_tau(trial_time) * self.cortical_input[int(trial_time)] / 0.4102)
        else:
            self.robot.update_state((tau_pos - tau_neg) * k)

        self.e_old = np.concatenate((self.e_old, np.zeros((1, self.n_joints))))
        for k in range(self.n_joints):
            # b) Inject robot error in IO
            # joint position has just been updated to tt with tau(tt - T), so select tt!
            e_new = self.get_error_value(trial_time + self.T, self.robot.q[k], k)  # self.robot.q[j])
            self.e_old[-1, k] = e_new
            e = self.e_old[0, k]

            if self.n_joints == 1: e = e * 5.
            if self.n_joints == 2: e = e * 3.
            if e > 0.:
                if e > 7.: e = 7.
                # set the future spike trains (in [tt + T, tt + 2T])
                set_poisson_fr(self.nest_, e, [self.Cereb_class.CTX_pops['US_p'][k]], total_time + self.T,
                               self.T, self.rng)
            elif e < 0.:
                if e < -7.: e = -7.
                # set the future spike trains (in [tt + T, tt + 2T])
                set_poisson_fr(self.nest_, -e, [self.Cereb_class.CTX_pops['US_n'][k]], total_time + self.T,
                               self.T, self.rng)
        self.e_old = self.e_old[1:, :]

        # update tau value for next step
        new_tau = calculate_instantaneous_fr(self.nest_, self.sd_list_R, self.pop_list_to_robot,
                                             time=total_time, T_sample=self.T)
        # new_tau, t_prev = calculate_instantaneous_burst_activity(self.nest_, self.sd_list_R, self.pop_list_to_robot,
        #                                      time=actual_sim_time, t_prev=t_prev)
        self.tau, self.tau_old = evaluate_fir(self.tau_old, new_tau.reshape((1, self.u_R_dim)), kernel=self.kernel_robot)
        if self.tau_sol is None:
            self.tau_sol = np.array([self.tau])
        else:
            self.tau_sol = np.concatenate((self.tau_sol, [self.tau]), axis=0)

        # print(self.tau_sol)

        # update io
        # new_io = calculate_instantaneous_fr(self.nest_, sd_list_io, list_io,
        #                                     time=actual_sim_time, T_sample=self.T)
        # io, io_old = evaluate_fir(io_old, new_io.reshape((1, 2 * self.n_joints)), kernel=kernel)
        # if io_sol is None:
        #     io_sol = np.array([io])
        # else:
        #     io_sol = np.concatenate((io_sol, [io]), axis=0)

    # def after_loop(self):
        # self.io_sol = io_sol
        # self.tau_sol = tau_sol


    def cereb_control_setup(self, Sim_time, T_sample):
        # set proper sampling time in cereb control
        self.set_T_and_max_func(T_sample, Sim_time)

        # divide dcn in pos tau and neg tau
        pop_list_to_robot = self.Cereb_class.get_dcn_indexes(self.n_joints)

        self.u_R_dim = len(pop_list_to_robot)   # total number of moments
        self.y_R_dim = self.u_R_dim     # total number of error signals to be sent to IO

        return pop_list_to_robot    # , pf_pc_conn, w0


    def cereb_control_buffers(self):
        tau0 = np.array([0.] * self.u_R_dim)
        self.tau_sol = None  # will contain all solutions over time

        self.kernel_robot = half_gaussian(sigma=600., width=300., sampling_time=self.T)

        self.tau = tau0
        self.prev_steps = len(self.kernel_robot)
        self.tau_old = np.zeros((self.prev_steps, 2 * self.n_joints))

        # sd_list_io, list_io = self.attach_io_spike_det()

        # io0 = np.array([0.] * 2)
        # io_sol = None  # will contain all solutions over time

        # io = io0
        # prev_steps_io = len(kernel)
        # io_old = np.zeros((prev_steps, 2 * self.n_joints))

        self.robot.reset_state(self.starting_state)

        self.e_old = np.zeros((int(140 / self.T), self.n_joints))


    def generate_RBF_activity(self, trajectory_time, simulation_time, ctx_input=80.):
        """
        :param time: actual time, in ms
        :param CTX_pops: cortex populations, projecting to glomeruli
        :param ctx_input: signal amplitude
        :return:
        """
        sd = 3. / len(self.des_trajec)

        input_mf_dim = len(self.CTX_pop_list[0])     # dimension of every mf input (j1pos, j1vel, j2pos, ...)
        glom_grid = np.linspace(1, input_mf_dim, input_mf_dim, endpoint=True)

        if self.des_trajec_vel is None:
            for k, des_tr in enumerate(self.des_trajec):
                min_val = self.min_traj[k]
                max_val = self.max_traj[k]
                mu_x = 1 + (input_mf_dim - 1) * (des_tr(trajectory_time) - min_val) / (max_val - min_val)
                # microm
                fr = gaussian_1D(glom_grid, mu_x, sd) * ctx_input

                spike_times = generate_poisson_trains(self.CTX_pop_list[k], fr, self.T, simulation_time, self.rng)
                generator_params = [{"spike_times": s_t, "spike_weights": [1.] * len(s_t)} for s_t in spike_times]
                self.nest_.SetStatus(self.CTX_pop_list[k], generator_params)

        elif self.des_trajec_vel is not None:
            for k, des_tr, des_tr_vel in zip(range(len(self.des_trajec)), self.des_trajec, self.des_trajec_vel):
                if ctx_input < 0.:
                    ctx_input = 0.

                min_val = self.min_traj[k]
                max_val = self.max_traj[k]
                mu_x_pos = 1 + (input_mf_dim-1) * (des_tr(trajectory_time) - min_val) / (max_val - min_val)  # microm
                fr_pos = gaussian_1D(glom_grid, mu_x_pos, sd) * ctx_input

                min_val = self.min_traj_vel[k]
                max_val = self.max_traj_vel[k]
                mu_x_vel = 1 + (input_mf_dim-1) * (des_tr_vel(trajectory_time) - min_val) / (max_val - min_val)  # microm
                fr_vel = gaussian_1D(glom_grid, mu_x_vel, sd) * ctx_input

                spike_times_pos = generate_poisson_trains(self.CTX_pop_list[2*k], fr_pos, self.T, simulation_time, self.rng)
                generator_params = [{"spike_times": s_t, "spike_weights": [1.] * len(s_t)} for s_t in spike_times_pos]
                self.nest_.SetStatus(self.CTX_pop_list[2*k], generator_params)

                spike_times_vel = generate_poisson_trains(self.CTX_pop_list[2*k+1], fr_vel, self.T, simulation_time, self.rng)
                generator_params = [{"spike_times": s_t, "spike_weights": [1.] * len(s_t)} for s_t in spike_times_vel]
                self.nest_.SetStatus(self.CTX_pop_list[2*k+1], generator_params)

    def get_CTX_pops(self):
        '''
        Separate cortical population in subpopulation according to the order:
        J1_pos, J1_vel, J2_pos, J2_vel, ...
        (if vel is not defined: J1_pos, J2_pos, ...)
        :return: List of cortical populations according
        '''
        n_joints = len(self.des_trajec)
        CTX_len = len(self.Cereb_class.CTX_pops['CTX'])

        if self.des_trajec_vel is None:
            sub_list_len = int(CTX_len/n_joints)
            pop_list = [self.Cereb_class.CTX_pops['CTX'][j*sub_list_len:(j+1)*sub_list_len] for j in range(n_joints)]

        elif self.des_trajec_vel is not None:
            sub_list_len = int(CTX_len/(n_joints * 2))
            pop_list = [self.Cereb_class.CTX_pops['CTX'][j*sub_list_len:(j+1)*sub_list_len] for j in range(n_joints * 2)]

        return pop_list


    def set_T_and_max_func(self, t_sample_, sim_time_):
        self.T = t_sample_

        t_grid = np.linspace(0, (sim_time_ - sim_time_ % t_sample_), int(sim_time_ / t_sample_), endpoint=False)
        self.max_traj = [max([des_tr(t) for t in t_grid]) for des_tr in self.des_trajec]
        self.min_traj = [min([des_tr(t) for t in t_grid]) for des_tr in self.des_trajec]

        if self.des_trajec_vel is not None:
            self.max_traj_vel = [max([des_tr(t) for t in t_grid]) for des_tr in self.des_trajec_vel]
            self.min_traj_vel = [min([des_tr(t) for t in t_grid]) for des_tr in self.des_trajec_vel]


    def get_error_value(self, time, robot_state, joint):
        error = self.des_trajec[joint](time) - robot_state
        return error


    def attach_io_spike_det(self):
        io_list = []
        sub_pop_IO_len = int(len(self.Cereb_class.Cereb_pops['IO']) / self.n_joints)
        for j in range(self.n_joints):
            io_neg = list(self.Cereb_class.Cereb_pops['IO'][
                          2 * j * sub_pop_IO_len // 2:(2 * j + 1) * sub_pop_IO_len // 2])
            io_pos = list(self.Cereb_class.Cereb_pops['IO'][
                          (2 * j + 1) * sub_pop_IO_len // 2:(2 * j + 2) * sub_pop_IO_len // 2])
            io_list += [io_neg] + [io_pos]

        return utils.attach_spikedetector(self.nest_, io_list), io_list



class rbot():
    def __init__(self):
        q0 = np.zeros((1, 1))
        qd0 = np.zeros((1, 1))
        g = 9.81
        l2 = 1.  # m
        m2 = 1.  # kg

        self.int_t = None   # to be set by call in simulation_handler

        self.q = q0
        self.qd = qd0
        self.joint_pos = None
        self.joint_vel = None
        self.rhs = self.robot_rhs(m2, l2, g)


    def set_robot_T(self, t_sample):
        sub_interv = 10  # integrate in 1 T period, sampling every T/sub_interv
        self.int_t = np.linspace(0, t_sample / 1000., sub_interv + 1)


    def robot_rhs(self, m, l, g):
        I = m * l ** 2 / 3

        def rhs(x, t, u):
            qd = x[1]
            qdd = (u[0] - g * m * l / 2 * np.sin(x[0])) / I
            return np.array([qd, qdd])

        return rhs

    def update_state(self, tau):

        state = np.concatenate((self.q, self.qd))

        sol = odeint(self.rhs, state, self.int_t, args=(tau,))

        self.q = np.array([sol[-1, 0]])
        self.qd = np.array([sol[-1, 1]])

        if self.joint_pos is None:
            self.joint_pos = np.array([self.q])
        else:
            self.joint_pos = np.concatenate((self.joint_pos, [self.q]), axis=0)

        if self.joint_vel is None:
            self.joint_vel = np.array([self.qd])
        else:
            self.joint_vel = np.concatenate((self.joint_vel, [self.qd]), axis=0)

    def reset_state(self, starting_state_):
        self.q = np.array([starting_state_[0]])
        self.qd = np.array([starting_state_[1]])

        # self.joint_pos = np.concatenate((self.joint_pos, self.q * np.ones(1)), axis=0)


class rrbot():
    def __init__(self):
        q0 = np.zeros(2)
        qd0 = np.zeros(2)
        g = 0. # 9.81
        l1 = 1.  # m
        l2 = 1.  # m
        m1 = 1.  # kg
        m2 = 1.  # kg

        self.int_t = None   # to be set by call in simulation_handler

        self.q = q0
        self.qd = qd0
        self.joint_pos = None
        self.joint_vel = None
        self.rhs = self.robot_rhs(m1, m2, l1, l2, g)


    def set_robot_T(self, t_sample):
        sub_interv = 10  # integrate in 1 T period, sampling every T/sub_interv
        self.int_t = np.linspace(0, t_sample / 1000., sub_interv + 1)


    def robot_rhs(self, m1, m2, l1, l2, g):

        M = lambda q: np.array([[(m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * np.cos(q[1]), m2 * l2 ** 2 + m2 * l1 * l2 * np.cos(q[1])],
                                [m2 * l2 ** 2 + m2 * l1 * l2 * np.cos(q[1]), m2 * l2 ** 2]])

        C = lambda q, qd: np.array([[-m2 * l1 * l2 * (2 * qd[0] * qd[1] + qd[0] ** 2) * np.sin(q[1])],   # change sign according to notation
                                    [-m2 * l1 * l2 * qd[0] * qd[1] * np.sin(q[1])]])

        G = lambda q, qd : np.array([[-(m1 + m2) * g * l1 * np.sin(q[0]) - m2 * g * l2 * np.sin(q[0] + q[1])],
                                     [-m2 * g * l2 * np.sin(q[0] + q[1])]])

        def rhs(x, t, u):
            q0 = x[0:2].T
            qd0 = x[2:4].T
            qd = qd0
            qdd = np.linalg.solve(M(q0), u - (G(q0, qd0).T)[0] - (C(q0, qd0).T)[0])
            return np.concatenate((qd, qdd))

        return rhs

    def update_state(self, tau):

        state = np.concatenate((self.q, self.qd))

        sol = odeint(self.rhs, state, self.int_t, args=(tau,))

        self.q = sol[-1, 0:2]
        self.qd = sol[-1, 2:4]

        if self.joint_pos is None:
            self.joint_pos = np.array([self.q])
        else:
            self.joint_pos = np.concatenate((self.joint_pos, [self.q]), axis=0)

        if self.joint_vel is None:
            self.joint_vel = np.array([self.qd])
        else:
            self.joint_vel = np.concatenate((self.joint_vel, [self.qd]), axis=0)

    def reset_state(self, starting_state_):
        self.q = np.array(starting_state_[0:2])
        self.qd = np.array(starting_state_[2:4])

        # self.joint_pos = np.concatenate((self.joint_pos, self.q * np.ones(1)), axis=0)


def half_gaussian(sigma, width, sampling_time):
    s = sigma/sampling_time
    w = int(width/sampling_time)

    time_p = w - np.linspace(0, w, w+1)   # create kernel linspace
    # create a half gaussian kernel:
    kernel = 1. / (np.sqrt(2. * np.pi) * s) * np.exp(-np.power((time_p - 0.) / s, 2.) / 2)
    kernel = kernel/kernel.sum()                # normalize
    # vsl.simple_plot(-time_p, kernel)          # visualize

    return kernel



def gaussian_2D(x, mu, sigma):
    g = np.exp(-np.power(np.linalg.norm(x - mu, axis=1) / sigma, 2) / 2) / np.exp(0)
    return np.round(g, 3)


def gaussian_1D(x, mu, sigma):
    g = np.exp(-np.power((np.array(x) - mu) / sigma, 2) / 2) / np.exp(0)
    return np.round(g, 3)



def plot_3D(x, z, fr, tr_time):
    if tr_time % 200 == 0:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x, z, fr, cmap='viridis', edgecolor='none')
        ax.set_title('Surface plot')
        ax.set_xlim3d(0, 400)
        ax.set_ylim3d(0, 400)
        ax.set_zlim3d(0, 40)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        plt.show()