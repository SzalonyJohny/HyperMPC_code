import casadi as cs
import numpy as np
import scipy.linalg
import torch
import l4casadi as l4c
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
import time
import os
import scipy
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random
import string



class MPC:
    def __init__(self, model, dt, cfg_mpc, random_name=False):
        self.N = cfg_mpc.N
        self.horizon = dt * self.N
        self.cfg_mpc = cfg_mpc
        self.model = model

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())
        

    def ocp(self):
        model = self.model

        t_horizon = self.horizon
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Dimensions
        nx = 3+4+3+3
        nu = 4
        ny = nx + nu
        ny_e = nx

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.dims.ny_e = ny_e
        ocp.solver_options.tf = t_horizon
        ocp.solver_options.N_horizon = N

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        
        Q_diag = np.array([
            100.0,      # x
            100.0,      # y
            100.0,      # z
            1.0e1,     # qw
            1.0e1,     # qx
            1.0e1,     # qy
            1.0e1,     # qz
            0.001,        # vbx
            0.001,        # vby
            0.001,        # vbz
            1.0,       # wx
            1.0,       # wy
            1.0,       # wz
        ])
        Q = np.diag(Q_diag)
        R = 0.01 * np.eye(nu)
        
        Vx = np.zeros((ny, nx))
        Vx[:nx, :] = np.eye(nx)
        
        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :] = np.eye(nx)
        
        Vu = np.zeros((ny, nu))
        Vu[nx:ny, :] = np.eye(nu)
        
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q * 1.0 # FIXME
        ocp.cost.Vx = Vx
        ocp.cost.Vx_e = Vx_e
        ocp.cost.Vu = Vu

        hov_w = 0.5 #  np.sqrt((mq*g0)/(4*Ct))
        ocp.cost.yref   = np.array([0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, hov_w, hov_w, hov_w, hov_w])
        ocp.cost.yref_e = np.array([0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # state constrains
        ocp.constraints.lbx = np.array([])
        ocp.constraints.ubx = np.array([])
        ocp.constraints.idxbx = np.array([])
        ocp.constraints.x0 = model.x_start

        # control constraints
        max_thrust = 13.0
        ocp.constraints.lbu = np.array([0,0,0,0])
        ocp.constraints.ubu = np.array([max_thrust, max_thrust, max_thrust, max_thrust])
        ocp.constraints.idxbu = np.array([0,1,2,3])

        # Solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # ocp.solver_options.regularize_method = 'NO_REGULARIZE'
        # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        
        ocp.solver_options.integrator_type = 'ERK'  # self.params.integrator_type
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.parameter_values = model.parameter_values
        

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        return model_ac
