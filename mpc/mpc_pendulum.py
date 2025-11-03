import casadi as cs
import numpy as np
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
    def __init__(self, model, dt, cfg_mpc, random_name = False):
        self.N = cfg_mpc.N
        self.model = model
        self.horizon = dt * self.N
        self.cfg_mpc = cfg_mpc
        
        self.export_code_dir = 'c_generated_code/'
        
        if random_name:
            self.export_code_dir += ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

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
        nx = 3
        nu = 1
        ny = nx + nu
        ny_e = nx

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon
        ocp.solver_options.N_horizon = N
        
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        
        # l(x,u,z,p) = 0.5 || y(x,u,z,p) - y_ref ||^2_W
        # m(x,p) = 0.5 || y^e(x,p) - y_ref^e||^2_{W^e}
        t = self.cfg_mpc

        Q_mat = np.diag([t.Q_q, t.Q_dq,  t.Q_u])
        R_mat = np.diag([t.R])
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
                    
        ocp.cost.W_e =  Q_mat * t.Qe_scaler * N

        ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x
        
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))
            
        # max tau
        max_tau = t.max_torque
        ocp.constraints.lbx = np.array([-max_tau])
        ocp.constraints.ubx = np.array([max_tau])
        ocp.constraints.idxbx = np.array([2]) # index of tau
        ocp.constraints.lbx_e = ocp.constraints.lbx
        ocp.constraints.ubx_e = ocp.constraints.ubx
        ocp.constraints.idxbx_e = ocp.constraints.idxbx

        # Set constraints
        d_tau_max = t.max_dtorque
        ocp.constraints.lbu = np.array([-d_tau_max])
        ocp.constraints.ubu = np.array([d_tau_max])
        ocp.constraints.idxbu = np.array([0])
        
        ocp.constraints.x0 = model.x_start

        # Solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # ocp.solver_options.regularize_method = 'PROJECT'
        # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        ocp.solver_options.integrator_type = self.cfg_mpc.integrator_type
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.parameter_values = model.parameter_values        
        ocp.code_export_directory = self.export_code_dir
        # ocp.solver_options.levenberg_marquardt = 1e-6
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

