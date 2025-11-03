
import casadi
import numpy as np


def pendulum_dynamics():

    q = casadi.MX.sym('q1')
    dq = casadi.MX.sym('dq1')
    tau = casadi.MX.sym('tau')
    x = casadi.vertcat(q, dq, tau)
    
    ddq1_sym = casadi.MX.sym('ddq1')
    dq1_syym = casadi.MX.sym('dq1_sym')
    u_sym = casadi.MX.sym('u_sym')
    x_dot = casadi.vertcat(dq1_syym, ddq1_sym, u_sym)

    u = casadi.MX.sym('u')
    g = 9.81

    # m, l, b, f, gr
    m = casadi.MX.sym('m')  
    l = casadi.MX.sym('l')
    b = casadi.MX.sym('b')
    f = casadi.MX.sym('f')
    gr = casadi.MX.sym('gr')
    p = casadi.vertcat(m, l, b, f, gr)
    
    sign_tanh_aprox = 30.0
    q_offset = q - casadi.pi
    u_with_losses = gr * tau - b * dq - f * casadi.tanh(sign_tanh_aprox * dq)
    ddq = 3 * g / (2 * l) * casadi.sin(q_offset) + 3.0 / (m * l**2) * u_with_losses
    
    f_expl = casadi.vertcat(dq, ddq, u)

    default_params = np.array([1.0, 1.0, 0.2, 0.001, 1.0])

    dyn_fun = casadi.Function('pendulum_dyn', [x, u, p], [
                              f_expl], ['x', 'u', 'p'], ['f_expl'])
    

    # store to struct
    model = casadi.types.SimpleNamespace()
    model.x = x
    model.xdot = x_dot
    model.u = u
    model.z = casadi.vertcat([])
    model.p = p
    model.parameter_values = default_params
    model.f_expl = f_expl
    model.x_start = np.zeros((3))
    model.constraints = casadi.vertcat([])
    model.name = "pendulum_casadi"

    model.dyn_fun = dyn_fun
    return model
