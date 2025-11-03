import casadi
import numpy as np

def pacejca_single_track_casadi():
    # State variables
    v_x = casadi.MX.sym('v_x')
    v_y = casadi.MX.sym('v_y')
    r = casadi.MX.sym('r')
    friction = casadi.MX.sym('friction')
    state = casadi.vertcat(v_x, v_y, r, friction)

    # Control inputs
    omega_wheels = casadi.MX.sym('omega_wheels')
    delta = casadi.MX.sym('delta')
    control = casadi.vertcat(omega_wheels, delta)

    # Parameters
    p = casadi.MX.sym('p', 30)  # Total number of parameters

    # Constants
    m = 5.1
    g = 9.81
    L = 0.33
    eps = 1e-6

    # Vehicle parameters
    I_z = p[0]
    lr = p[1]
    lf = L - lr
    Cd0 = p[2]
    Cd2 = p[3]
    Cd1 = p[4]
    I_e = p[5]
    K_fi = p[6]
    b1 = p[7]
    b0 = p[8]
    R = p[9]

    # Front tire parameters
    Sx_p_front = p[10]
    Alpha_p_front = p[11]
    By_front = p[12]
    Cy_front = p[13]
    Dy_front = p[14]
    Ey_front = p[15]
    Bx_front = p[16]
    Cx_front = p[17]
    Dx_front = p[18]
    Ex_front = p[19]

    # Rear tire parameters
    Sx_p_rear = p[20]
    Alpha_p_rear = p[21]
    By_rear = p[22]
    Cy_rear = p[23]
    Dy_rear = p[24]
    Ey_rear = p[25]
    Bx_rear = p[26]
    Cx_rear = p[27]
    Dx_rear = p[28]
    Ex_rear = p[29]

    # Define tire forces model function
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        # Convert slip angle from radians to degrees
        slip_angle_deg = slip_angle_rad * 180.0 / casadi.pi

        # Calculate normalized slip ratio and slip angle
        Sx_norm = slip_ratio / wp['Sx_p']
        Alpha_norm = slip_angle_deg / wp['Alpha_p']

        # Compute the resultant slip
        S_resultant = casadi.sqrt(Sx_norm**2 + Alpha_norm**2)

        # Find the modified slip factors
        Sx_mod = S_resultant * wp['Sx_p']
        Alpha_mod = S_resultant * wp['Alpha_p']

        # Calculate the Lateral Force using Pacejka formula
        Fy = ((Alpha_norm / S_resultant) * wp['Dy'] * casadi.sin(wp['Cy'] * casadi.atan((wp['By'] * Alpha_mod) -
             wp['Ey']  * -1.0 * (wp['By'] * Alpha_mod - casadi.atan(wp['By'] * Alpha_mod)))))

        # Calculate the Longitudinal Force using Pacejka formula
        Fx = ((Sx_norm / S_resultant) * wp['Dx'] * casadi.sin(wp['Cx'] * casadi.atan((wp['Bx'] * Sx_mod) -
             wp['Ex'] * -1.0 *  (wp['Bx'] * Sx_mod - casadi.atan(wp['Bx'] * Sx_mod)))))

        return Fx, - Fy

    # Wrap tire parameters into dictionaries
    wp_front = {
        'Sx_p': Sx_p_front,
        'Alpha_p': Alpha_p_front,
        'By': By_front,
        'Cy': Cy_front,
        'Dy': Dy_front,
        'Ey': Ey_front,
        'Bx': Bx_front,
        'Cx': Cx_front,
        'Dx': Dx_front,
        'Ex': Ex_front,
    }

    wp_rear = {
        'Sx_p': Sx_p_rear,
        'Alpha_p': Alpha_p_rear,
        'By': By_rear,
        'Cy': Cy_rear,
        'Dy': Dy_rear,
        'Ey': Ey_rear,
        'Bx': Bx_rear,
        'Cx': Cx_rear,
        'Dx': Dx_rear,
        'Ex': Ex_rear,
    }


    # Compute F_drag
    F_drag = Cd0 * casadi.sign(v_x) + Cd1 * v_x + Cd2 * v_x * v_x

    # Compute slip angles
    slip_angle_front = casadi.atan((v_y + lf * r) / (v_x + eps)) - delta
    slip_angle_rear =  casadi.atan((v_y - lf * r) / (v_x + eps))

    # Compute slip ratios
    v_front = v_x * casadi.cos(delta) + (v_y + r * lr) * casadi.sin(delta)
    slip_ratio_front = (omega_wheels - v_front) / (v_front + eps)
    slip_ratio_rear = (omega_wheels - v_x) / (v_x + eps)

    # Compute tire forces using tire model
    Fx_front, Fy_front = tire_forces_model(slip_angle_front, slip_ratio_front, wp_front)
    Fx_rear, Fy_rear = tire_forces_model(slip_angle_rear, slip_ratio_rear, wp_rear)

    # Normal forces
    Fz_front = (m * g * lr) / L
    Fz_rear = (m * g * lf) / L

    # Compute tire forces
    Fxf = Fz_front * Fx_front * friction
    Fyf = Fz_front * Fy_front * friction
    Fxr = Fz_rear * Fx_rear * friction
    Fyr = Fz_rear * Fy_rear * friction

    # Compute dynamics
    v_x_dot = (1.0 / m) * (Fxr + Fxf * casadi.cos(delta) -
                           Fyf * casadi.sin(delta) - F_drag + m * v_y * r)
    v_y_dot = (1.0 / m) * (Fxf * casadi.sin(delta) + Fyr +
                           Fyf * casadi.cos(delta) - m * v_x * r)
    r_dot = (1.0 / I_z) * ((Fxf * casadi.sin(delta) + Fyf * casadi.cos(delta)) * lf - Fyr * lr)

    # Friction does not change over time in this model
    friction_dot = casadi.MX.zeros(1)

    # State derivatives
    state_dot = casadi.vertcat(v_x_dot, v_y_dot, r_dot, friction_dot)

    # Create CasADi function
    dyn_fun = casadi.Function('car_dyn', [state, control, p], [state_dot],
                              ['state', 'control', 'params'], ['state_dot'])
        
    # get state return tire forces
    tire_forc_fun = casadi.Function('tire_forces', [state, control, p], [casadi.vertcat(Fyf, Fyr, Fxf, Fxr)],
                                    ['state', 'control', 'params'], ['tire_forces']) 
    
    # return slips 
    slip_fun = casadi.Function('slip', [state, control, p], [casadi.vertcat(slip_angle_front, slip_angle_rear, slip_ratio_front, slip_ratio_rear)],
                                    ['state', 'control', 'params'], ['tire_forces'])
    
    return dyn_fun, tire_forc_fun