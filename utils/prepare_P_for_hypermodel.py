import numpy as np
import scipy
import scipy.interpolate


class ObservationsQueue:
    def __init__(self, len, obs_dim):
        self.len = len
        self.obs_dim = obs_dim
        self.queue = np.zeros((len, obs_dim))

    def forward(self, obs):
        self.queue = np.roll(self.queue, -1, axis=0)
        self.queue[-1, :] = obs
        return self.queue


class HyperModelPreprocesor:
    def __init__(self, N_mpc, dt_mpc,
                 N_train, dt_train,
                 M_len, M_dim, M_from_x_fun=None):

        self.obs_queue = ObservationsQueue(M_len, M_dim)

        self.N_mpc = N_mpc + 1
        self.dt_mpc = dt_mpc

        self.N_nn = N_train
        self.dt_nn = dt_train

        self.t_mpc = np.linspace(0, (self.N_mpc-1) * self.dt_mpc, self.N_mpc)
        # print(f"t_mpc: {self.t_mpc.shape}")
        self.t_nn = np.linspace(0, (self.N_nn-1) * self.dt_nn, self.N_nn)
        # print(f"t_nn: {self.t_nn.shape}")
        # self.t_mpc = np.arange(0, self.N_mpc * self.dt_mpc, self.dt_mpc)
        # print(f"t_mpc: {self.t_mpc}")
        # self.t_nn = np.arange(0, self.N_nn * self.dt_nn, self.dt_nn)
        # print(f"t_nn: {self.t_nn}")
        
        if M_from_x_fun is None:
            self.M_from_x = self.M_from_x_car
        else:
            self.M_from_x = M_from_x_fun

    @staticmethod
    def M_from_x_car(x):
        return np.array([x[3], x[4], x[5], x[7], x[8]])

    def interpolate_Px(self, P_x):
        assert P_x.shape[0] == self.N_mpc, "P_x shape mismatch"

        P_x_out = np.zeros((self.N_nn, P_x.shape[-1]))

        for i in range(P_x.shape[-1]):
            P_x_out[:, i] = np.interp(self.t_nn, self.t_mpc, P_x[:, i])

        return P_x_out

    def forward(self, x_now: np.ndarray, P_x: np.ndarray):
        m_now = self.M_from_x(x_now)
        P = self.interpolate_Px(P_x)
        M = self.obs_queue.forward(m_now)
        return M, P

    def params_interp(self, p_nn):
        p_mpc = np.zeros((self.N_mpc, p_nn.shape[-1]))
        for i in range(p_nn.shape[-1]):
            p_mpc[:, i] = np.interp(self.t_mpc, self.t_nn, p_nn[:, i])
        return p_mpc


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N_mpc = 50  # Adjusted to account for the +1 in the class
    dt_mpc = 0.1
    N_train = 100
    dt_train = 0.05
    M_len = 10
    M_dim = 5

    preprocessor = HyperModelPreprocesor(
        N_mpc, dt_mpc, N_train, dt_train, M_len, M_dim)

    # Adjusted to account for the +1 in the class
    t_input = np.arange(0, (N_mpc + 1) * dt_mpc, dt_mpc)
    t_input = np.expand_dims(t_input, axis=1)
    t_input = np.repeat(t_input, 2, axis=1)
    P_x = np.sin(t_input)
    P_x[:, 0] *= 2.0

    t_output = np.arange(0, N_train * dt_train, dt_train)
    P_x_interp = preprocessor.interpolate_Px(P_x)

    plt.plot(t_input, P_x[:, 0], label='Original', marker='o')
    plt.plot(t_output, P_x_interp[:, 0], label='Interpolated', marker='x')

    plt.plot(t_input, P_x[:, 1], label='Original', marker='o')
    plt.plot(t_output, P_x_interp[:, 1], label='Interpolated', marker='x')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('P_x')
    plt.title('Interpolation of P_x using sin function')
    plt.show()

    x_now = np.random.rand(9)
    M, P = preprocessor.forward(x_now, P_x)
    print(M)

    x_now = np.random.rand(9) * 100
    M, P = preprocessor.forward(x_now, P_x)
    print(M)
