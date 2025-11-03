import numpy as np
import pathlib
import scipy
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd


class TrackReader:

    def __init__(self, filename, flip=False, reverse=False, corrected=True, loopback=True):

        self.file = pathlib.Path(filename)
        data = np.loadtxt(self.file, delimiter=",", skiprows=1)

        if not reverse:
            data = np.flip(data, axis=0)

        # spline smoothing factor in scipy.interpolate.splprep
        self.splprep_s = 2.0
        self.corrected = corrected
        self.loopback = loopback

        self.x = data[:, 0]
        self.y = data[:, 1]
        self.w_r = data[:, 2]
        self.w_l = data[:, 3]

        self.debug_plot = True
        self.points_per_meter = 5  # FIXME to prams

        self.preprocess_track()

    def preprocess_track(self):

        x, y, w_r, w_l = self.x, self.y, self.w_r, self.w_l

        # smooth x and y with savgol filter
        w_r = scipy.signal.savgol_filter(w_r, 10, 3)

        track_width = w_r + w_l
        self.track_width = track_width

        w_splprep = 1 / track_width

        if self.loopback:
            tck, u = scipy.interpolate.splprep(
                [x, y], w_splprep, s=self.splprep_s, k=5, per=len(x)
            )
        else:
            tck, u = scipy.interpolate.splprep(
                [x, y], w_splprep, s=self.splprep_s, k=5
            )    
        
        self.tck = tck

        x_s, y_s = scipy.interpolate.splev(u, tck)
        self.x_s = x_s
        self.y_s = y_s

        e = np.sqrt((x_s - x) ** 2 + (y_s - y) ** 2)
        
        if self.corrected:
            track_width_corrected = (track_width / 2 - e) * 2
        else:
            track_width_corrected = track_width

        rmse = np.sqrt(np.mean((x_s - x) ** 2 + (y_s - y) ** 2))
        self.rmse = rmse

        x_dot, y_dot = scipy.interpolate.splev(u, tck, der=1)
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.u = u

        x_ddot, y_ddot = scipy.interpolate.splev(u, tck, der=2)
        self.x_ddot = x_ddot
        self.y_ddot = y_ddot

        curvature = (x_dot * y_ddot - y_dot * x_ddot) / \
            (x_dot**2 + y_dot**2) ** (3 / 2)

        self.curvature = curvature

        path_s = scipy.integrate.cumulative_trapezoid(
            np.sqrt(x_dot**2 + y_dot**2), u, initial=0)
        self.s = path_s

        # reinterpolation
        track_lenght = path_s[-1]
        self.track_lenght = track_lenght

        self.re_N = int(track_lenght * self.points_per_meter)
        self.re_path_s = np.linspace(0, track_lenght, self.re_N)
        self.re_u = self.re_path_s / track_lenght
        self.re_x, self.re_y = scipy.interpolate.splev(self.re_u, self.tck)
        self.re_x_dot, self.re_y_dot = scipy.interpolate.splev(
            self.re_u, self.tck, der=1)
        self.re_x_ddot, self.re_y_ddot = scipy.interpolate.splev(
            self.re_u, self.tck, der=2)
        self.re_curvature = (self.re_x_dot * self.re_y_ddot - self.re_y_dot *
                             self.re_x_ddot) / (self.re_x_dot**2 + self.re_y_dot**2) ** (3 / 2)
        self.re_track_width = np.interp(
            self.re_path_s, self.s, self.track_width)
        self.re_track_width_corrected = np.interp(
            self.re_path_s, self.s, track_width_corrected)
        self.re_heading = np.arctan2(self.re_y_dot, self.re_x_dot)

    def plot_track(self):

        print(f"Track length: {self.track_lenght} m")
        print(f"{len(self.x)} points")
        plt.rcParams["figure.figsize"] = (16, 8)
        plt.figure()
        plt.arrow(self.x_s[0], self.y_s[0], self.x_s[1] - self.x_s[0],
                  self.y_s[1] - self.y_s[0], head_width=2, alpha=0.20, color='orange')

        if self.debug_plot:
            plt.scatter(self.x, self.y, label='org_track')

        plt.plot(self.re_x, self.re_y, 'r', label='re_track')
        plt.title(f'Track map {self.file.name}', fontsize=20)

        if self.debug_plot:
            for i in range(len(self.x)):
                circle = plt.Circle(
                    (self.x[i], self.y[i]), self.track_width[i] / 2, color='b', fill=False, alpha=0.15)
                plt.gcf().gca().add_artist(circle)

        for i in range(0, len(self.re_x)):
            circle_2 = plt.Circle(
                (self.re_x[i], self.re_y[i]), self.re_track_width_corrected[i] / 2, color='r', fill=False, alpha=0.15)
            plt.gcf().gca().add_artist(circle_2)
            
            # print(f"re_x: {self.re_x[i]}, re_y: {self.re_y[i]}, "
            #       f"re_track_width_corrected: {self.re_track_width_corrected[i]}")
            # arrow_from_track_to_baund = 0.5 * self.re_track_width_corrected[i] * \
            #     np.array([-self.re_y_dot[i], self.re_x_dot[i]])
            # plt.arrow(self.re_x[i], self.re_y[i],
            #           arrow_from_track_to_baund[0], arrow_from_track_to_baund[1],
            #           head_width=0.2, alpha=0.20, color='orange')

        for i in range(0, len(self.x_s) - 1, 60):
            plt.text(self.x_s[i] + 0.25, self.y_s[i],
                     f"{self.s[i]:.0f} ", fontsize=12)

        plt.grid()
        plt.axis('equal')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.tight_layout()
        plt.legend()

    def to_Cartesian(self, s: np.ndarray, n: np.ndarray, mu: np.ndarray = None):
        # TODO check s > track.lenght
        s = np.where(s > self.track_lenght, s - self.track_lenght, s)
        u = s / self.track_lenght
        x, y = scipy.interpolate.splev(u, self.tck)
        x_dot, y_dot = scipy.interpolate.splev(u, self.tck, der=1)
        m = np.hypot(x_dot, y_dot)
        N = np.array([-y_dot, x_dot]) / m
        x_out = x + n * N[0]
        y_out = y + n * N[1]
        yaw = np.arctan2(y_dot, x_dot) + mu
        return x_out, y_out, yaw

    def plot_points(self, s: np.ndarray, n: np.ndarray,
                    v_x=None, marker="o", alpha=1.0):
        x_out, y_out, yaw = self.to_Cartesian(s, n, np.zeros_like(s))
        if v_x is not None:
            plt.scatter(x_out, y_out, c=v_x, cmap="jet",
                        marker=marker, alpha=alpha)
            plt.colorbar().set_label("v_x [m/s]")
        else:
            plt.scatter(x_out, y_out, color="r", marker=marker, alpha=alpha)
            
    def plot_points_with_heading(self, s: np.ndarray, n: np.ndarray, mu: np.ndarray, vx=None):
        x_out, y_out, yaw = self.to_Cartesian(s, n, mu)
        if vx is not None:
            plt.scatter(x_out, y_out, c=vx, cmap="jet")
            plt.colorbar().set_label("v_x [m/s]")
        else:
            plt.scatter(x_out, y_out, color="r")

        arrow_length = 0.1  # Length of the arrow
        for i in range(len(x_out)):
            plt.arrow(x_out[i], y_out[i], arrow_length * np.cos(yaw[i]),
                      arrow_length * np.sin(yaw[i]),
                      head_width=0.2, alpha=0.20, color='orange')

    def plot_points_cartesian(self, x, y, v_x=None):
        if v_x is not None:
            plt.scatter(x, y, c=v_x, cmap="jet")
            plt.colorbar().set_label("v_x [m/s]")
        else:
            plt.scatter(x, y, color="r")

    def length(self):
        return self.s[-1]

    def file_path(self) -> pathlib.Path:
        return self.file

    def track_name(self) -> str:
        return self.file.stem

    def plot_curvature(self):
        plt.figure()
        plt.plot(self.re_path_s, self.re_curvature, label="curvature")
        plt.plot(self.re_path_s, self.re_track_width_corrected, label="width")
        plt.xlabel("s [m]")
        plt.ylabel("curvature [1/m]")
        plt.title("Curvature")
        plt.grid()
        plt.legend()
        plt.tight_layout()

    def save_track(self, filename):
        df = pd.DataFrame(
            {
                "s": self.re_path_s,
                "x": self.re_x,
                "y": self.re_y,
                "heading": self.re_heading,
                "curvature": self.re_curvature,
                "track_width": self.re_track_width_corrected
            }
        )
        df.to_csv(filename, index=False)


if __name__ == "__main__":

    track_path = pathlib.Path(__file__).parent / "track_bounds_accurate_round_abudhabi.csv"

    track = TrackReader(track_path)
    print(f"RMSE {track.rmse}")

    track.plot_curvature()
    track.plot_track()

    # example plot track data
    s = np.linspace(0, track.length(), 1000)
    n = np.ones_like(s) * 0.05
    mu  = np.zeros_like(s)
    track.plot_points(s, n)
    track_name = track.track_name()
    track_file_name = pathlib.Path(__file__).parent / ("prep_" + track_name + ".csv")

    plt.show()

    track.save_track(track_file_name)
    print("Track saved")
