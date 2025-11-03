import numpy as np
import torch
from mpc.tracks.map_reader import getTrackCustom

class CartesianToFrenet:
    def __init__(self, track_path):
        # Load track data
        self.sref, self.xref, self.yref, self.psiref, _ = getTrackCustom(track_path)
        self.psiref = np.unwrap(self.psiref)
        
    def cart2frenet(self, psi, x, y):
        idxmindist = self._findClosestPoint(x, y, self.xref, self.yref)
        idxmindist2 = self._findClosestNeighbour(x, y, self.xref, self.yref, idxmindist)

        t = self._findProjection(x, y, self.xref, self.yref, self.sref, idxmindist, idxmindist2)
        s0 = (1-t)*self.sref[idxmindist] + t*self.sref[idxmindist2]
        x0 = (1-t)*self.xref[idxmindist] + t*self.xref[idxmindist2]
        y0 = (1-t)*self.yref[idxmindist] + t*self.yref[idxmindist2]
        psi0 = (1-t)*self.psiref[idxmindist] + t*self.psiref[idxmindist2]
        
        s = s0
        n = np.cos(psi0) * (y - y0) - np.sin(psi0) * (x - x0)    
        alpha = psi - psi0
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        s = np.abs(s)
        return np.array([s, n, alpha])
    

    def _findProjection(self, x, y, xref, yref, sref, idxmindist, idxmindist2):
        vabs = abs(sref[idxmindist] - sref[idxmindist2])
        vl = np.array([xref[idxmindist2] - xref[idxmindist], yref[idxmindist2] - yref[idxmindist]])
        u = np.array([x - xref[idxmindist], y - yref[idxmindist]])
        t = np.dot(vl, u) / vabs**2
        return t

    def _findClosestPoint(self, x, y, xref, yref):
        distances = np.sqrt((xref - x)**2 + (yref - y)**2)
        idxmindist = np.argmin(distances)
        return idxmindist

    def _findClosestNeighbour(self, x, y, xref, yref, idxmindist):
        idxBefore = idxmindist - 1
        idxAfter = (idxmindist + 1) % len(xref)
        distBefore = self._dist2D(x, xref[idxBefore], y, yref[idxBefore])
        distAfter = self._dist2D(x, xref[idxAfter], y, yref[idxAfter])
        return idxBefore if distBefore < distAfter else idxAfter

    def _dist2D(self, x1, x2, y1, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    

if __name__ == "__main__":
    track_path = "tro_out_icra_v1.csv"
    cart2frenet = CartesianToFrenet(track_path)
    l_time = []
    from time import perf_counter
    
    for i in range(1000):
        x = np.random.uniform(-5.0, 5.0)
        y = np.random.uniform(-5.0, 5.0)
        psi = np.random.uniform(-np.pi, np.pi)
        t_start = perf_counter()
        frenet = cart2frenet.cart2frenet(psi, x, y)
        l_time.append(perf_counter() - t_start)
        # print(f"frenet: {frenet}")
    
    print(f"mean time: {np.mean(l_time)}")
    print(f"std time: {np.std(l_time)}")
    print(f"max time: {np.max(l_time)}")
    print(f"min time: {np.min(l_time)}")        