import _dem_bones_core as _core
import numpy as np


class DemBones(_core.DemBones):
    def __init__(self, row_major_mats=True, row_major_pts=True):
        super(_core.DemBones, self).__init__()
        self.row_major_mats = row_major_mats
        self.row_major_pts = row_major_pts

    @property
    def rest_pose(self):
        ret = self._u
        if self.row_major_pts:
            ret = ret.T
        return ret

    @rest_pose.setter
    def rest_pose(self, pose):
        if self.row_major_pts:
            pose = pose.T
        self._u = pose

    @property
    def anim(self):
        ret = self._v
        ret = ret.reshape((-1, 3, ret.shape[-1]))
        if self.row_major_pts:
            ret = ret.swapaxes(-1, -2)
        return ret

    @anim.setter
    def anim(self, pose):
        if self.row_major_pts:
            pose = pose.swapaxes(-1, -2)
        self._v = pose.reshape((-1, pose.shape[-1]))

    @property
    def rotOrder(self):
        ary = self._rotOrder.T.flatten().tolist()
        axes = "xyz"
        ret = "".join(axes[r] for r in ary)
        return [ret[i : i + 3] for i in range(0, len(ret), 3)]

    @rotOrder.setter
    def rotOrder(self, pose):
        pose = "".join(pose).lower()
        off = ord("x")
        pose = np.array([ord(p) - off for p in pose], dtype="i1")
        self._rotOrder = pose.reshape((-1, 3)).T

    @property
    def weights(self):
        wvi = self._wvi
        wbi = self._wbi
        wfv = self._wfv

        ret = {}
        for vi, bi, w in zip(wvi, wbi, wfv):
            ret.setdefault(vi, {})[bi] = w
        return ret

    @weights.setter
    def weights(self, pose):
        wvi, wbi, wfv = [], [], []
        for vi, bdict in pose.items():
            for bi, w in bdict.items():
                wvi.append(vi)
                wbi.append(bi)
                wfv.append(w)

        self._wvi = np.array(wvi, dtype=int)
        self._wbi = np.array(wbi, dtype=int)
        self._wfv = np.array(wfv, dtype=np.float64)

    @property
    def bind(self):
        b = self._bind
        b = b.T.reshape((-1, 4, 4))
        if not self.row_major_mats:
            b = b.swapaxes(1, 2)
        return b

    @bind.setter
    def bind(self, pose):
        if not self.row_major_mats:
            pose = pose.swapaxes(1, 2)
        self._bind = pose.reshape((-1, 4)).T

    @property
    def preMulInv(self):
        b = self._preMulInv
        b = b.T.reshape((-1, 4, 4))
        if not self.row_major_mats:
            b = b.swapaxes(1, 2)
        return b

    @preMulInv.setter
    def preMulInv(self, pose):
        if not self.row_major_mats:
            pose = pose.swapaxes(1, 2)
        self._preMulInv = pose.reshape((-1, 4)).T

    def compute(self):
        super(_core.DemBones, self).compute()

        m = self._m
        numFrames = m.shape[0] // 4
        numBones = m.shape[1] // 4
        m = m.reshape((numFrames, 4, numBones, 4))
        m = m.swapaxes(1, 2)
        if self.row_major_mats:
            m = m.swapaxes(2, 3)
        return m
