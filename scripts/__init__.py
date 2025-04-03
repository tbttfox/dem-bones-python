import _dem_bones_core as _core
import numpy as np


class DemBones(_core.DemBones):
    def __init__(self, row_major_mats=True, row_major_pts=True):
        super(_core.DemBones, self).__init__()
        self.row_major_mats = row_major_mats
        self.row_major_pts = row_major_pts

    @property
    def rest_pose(self):
        """The rest vertex positions numpy array as:
        if self.row_major_pts:
            array[vertIdx][component]
        else:
            array[component][vertIdx]
        """
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
        """The vertex animation numpy array as:
        if self.row_major_pts:
            array[frame][vertIdx][component]
        else:
            array[frame][component][vertIdx]
        """
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
        """The rotation order list stored as
        ["xyz", "yzx", ...]
        """
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
        """The per-vertex per-bone weights:
        dict[vertIdx, dict[boneIdx, weightValue]]
        """
        ret = {}
        for vi, bi, w in zip(self._wvi, self._wbi, self._wfv):
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
        """The bind-pose matrices as a numpy array
        array[boneIdx] = 4*4 matrix
        """
        b = self._bind.T.reshape((-1, 4, 4))
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
        """The bind-pose matrices as a numpy array
        array[boneIdx] = 4*4 matrix
        """
        b = self._preMulInv.T.reshape((-1, 4, 4))
        if not self.row_major_mats:
            b = b.swapaxes(1, 2)
        return b

    @preMulInv.setter
    def preMulInv(self, pose):
        if not self.row_major_mats:
            pose = pose.swapaxes(1, 2)
        self._preMulInv = pose.reshape((-1, 4)).T

    def compute(self):
        """Compute the bone positions and new weights"""
        super(_core.DemBones, self).compute()

    @property
    def boneMats(self):
        """The bone animation data
        array[frame][boneIdx] = 4*4 matrix
        """
        m = self._m
        numFrames = m.shape[0] // 4
        numBones = m.shape[1] // 4
        m = m.reshape((numFrames, 4, numBones, 4))
        m = m.swapaxes(1, 2)
        if self.row_major_mats:
            m = m.swapaxes(2, 3)
        return m

    @boneMats.setter
    def boneMats(self, m):
        if self.row_major_mats:
            m = m.swapaxes(2, 3)
        m = m.swapaxes(1, 2)
        m = m.reshape(m.shape[0] * 4, m.shape[2] * 4)
        self._m = m

    def getOutputTransforms(self):
        """Get the extended output transform data
        if self.row_major_pts:
            return {
                "rot": array[frame][boneIndex][component],
                "tran": array[frame][boneIndex][component],
                "bindMats": array[boneIndex] = 4*4 matrix
                "bindRot": array[boneIndex][component]
                "bindTran": array[boneIndex][component]
            }
        else:
            return {
                "rot": array[frame][component][boneIndex],
                "tran": array[frame][component][boneIndex],
                "bindMats": array[boneIndex] = 4*4 matrix
                "bindRot": array[component][boneIndex]
                "bindTran": array[component][boneIndex]
            }
        """
        rot = self._lr
        tran = self._lt
        bindRot = self._lbr
        bindTran = self._lbt
        bindMats = self._gb.T.reshape((-1, 4, 4))

        rot = rot.reshape((-1, 3, rot.shape[-1]))
        tran = tran.reshape((-1, 3, tran.shape[-1]))
        if self.row_major_pts:
            rot = rot.swapaxes(1, 2)
            tran = tran.swapaxes(1, 2)
            bindRot = bindRot.T
            bindTran = bindTran.T

        if not self.row_major_mats:
            bindMats = bindMats.swapaxes(1, 2)

        return {
            "rot": rot,
            "tran": tran,
            "bindRot": bindRot,
            "bindTran": bindTran,
            "bindMats": bindMats,
        }
