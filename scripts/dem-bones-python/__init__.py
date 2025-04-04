import sys
sys.path.insert(0, r'C:\blur\dev\GitHub\InProgress\dem-bones-python\output_Python')
import _dem_bones_core
import numpy as np
__version__ = "v0.0.1-dev"


class DemBones(_dem_bones_core.DemBones):
    """A Python Wrapper for the dem-bones solver from EA

    To use this class, make an instance, and then set at least the
    `restPose`, `anim`, and `boneName` properties.
    Then call `.compute()`
    The `weights`, `boneMats`, and `outputTransforms` properties will
    then be updated.

    To use pre-defined weights, make sure to set the lockW property to
    `dbsolver.lockW = np.ones(len(dbsolver.restPose))`

    To solve weights for given bone animation, make sure to set the
    lockM property to
    `dbsolver.lockM = np.ones(len(dbsolver.boneName), dtype=int)`

    NOTE: Getting and setting data through this interface always
    involves a copy of the data. This may change in the future,
    but be aware of this for performance reasons

    Arguments:
        rowMajorMats (bool): Accept and return all matrices in a
            row-major format. This means the first 3 rows of the
            matrix will be the x, y, and z basis vectors, and the
            last row of the matrix will have the translation values.
            If False, then the columns will have the basis vectors.
            Defaults to True
        rowMajorPts (bool): Accept and return all point and vector
            data in a row-major format. This (generally) means that
            the last dimension of the numpy array will have length 3
            If False, then the next-to-last dimension will have length
            3 instead.
            Defaults to True

    Properties
        tolerance(float):
            If the solver fails to converge faster than `tolerance`
            for `patience` iterations, bail out. default = 1e-3

        patience(int):
            If the solver fails to converge faster than `tolerance`
            for `patience` iterations, bail out. default = 3

        nIters(int):
            Number of global iterations, default = 30

        nTransIters(int):
            Number of bone transformations update iterations per global iteration
            default = 5

        transAffine(float):
            Translations affinity soft constraint, default = 10.0

        transAffineNorm(float):
            p-norm for bone translations affinity soft constraint, default = 4.0

        nWeightsIters(int):
            Number of weights update iterations per global iteration, default = 3

        nnz(int):
            Number of non-zero weights per vertex (like maya's "Max Influences")
            default = 8

        weightsSmooth(float):
            Weights smoothness soft constraint, default = 1e-4

        weightsSmoothStep(float):
            Step size for the weights smoothness soft constraint, default = 1.0

        weightsEps(float):
            Epsilon for weights solver, default = 1e-15

        lockW(np.array[float]):
            The lock percent of each vertex. Defaults to all zero

        lockM(np.array[Literal[0, 1]]):
            The bone transformations lock control. Defaults to all zero

        fv(list[list[int]]):
            The mesh topology

        fTime(np.array[float]):
            The timestamps per-frame. Defaults to all zero

        boneName(list[str]):
            Required. The names of the bones. The names are unimportant, only the length
            is used.

        parent(list[int]):
            The indices of the parents of each bone, or -1 if that bone has no parent
            Defaults to all -1
    """

    def __init__(self, rowMajorMats=True, rowMajorPts=True):
        super(DemBones, self).__init__()
        self.rowMajorMats = rowMajorMats
        self.rowMajorPts = rowMajorPts

    @property
    def restPose(self):
        """Required. The rest vertex positions numpy array as:
        if self.rowMajorPts:
            array[vertIdx][component]
        else:
            array[component][vertIdx]
        """
        ret = self._u
        if self.rowMajorPts:
            ret = ret.T
        return ret

    @restPose.setter
    def restPose(self, pose):
        if self.rowMajorPts:
            pose = pose.T
        self._u = pose

    @property
    def anim(self):
        """Required. The vertex animation numpy array as:
        if self.rowMajorPts:
            array[frame][vertIdx][component]
        else:
            array[frame][component][vertIdx]
        """
        ret = self._v
        ret = ret.reshape((-1, 3, ret.shape[-1]))
        if self.rowMajorPts:
            ret = ret.swapaxes(-1, -2)
        return ret

    @anim.setter
    def anim(self, pose):
        if self.rowMajorPts:
            pose = pose.swapaxes(-1, -2)
        self._v = pose.reshape((-1, pose.shape[-1]))

    @property
    def rotOrder(self):
        """The rotation order list stored as ["xyz", "yzx", ...]
        Defaults to all ["xyz"]
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
        Defaults to all identity matrices
        """
        b = self._bind.T.reshape((-1, 4, 4))
        if not self.rowMajorMats:
            b = b.swapaxes(1, 2)
        return b

    @bind.setter
    def bind(self, pose):
        if not self.rowMajorMats:
            pose = pose.swapaxes(1, 2)
        self._bind = pose.reshape((-1, 4)).T

    @property
    def preMulInv(self):
        """The bind-pose matrices as a numpy array
        array[boneIdx] = 4*4 matrix
        Defaults to all identity matrices
        """
        b = self._preMulInv.T.reshape((-1, 4, 4))
        if not self.rowMajorMats:
            b = b.swapaxes(1, 2)
        return b

    @preMulInv.setter
    def preMulInv(self, pose):
        if not self.rowMajorMats:
            pose = pose.swapaxes(1, 2)
        self._preMulInv = pose.reshape((-1, 4)).T

    @property
    def orient(self):
        ret = self._orient
        if self.rowMajorPts:
            ret = ret.T
        return ret

    @orient.setter
    def orient(self, val):
        """The maya-style joint orient for each bone
        if self.rowMajorPts:
            array[bone][component]
        else:
            array[component][bone]
        Defaults to all zeros
        """
        if self.rowMajorPts:
            val = val.T
        self._orient = val

    def compute(self):
        """Compute the bone positions and new weights"""
        super(_dem_bones_core.DemBones, self).compute()

    @property
    def boneMats(self):
        """The bone animation data
        array[frame][boneIdx] = 4*4 matrix
        Defaults to all identity matrices
        """
        m = self._m
        numFrames = m.shape[0] // 4
        numBones = m.shape[1] // 4
        m = m.reshape((numFrames, 4, numBones, 4))
        m = m.swapaxes(1, 2)
        if self.rowMajorMats:
            m = m.swapaxes(2, 3)
        return m

    @boneMats.setter
    def boneMats(self, m):
        if self.rowMajorMats:
            m = m.swapaxes(2, 3)
        m = m.swapaxes(1, 2)
        m = m.reshape(m.shape[0] * 4, m.shape[2] * 4)
        self._m = m

    @property
    def outputTransforms(self):
        """Get the extended output transform data
        if self.rowMajorPts:
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
        if self.rowMajorPts:
            rot = rot.swapaxes(1, 2)
            tran = tran.swapaxes(1, 2)
            bindRot = bindRot.T
            bindTran = bindTran.T

        if not self.rowMajorMats:
            bindMats = bindMats.swapaxes(1, 2)

        return {
            "rot": rot,
            "tran": tran,
            "bindRot": bindRot,
            "bindTran": bindTran,
            "bindMats": bindMats,
        }
