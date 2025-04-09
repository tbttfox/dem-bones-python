from . import _dem_bones_core
import numpy as np
from numpy import typing as npt
from typing import Literal

__version__ = "v0.0.1-dev"


RORD = Literal["xyz", "yzx", "zxy", "xzy", "yxz", "zyx"]


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
        self.rowMajorMats: bool = rowMajorMats
        self.rowMajorPts: bool = rowMajorPts

    @property
    def restPose(self) -> np.ndarray:
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
    def restPose(self, pose: npt.ArrayLike):
        pose = np.asarray(pose)
        if self.rowMajorPts:
            pose = pose.T
        self._u = pose

    @property
    def anim(self) -> np.ndarray:
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
    def anim(self, pose: npt.ArrayLike):
        pose = np.asarray(pose)
        if self.rowMajorPts:
            pose = pose.swapaxes(-1, -2)
        self._v = pose.reshape((-1, pose.shape[-1]))

    @property
    def rotOrder(self) -> list[RORD]:
        """The rotation order list stored as ["xyz", "yzx", ...]
        Defaults to all ["xyz"]
        """
        dd: dict[tuple, RORD] = {
            (0, 1, 2): "xyz",
            (1, 2, 0): "yzx",
            (2, 0, 1): "zxy",
            (0, 2, 1): "xzy",
            (1, 0, 2): "yxz",
            (2, 1, 0): "zyx",
        }
        return [dd[tuple(row)] for row in self._rotOrder.T]

    @rotOrder.setter
    def rotOrder(self, pose: list[RORD]):
        flat = "".join(pose).lower()
        off = ord("x")
        ary = np.array([ord(p) - off for p in flat], dtype="i1")
        self._rotOrder = ary.reshape((-1, 3)).T

    @property
    def weights(self) -> dict[int, dict[int, float]]:
        """The per-vertex per-bone weights:
        dict[vertIdx, dict[boneIdx, weightValue]]
        """
        ret = {}
        for vi, bi, w in zip(self._wvi, self._wbi, self._wfv):
            ret.setdefault(vi, {})[bi] = w
        return ret

    @weights.setter
    def weights(self, pose: dict[int, dict[int, float]]):
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
    def bind(self) -> np.ndarray:
        """The bind-pose matrices as a numpy array
        array[boneIdx] = 4*4 matrix
        Defaults to all identity matrices
        """
        b = self._bind.T.reshape((-1, 4, 4))
        if not self.rowMajorMats:
            b = b.swapaxes(1, 2)
        return b

    @bind.setter
    def bind(self, pose: npt.ArrayLike):
        pose = np.asarray(pose)
        if not self.rowMajorMats:
            pose = pose.swapaxes(1, 2)
        self._bind = pose.reshape((-1, 4)).T

    @property
    def preMulInv(self) -> np.ndarray:
        """The bind-pose matrices as a numpy array
        array[boneIdx] = 4*4 matrix
        Defaults to all identity matrices
        """
        b = self._preMulInv.T.reshape((-1, 4, 4))
        if not self.rowMajorMats:
            b = b.swapaxes(1, 2)
        return b

    @preMulInv.setter
    def preMulInv(self, pose: npt.ArrayLike):
        pose = np.asarray(pose)
        if not self.rowMajorMats:
            pose = pose.swapaxes(1, 2)
        self._preMulInv = pose.reshape((-1, 4)).T

    @property
    def orient(self) -> np.ndarray:
        ret = self._orient
        if self.rowMajorPts:
            ret = ret.T
        return ret

    @orient.setter
    def orient(self, val: npt.ArrayLike):
        """The maya-style joint orient for each bone
        if self.rowMajorPts:
            array[bone][component]
        else:
            array[component][bone]
        Defaults to all zeros
        """
        val = np.asarray(val)
        if self.rowMajorPts:
            val = val.T
        self._orient = val

    def compute(self):
        """Compute the bone positions and new weights"""
        super(DemBones, self).compute()

    @property
    def boneMats(self) -> np.ndarray:
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
    def boneMats(self, m: npt.ArrayLike):
        m = np.asarray(m)
        if self.rowMajorMats:
            m = m.swapaxes(2, 3)
        m = m.swapaxes(1, 2)
        m = m.reshape(m.shape[0] * 4, m.shape[2] * 4)
        self._m = m

    @property
    def outputTransforms(self) -> dict[str, np.ndarray]:
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
