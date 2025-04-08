from itertools import groupby
from typing import Optional, Union
from maya import cmds
from maya.api import OpenMaya as om2, OpenMayaAnim as oma2
import numpy as np
import time
from . import DemBones, RORD


def getMFnMesh(name: str) -> om2.MFnMesh:
    """Get the MFnMesh of a given transform object extended to its shape"""
    selectionList = om2.MSelectionList()
    selectionList.add(name)
    dp = selectionList.getDagPath(0)
    dps = dp.extendToShape()
    mfm = om2.MFnMesh(dps)
    return mfm


def extractAnimation(animatedMesh: str, animFrames: list[Union[int, float]]) -> np.ndarray:
    """Get the vertex positions of the given mesh on the given frames"""
    curtime = cmds.currentTime(query=True)
    mt = om2.MTime()
    mesh = getMFnMesh(animatedMesh)
    animData = np.empty((len(animFrames), mesh.numVertices, 3))
    for i, frame in enumerate(animFrames):
        cmds.currentTime(frame)
        cmds.polyEvaluate(animatedMesh, vertex=True)  # Force update
        mt.value = float(frame)
        animData[i] = np.array(mesh.getPoints())[:, :3]

    cmds.currentTime(curtime)
    return animData


def getTimeData(animFrames: list[Union[int, float]]) -> list[float]:
    """Get the time-in-seconds value for each frame in the list"""
    fTimes = []
    mt = om2.MTime()
    for frame in animFrames:
        mt.value = float(frame)
        fTimes.append(mt.asUnits(mt.kSeconds))
    return fTimes


def extractRest(restMesh: str) -> np.ndarray:
    """Get the rest point positions"""
    mesh = getMFnMesh(restMesh)
    return np.array(mesh.getPoints())[:, :3]


def toRanges(idxs: list[Union[int, float]]) -> list[tuple[int, int]]:
    """Convert a flat list of numbers to list of inclusive ranges"""
    ret = []
    grps = groupby(enumerate(idxs), key=lambda x: x[1] - x[0])
    for _, grp in grps:
        g = list(grp)
        start = g[0][1]
        end = g[-1][1]
        ret.append((start, end))
    return ret


def toMelRange(rng: tuple[int, int]) -> str:
    """Turn the inclusive range into a mel-compatible range"""
    if rng[0] == rng[1]:
        return str(rng[0])
    return f"{rng[0]}:{rng[1]}"


def getSkinCluster(skinnedMesh: str) -> str:
    """Get the skincluster from the history of the given mesh transform"""
    hist: list[str] = cmds.listHistory(skinnedMesh)
    cl = cmds.ls(hist, type="skinCluster")
    if not cl:
        raise ValueError(f"No skin cluster found on object: {skinnedMesh}")
    return cl[0]


def getSkinWeights(cl: str) -> dict[int, dict[int, float]]:
    """Get the skin weights as a dictionary"""
    weights = {}
    ranges = toRanges(cmds.getAttr(f"{cl}.weightList", multiIndices=True))
    for rng in ranges:
        mr = toMelRange(rng)
        vals = cmds.getAttr("{}.weightList[{}].weights".format(cl, mr))
        boneIdxs = cmds.getAttr("{}.weightList[{}].weights".format(cl, mr), multiIndices=True)

        p = 0
        for i, vIdx in enumerate(range(rng[0], rng[1] + 1)):
            wvals = vals[i]
            newp = p + len(wvals)
            weights[vIdx] = dict(zip(boneIdxs[p:newp], wvals))
            p = newp
    return weights


def getBindMats(cl: str, joints: list[str]) -> np.ndarray:
    """Get the bind matrices for the given joints"""
    bindMats = np.zeros((len(joints), 4, 4))
    for i in range(len(joints)):
        bpm = cmds.getAttr(f"{cl}.bindPreMatrix[{i}]")
        if not bpm:
            bpm = np.eye(4)
        else:
            bpm = np.array(bpm).reshape((4, 4))
        bindMats[i] = bpm
    bindMats = np.linalg.inv(bindMats)
    return bindMats


def getInfluenceHierarchy(cl: str) -> tuple[list[str], list[Union[int, float]]]:
    """Get the DemBones compatible hierarchy of the joints in the given skincluster"""
    joints: Optional[list[str]] = cmds.skinCluster(cl, query=True, influence=True)
    if not joints:
        raise ValueError("Skinned target must have influences")

    joints = cmds.ls(joints, long=True)

    jset = set(joints)
    extras = []
    for joint in joints:
        sp = joint.split("|")
        for i in range(2, len(sp) + 1):
            chk = "|".join(sp[:i])
            if chk not in jset:
                extras.append(chk)
                jset.add(chk)

    joints = joints + extras
    pars = []
    for j in joints:
        par = cmds.ls(cmds.listRelatives(j, parent=True), long=True)
        if not par:
            pars.append(-1)
            continue
        par = cmds.ls(par[0], long=True)[0]
        pars.append(joints.index(par))
    return joints, pars


def getRotateOrders(joints: list[str]) -> list[RORD]:
    """Get the rotate orders for the list of joints"""
    roIdxs = ["xyz", "yzx", "zxy", "xzy", "yxz", "zyx"]
    return [roIdxs[cmds.getAttr(f"{j}.rotateOrder")] for j in joints]


def getParentInverseMatrces(joints: list[str]) -> np.ndarray:
    """Get the ParentInverseMatrices for the list of joints"""
    pims = [cmds.getAttr(f"{j}.parentInverseMatrix[0]") for j in joints]
    return np.array(pims).reshape((-1, 4, 4))


def getJointOrients(joints: list[str]) -> list[tuple[float, float, float]]:
    """Get the joint orient values for the list of joints"""
    jos = []
    for j in joints:
        v = (0.0, 0.0, 0.0)
        if cmds.nodeType(j) == "joint":
            v = cmds.getAttr(f"{j}.jointOrient")[0]
        jos.append(v)
    return jos


def getTopo(meshName: str) -> list[list[Union[int, float]]]:
    """Get the demBones compatible topology data for the given mesh"""
    mesh = getMFnMesh(meshName)
    counts, connects = mesh.getVertices()
    connects = list(connects)
    ret = []
    p = 0
    for c in counts:
        ret.append(connects[p : p + c])
        p += c
    return ret


def applyBoneTransforms(solver: DemBones, animFrames: list[Union[int, float]]):
    """Apply transforms to the bones stored by the solver"""
    bones = solver.boneName
    allmats = solver.bind[None] @ solver.boneMats @ solver.preMulInv[None]

    curtime = cmds.currentTime(query=True)
    for i, f in enumerate(animFrames):
        cmds.currentTime(f)
        for j, bone in enumerate(bones):
            cmds.xform(bone, matrix=allmats[i, j].flatten(), worldSpace=False)
        cmds.setKeyframe(bones)
    cmds.currentTime(curtime)


def applyWeights(solver: DemBones, skinCls: str, skinMesh: str):
    """Set the weights stored by the solver to the given skincluster"""
    ww = solver.weights
    weightArray = np.zeros((len(solver.restPose), len(solver.boneName)))
    for bidx, tt in ww.items():
        for vidx, val in tt.items():
            weightArray[vidx, bidx] = val

    bones = solver.boneName

    sel = om2.MSelectionList()
    sel.add(skinCls)
    skin_cluster_obj = sel.getDependNode(0)
    skin_cluster_fn = oma2.MFnSkinCluster(skin_cluster_obj)

    sel = om2.MSelectionList()
    sel.add(skinMesh)
    mesh_dag = sel.getDagPath(0)
    mesh_dag.extendToShape()

    skin_cluster_fn.setWeights(
        mesh_dag, om2.MObject(), om2.MIntArray(range(len(bones))), om2.MDoubleArray(weightArray.flatten())
    )


def applySolution(solver: DemBones, animFrames: list[Union[int, float]], skinCls: str, skinMesh: str):
    """Apply the solution that a solver has computed to animated joints
    a skincluster, and a mesh

    Arguments:
        solver (DemBones): A dembones solver that has been computed
        animFrames (list): The frames to set the keys on
        skinCls (str): The name of the skincluster node
        skinMesh (str): The name of the skinned mesh transform
    """
    applyBoneTransforms(solver, animFrames)
    applyWeights(solver, skinCls, skinMesh)


def mayaDemBones(
    restObj: str,
    restFrame: int,
    animObj: str,
    animFrames: list[Union[int, float]],
    skinnedTarget: str,
    solver: Optional[DemBones] = None,
    verbose: bool = False,
    **kwargs,
) -> DemBones:
    """Run dembones on a maya object

    Still TODO: Add mechanisms for locking weights or transforms

    Arguments:
        restObj: The rest-pose object
        restFrame: The frame to extract the restpose on
        animObj: The object with vertex animation on it
        animFrames: A list of frame numbers that have animation
        skinnedTarget: The pre-bound target object. Animation will
            be added directly to the bones of this object
    """
    if solver is None:
        solver = DemBones()

    cmds.currentTime(restFrame)

    # All done at rest-frame time
    topo = getTopo(restObj)
    cl = getSkinCluster(skinnedTarget)
    weights = getSkinWeights(cl)
    joints, parIdxs = getInfluenceHierarchy(cl)
    rotateOrders = getRotateOrders(joints)
    parInvMats = getParentInverseMatrces(joints)
    jointOrients = getJointOrients(joints)

    restPts = extractRest(restObj)
    bind = getBindMats(cl, joints)
    fTime = getTimeData(animFrames)
    anim = extractAnimation(animObj, animFrames)

    solver.restPose = restPts
    solver.anim = anim
    solver.fv = topo
    solver.fTime = fTime
    solver.boneName = joints
    solver.parent = parIdxs
    solver.weights = weights
    solver.lockW = np.zeros(len(restPts))  # leave it unlocked
    solver.rotOrder = rotateOrders
    solver.orient = jointOrients
    solver.bind = bind
    solver.preMulInv = parInvMats
    solver.verbose = verbose

    for k, v in kwargs.items():
        setattr(solver, k, v)

    start = time.time()
    solver.compute()
    end = time.time()
    if verbose:
        print(f"Solve took: {end - start}s")

    applySolution(solver, animFrames, cl, skinnedTarget)
    return solver


def test():
    _solver = mayaDemBones(
        "face_skinned",
        1001,
        "face_shapes",
        list(range(1001, 1053)),
        "face_skinned",
        patience=1000,
        num_iterations=300,
        num_transform_iterations=50,
        num_weight_iterations=30,
        max_influences=2,
        weights_smooth=1e-8,
    )


if __name__ == "__main__":
    test()
