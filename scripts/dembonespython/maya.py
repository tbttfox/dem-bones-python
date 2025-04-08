from itertools import groupby
from typing import Optional
from maya import cmds
from maya.api import OpenMaya as om2, OpenMayaAnim as oma2
import numpy as np
from . import DemBones


def getMesh(name: str):
    selectionList = om2.MSelectionList()
    selectionList.add(name)
    dp = selectionList.getDagPath(0)
    dps = dp.extendToShape()
    mfm = om2.MFnMesh(dps)
    return mfm


def extractAnimation(animatedMesh: str, animFrames: list):
    curtime = cmds.currentTime(query=True)
    fTimes = []  # The time in seconds
    mt = om2.MTime()
    mesh = getMesh(animatedMesh)
    animData = np.empty((len(animFrames), mesh.numVertices, 3))
    for i, frame in enumerate(animFrames):
        cmds.currentTime(frame)
        cmds.polyEvaluate(animatedMesh, vertex=True)  # Force update
        mt.value = float(frame)
        fTimes.append(mt.asUnits(mt.kSeconds))
        animData[i] = np.array(mesh.getPoints())[:, :3]

    cmds.currentTime(curtime)
    return animData, fTimes


def extractRest(restMesh: str):
    mesh = getMesh(restMesh)
    return np.array(mesh.getPoints())[:, :3]


def toRanges(idxs):
    ret = []
    grps = groupby(enumerate(idxs), key=lambda x: x[1] - x[0])
    for _, grp in grps:
        g = list(grp)
        start = g[0][1]
        end = g[-1][1]
        ret.append((start, end))
    return ret


def toMelRange(rng):
    if rng[0] == rng[1]:
        return str(rng[0])
    return f"{rng[0]}:{rng[1]}"


def getSkinCluster(skinnedMesh: str):
    hist: list[str] = cmds.listHistory(skinnedMesh)
    cl = cmds.ls(hist, type="skinCluster")
    if not cl:
        raise ValueError(f"No skin cluster found on object: {skinnedMesh}")
    return cl[0]


def getSkinWeights(cl):
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


def getBindMats(cl, joints):
    bindMats = np.zeros((len(joints), 4, 4))
    for i, j in enumerate(joints):
        bpm = cmds.getAttr(f"{cl}.bindPreMatrix[{i}]")
        if not bpm:
            bpm = np.eye(4)
        else:
            bpm = np.array(bpm).reshape((4, 4))
        bindMats[i] = bpm
    bindMats = np.linalg.inv(bindMats)
    return bindMats


def getInfluenceHierarchy(cl):
    joints: list[str] = cmds.skinCluster(cl, query=True, influence=True)
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
    for i, j in enumerate(joints):
        par = cmds.ls(cmds.listRelatives(j, parent=True), long=True)
        if not par:
            pars.append(-1)
            continue
        par = cmds.ls(par[0], long=True)[0]
        pars.append(joints.index(par))
    return joints, pars


def getJointData(joints):
    roIdxs = ["xyz", "yzx", "zxy", "xzy", "yxz", "zyx"]
    ros = []
    jos = []
    pims = []
    for j in joints:
        ros.append(roIdxs[cmds.getAttr(f"{j}.rotateOrder")])
        if cmds.nodeType(j) == "joint":
            jos.extend(cmds.getAttr(f"{j}.jointOrient"))
        else:
            jos.append((0.0, 0.0, 0.0))

        pims.append(cmds.getAttr(f"{j}.parentInverseMatrix[0]"))

    pims = np.array(pims).reshape((-1, 4, 4))
    return ros, jos, pims


def getTopo(meshName: str):
    mesh = getMesh(meshName)
    counts, connects = mesh.getVertices()
    connects = list(connects)
    ret = []
    p = 0
    for c in counts:
        ret.append(connects[p : p + c])
        p += c
    return ret


def applySolution(solver: DemBones, animFrames: list, skinCls: str, skinMesh: str):
    """Apply the solution that a solver has computed to animated joints
    a skincluster, and a mesh

    Arguments:
        solver (DemBones): A dembones solver that has been computed
        animFrames (list): The frames to set the keys on
        skinCls (str): The name of the skincluster node
        skinMesh (str): The name of the skinned mesh transform
    """
    anim = solver.boneMats
    bones = solver.boneName
    bind = solver.bind
    pmi = solver.preMulInv

    for i, f in enumerate(animFrames):
        cmds.currentTime(f)
        for j, bone in enumerate(bones):
            mm =  bind[j] @ anim[i, j] @ pmi[j]
            cmds.xform(bone, matrix=mm.flatten(), worldSpace=False)
        cmds.setKeyframe(bones)

    ww = solver.weights
    weightArray = np.zeros((len(solver.restPose), len(solver.boneName)))
    for bidx, tt in ww.items():
        for vidx, val in tt.items():
            weightArray[vidx, bidx] = val

    sel = om2.MSelectionList()
    sel.add(skinCls)
    skin_cluster_obj = sel.getDependNode(0)
    skin_cluster_fn = oma2.MFnSkinCluster(skin_cluster_obj)

    sel = om2.MSelectionList()
    sel.add(skinMesh)
    mesh_dag = sel.getDagPath(0)
    mesh_dag.extendToShape()

    skin_cluster_fn.setWeights(
        mesh_dag, om2.MObject(), om2.MIntArray(range(106)), om2.MDoubleArray(weightArray.flatten())
    )


def mayaDemBones(
    restObj: str,
    restFrame: int,
    animObj: str,
    animFrames: list,
    skinnedTarget: str,
    solver: Optional[DemBones] = None,
    **kwargs,
):
    """Run dembones on a maya object

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
    rotateOrders, jointOrients, parInvMats = getJointData(joints)
    restPts = extractRest(restObj)
    bind = getBindMats(cl, joints)

    anim, fTime = extractAnimation(animObj, animFrames)

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

    for k, v in kwargs.items():
        setattr(solver, k, v)

    solver.compute()

    applySolution(solver, animFrames, cl, skinnedTarget)
    return solver


def test():
    solver = mayaDemBones(
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
