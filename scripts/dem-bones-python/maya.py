from itertools import groupby
from maya import cmds
from maya.api import OpenMaya as om2


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
        mt.setValue(frame)
        fTimes.append(mt.asUnits(mt.kSeconds))
        animData[i] = np.array(mesh.getPoints())[:, :3]

    cmds.currentTime(curtime)
    return animData, fTimes


def extractRest(restMesh: str, restFrame: int):
    mesh = getMesh(restMesh)
    return np.array(mesh.getPoints())[:, :3]


def toRanges(idxs):
    ret = []
    grps = groupby(enumerate(ordered), key=lambda x: x[1] - x[0])
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
    cl = cmds.ls(cmds.listHistory(skinnedMesh), type="skinCluster")
    if not cl:
        raise ValueError(f"No skin cluster found on object: {skinnedMesh}")
    return cl[0]


def getSkinWeights(cl):
    weights = {}
    ranges = toRanges(cmds.getAttr(f"{cl}.weightList", mi=True))
    for rng in ranges:
        mr = toMelRange(rng)
        vals = cmds.getAttr("{}.weightList[{}].weights".format(cl, mr))
        boneIdxs = cmds.getAttr("{}.weightList[{}].weights".format(cl, mr), mi=True)

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
    return bindMats


def getInfluenceHierarchy(cl):
    joints = cmds.skinCluster(cl, query=True, influence=True)
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
            parInvMats[i] = np.eye(4)
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


def mayaDemBones(
    solver,
    restObj: str,
    restFrame: int,
    animObj: str,
    animFrames: list,
    skinnedTarget: str,
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
    cmds.currentTime(restFrame)

    # All done at rest-frame time
    topo = getTopo(restObj)
    cl = getSkinCluster(skinnedMesh)
    weights = getSkinWeights(cl)
    joints, parIdxs = getInfluenceHierarchy(cl)
    rotateOrders, jointOrients, parInvMats = getJointData(joints)
    restPts = extractRest(restObj, restFrame)
    bind = getBindMats(cl, joints)

    anim, fTime = extractAnimation(animObj, animFrames)

    solver.restPose = restPts
    solver.anim = anim
    solver.fv = topo
    solver.fTime = fTime
    solver.boneName = joints
    solver.parent = parIdxs
    solver.weights = weights
    solver.lockW = np.ones(len(restPts))
    solver.rotOrder = rotateOrders
    solver.orient = jointOrients
    solver.bind = bind
    solver.preMulInv = parentInvMats

    for k, v in kwargs.items():
        setattr(solver, k, v)

    solver.compute()


solver = DemBones()
mayaDemBones(
    solver,
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
