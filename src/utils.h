#pragma once

#include <maya/MAnimControl.h>
#include <maya/MColor.h>
#include <maya/MColorArray.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MEulerRotation.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnMesh.h>
#include <maya/MFnSkinCluster.h>
#include <maya/MGlobal.h>
#include <maya/MItDependencyGraph.h>
#include <maya/MMatrix.h>
#include <maya/MObject.h>
#include <maya/MPlug.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MQuaternion.h>
#include <maya/MSelectionList.h>
#include <maya/MString.h>
#include <maya/MTime.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MTypes.h>

#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

#define LOG(str)     \
    {                \
        cout << str; \
    }

#define CHECK_MSTATUS_AND_THROW(status)                                          \
    {                                                                            \
        if (status.error()) throw std::exception(status.errorString().asChar()); \
    }

namespace Conversion {
MDagPath toMDagPath(string& name, bool shape) {
    MStatus status;
    MDagPath dag;
    MSelectionList selection;

    status = selection.add(MString(name.c_str()));
    CHECK_MSTATUS_AND_THROW(status);
    status = selection.getDagPath(0, dag, MObject::kNullObj);
    CHECK_MSTATUS_AND_THROW(status);

    if (shape) {
        status = dag.extendToShape();
        CHECK_MSTATUS_AND_THROW(status);
    }

    return dag;
};

Matrix4d toMatrix4D(MMatrix& source) {
    Matrix4d target;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            target(j, i) = source(i, j);
        }
    }

    return target;
};

MMatrix toMMatrix(
    MVector& translate, MVector& rotate, MTransformationMatrix::RotationOrder rotateOrder
) {
    MStatus status;
    MTransformationMatrix matrix;

    status = matrix.setTranslation(translate, MSpace::kObject);
    CHECK_MSTATUS_AND_THROW(status);

    const double rotation[3] = {rotate.x, rotate.y, rotate.z};
    status = matrix.setRotation(rotation, rotateOrder, MSpace::kObject);
    CHECK_MSTATUS_AND_THROW(status);
    return matrix.asMatrix();
};

array<double, 16> toMatrixArray(MMatrix matrix) {
    array<double, 16> matrixArray;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            matrixArray[i * 4 + j] = matrix(i, j);
        }
    }

    return matrixArray;
};

double toGrayScale(const MColor& c) { return 0.2989 * c.r + 0.5870 * c.g + 0.1140 * c.b; }
}  // namespace Conversion
