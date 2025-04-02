#pragma once

#include <DemBones/DemBonesExt.h>
#include <DemBones/MatBlocks.h>
#include <Python.h>
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

#define LOG(str)             \
    {                        \
        cout << str << endl; \
    }

#define CHECK_MSTATUS_AND_THROW(status)                                          \
    {                                                                            \
        if (status.error()) throw std::exception(status.errorString().asChar()); \
    }

MDagPath toMDagPath(std::string& name, bool shape) {
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

Eigen::Matrix4d toMatrix4D(MMatrix& source) {
    Eigen::Matrix4d target;

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
    // status = matrix.setRotation(rotation, rotateOrder, MSpace::kObject);
    status = matrix.setRotation(rotation, rotateOrder);
    CHECK_MSTATUS_AND_THROW(status);
    return matrix.asMatrix();
};

std::array<double, 16> toMatrixArray(MMatrix matrix) {
    std::array<double, 16> matrixArray;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            matrixArray[i * 4 + j] = matrix(i, j);
        }
    }

    return matrixArray;
};

template <typename Scalar, typename AniMeshScalar>
class DemBonesModel : public Dem::DemBonesExt<Scalar, AniMeshScalar> {
   public:
    double tolerance;
    int patience;
    std::vector<AniMeshScalar> vertices;
    std::vector<AniMeshScalar> animation;
    std::vector<size_t> counts;
    std::vector<size_t> connects;

    std::vector < std::string >> bone_names;    // "The name of each bone influence."
    std::vector < std::string >> bone_parents;  // "The name of each bone parent, or None if it has
                                                // no parent"
    std::vector<Scalar> bind;                   // "The bind pre-matrix per bone as a numpy array"
    std::vector<Scalar> pre_mul_inv;  // "The inverse of any pre-local transformations per bone as a
                                      // numpy array"
    std::vector<char> rot_order;  // "The rotation order per bone as a numpy array. 0=X, 1=Y, 2=Z,
                                  // so [0, 1, 2] is XYZ order"
    std::vector<Scalar> orient;   // "The euler rotation per bone as a numpy array in degrees"

    std::vector<Scalar> weight_vals;   // "The weight influence values"
    std::vector<size_t> weight_verts;  // "The vertex index for each weight value"
    std::vector<size_t> weight_bones;  // "The bone index for each weight value"
    std::vector<Scalar> weight_locks;  // "The lock percent of each vertex, where 1.0 is fully
                                       // locked"

    DemBonesModel() : tolerance(1e-3), patience(3) {
        nIters = 30;
        clear();
    }

    void clear() {
        DemBonesExt<Scalar, AniMeshScalar>::clear();
        points.clear();
        boneIndex.clear();
    }

    void cbIterBegin() { LOG("  iteration #" << iter); }

    bool cbIterEnd() {
        double err = rmse();
        LOG("    rmse = " << err);
        if ((err < prevErr * (1 + weightEps)) && ((prevErr - err) < tolerance * prevErr)) {
            patience_count--;
            if (patience_count == 0) {
                LOG("  convergence is reached");
                return true;
            }
        } else {
            patience_count = patience;
        }
        prevErr = err;
        return false;
    }

    void cbInitSplitBegin() {}

    void cbInitSplitEnd() {}

    void cbWeightsBegin() {}

    void cbWeightsEnd() { LOG("    updated weights..."); }

    void cbTranformationsBegin() {}

    void cbTransformationsEnd() { LOG("    updated transforms..."); }

    bool cbTransformationsIterEnd() { return false; }

    bool cbWeightsIterEnd() { return false; }

    void compute(bool row_major_mats, bool row_major_verts) { ; }

    void set_weights(const py::list& weights, bool lock_weights) {
        // TODO: Raise an exception
        unsigned int vertIdx = 0;
        weight_verts.clear();
        weight_bones.clear();
        weight_vals.clear();

        for (const auto& vertItem : weights) {
            if (!py::isinstance<py::dict>(vertItem)) {
                continue;
            }
            py::dict vertDict = vertItem.cast(py::dict>();
            for (const auto &weightItem: vertDict){
                if (!py::isinstance<py::int_>(weightItem.first)) {
                    continue;
                }
                if (!py::isinstance<py::float_>(weightItem.second)) {
                    continue;
                }

                weight_verts.push_back(vertIdx);
                weight_bones.push_back(weightItem.first.cast<int>());
                weight_vals.push_back(weightItem.second.cast<Scalar>());
            }
            vertIdx++;
        }

        weight_locks.clear();
        weight_locks.resize(vertIdx, lock_weights ? 1.0 : 0.0);
    }

    py::list get_weights() {
        py::list ret;

        return ret;
    }

    py::array_t<Scalar> get_bone_transforms() {
        py::array_t<Scalar> ret;

        return ret;
    }

   private:
    double prevErr;
    int patience_count;
};

typedef DemBonesModel<double, float> DBM;


// TODO: Switch to getter/setter

PYBIND11_MODULE(_core, m) {
    py::class_<DBM>(m, "DemBones")
        .def(py::init<>())

        // Python behavior params

        .def_readwrite(
            "tolerance", &DBM::tolerance,
            "If the solver fails to converge faster than `tolerance` for `patience` iterations, "
            "bail out\n"
            "default = 1e-3"
        )
        .def_readwrite(
            "patience", &DBM::patience,
            "If the solver fails to converge faster than `tolerance` for `patience` iterations, "
            "bail out\n"
            "default = 3"
        )

        // Solver params
        .def_readwrite("num_iterations", &DBM::nIters, "Number of global iterations, default = 30")
        .def_readwrite(
            "num_transform_iterations", &DBM::nTransIters,
            "Number of bone transformations update iterations per global iteration, default = 5"
        )
        .def_readwrite(
            "translation_affine", &DBM::transAffine,
            "Translations affinity soft constraint, default = 10.0"
        )
        .def_readwrite(
            "translation_affine_norm", &DBM::transAffineNorm,
            "p-norm for bone translations affinity soft constraint, default = 4.0"
        )
        .def_readwrite(
            "num_weight_iterations", &DBM::nWeightsIters,
            "Number of weights update iterations per global iteration, default = 3"
        )
        .def_readwrite(
            "max_influences", &DBM::nnz, "Number of non-zero weights per vertex, default = 8"
        )
        .def_readwrite(
            "weights_smooth", &DBM::weightsSmooth,
            "Weights smoothness soft constraint, default = 1e-4"
        )
        .def_readwrite(
            "weights_smooth_step", &DBM::weightsSmoothStep,
            "Step size for the weights smoothness soft constraint, default = 1.0"
        )
        .def_readwrite(
            "weights_epsilon", &DBM::weightEps, "Epsilon for weights solver, default = 1e-15"
        )

        // Mesh definition
        .def_readwrite("vertices", &DBM::vertices, "Numpy array of vertices")
        .def_readwrite("animation", &DBM::animation, "Numpy array of vertex animation")
        .def_readwrite("counts", &DBM::counts, "Numpy array of face counts")
        .def_readwrite("connects", &DBM::connects, "Numpy array of face connects")

        // Bone definition
        .def_readwrite("bone_names", &DBM::bone_names, "The name of each bone influence.")
        .def_readwrite(
            "bone_parents", &DBM::bone_parents,
            "The name of each bone parent, or None if it has no parent"
        )
        .def_readwrite("bind", &DBM::bind, "The bind pre-matrix per bone as a numpy array")
        .def_readwrite(
            "pre_mul_inv", &DBM::pre_mul_inv,
            "The inverse of any pre-local transformations per bone as a numpy array"
        )
        .def_readwrite(
            "rot_order", &DBM::rot_order,
            "The rotation order per bone as a numpy array. 0=X, 1=Y, 2=Z, so [0, 1, 2] is XYZ order"
        )
        .def_readwrite(
            "orient", &DBM::orient, "The euler rotation per bone as a numpy array in degrees"
        )

        // Weight definition
        .def_readwrite("weight_vals", &DBM::weight_vals, "The weight influence values")
        .def_readwrite("weight_verts", &DBM::weight_verts, "The vertex index for each weight value")
        .def_readwrite("weight_bones", &DBM::weight_bones, "The bone index for each weight value")
        .def_readwrite(
            "weight_locks", &DBM::weight_locks,
            "The lock percent of each vertex, where 1.0 is fully locked"
        )

        // Functions
        .def("get_rmse", &DBM::rmse, "Root mean squared reconstruction error")
        .def(
            "compute", &DBM::compute,
            "Skinning decomposition of alternative updating weights and bone transformations\n"
            "\n"
            "Arguments:\n"
            "    row-major-mats (bool):\n"
            "        Whether the provided matrices will be expected as row-major (the default) or "
            "column-major\n"
            "        Row-major matrices have the translation values along the bottom row\n"
            "    row-major-verts (bool):\n"
            "        Whether the provided vertex arrays will be expected as row-major (the "
            "default) or column-major\n"
            "        Row-major arrays are indexed like array[vertIdx][xyz-component]\n",
            py::arg("row_major_mats") = true, py::arg("row_major_verts" = true)
        )
        .def(
            "set_weights", &DBM::set_weights,
            "Conveinence function for setting the weight values. Takes a list[dict[int, float]] "
            "where\n"
            "the list index is the vertex index, and each dictionary is the bone index mapped to a "
            "weight\n"
            "By default, this locks all the weights unless you pass lock_weights=False",
            py::arg("weights"), py::arg("lock_weights" = true)
        )
        .def(
            "get_weights", &DBM::get_weights,
            "Conveinence function for getting the weight values. Returns a list[dict[int, float]] "
            "where\n"
            "the list index is the vertex index, and each dictionary is the bone index mapped to a "
            "weight"
        )
        .def(
            "get_bone_transforms", &DBM::get_bone_transforms,
            "Get the solved bone transforms as a numpy array. returnValue[boneIdx][frameIdx] = 4x4 "
            "matrix\n"
            "Returns identity matrices if the `compute` function has not been run"
        )
}
