#include <Python.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dbmodel.h"

namespace py = pybind11;

PYBIND11_MODULE(_dem_bones_core, m) {
    py::class_<DemBonesModel>(m, "DemBones")
        .def(py::init<>())

        // Python behavior params
        .def_readwrite(
            "tolerance", &DemBonesModel::tolerance,
            "If the solver fails to converge faster than `tolerance` for `patience` iterations, "
            "bail out\n"
            "default = 1e-3"
        )
        .def_readwrite(
            "patience", &DemBonesModel::patience,
            "If the solver fails to converge faster than `tolerance` for `patience` iterations, "
            "bail out\n"
            "default = 3"
        )

        // Solver params
        .def_readwrite(
            "num_iterations", &DemBonesModel::nIters, "Number of global iterations, default = 30"
        )
        .def_readwrite(
            "num_transform_iterations", &DemBonesModel::nTransIters,
            "Number of bone transformations update iterations per global iteration, default = 5"
        )
        .def_readwrite(
            "translation_affine", &DemBonesModel::transAffine,
            "Translations affinity soft constraint, default = 10.0"
        )
        .def_readwrite(
            "translation_affine_norm", &DemBonesModel::transAffineNorm,
            "p-norm for bone translations affinity soft constraint, default = 4.0"
        )
        .def_readwrite(
            "num_weight_iterations", &DemBonesModel::nWeightsIters,
            "Number of weights update iterations per global iteration, default = 3"
        )
        .def_readwrite(
            "max_influences", &DemBonesModel::nnz,
            "Number of non-zero weights per vertex, default = 8"
        )
        .def_readwrite(
            "weights_smooth", &DemBonesModel::weightsSmooth,
            "Weights smoothness soft constraint, default = 1e-4"
        )
        .def_readwrite(
            "weights_smooth_step", &DemBonesModel::weightsSmoothStep,
            "Step size for the weights smoothness soft constraint, default = 1.0"
        )
        .def_readwrite(
            "weights_epsilon", &DemBonesModel::weightEps,
            "Epsilon for weights solver, default = 1e-15"
        )

        .def_readwrite(
            "lock_weights", &DemBonesModel::lock_weights,
            "If weight locks are unset, then lock them all"
        )
        .def_readwrite(
            "lock_bones", &DemBonesModel::lock_bones, "If bone locks are unset, then lock them all"
        )

        // Array and vector data
        .def_readwrite("_u", &DemBonesModel::u, "Internal storage for rest verts")
        .def_property(
            "weights", &DemBonesModel::get_weights, &DemBonesModel::set_weights,
            "Set the weight values in the form of a list like: list[dict[int, float]]\n"
            "Where the list index is the vertex index, and each dictionary is the bone index\n"
            "mapped to a weight\n"
        )

        .def_readwrite("lockW", &DemBonesModel::lockW, "The lock percent of each vertex")
        .def_readwrite("_m", &DemBonesModel::m, "The bone transformations")
        .def_readwrite("lockM", &DemBonesModel::lockM, "The bone transformations lock control")
        .def_readwrite("fv", &DemBonesModel::fv, "The mesh topology")
        .def_readwrite("fTime", &DemBonesModel::fTime, "The timestamps per-frame")
        .def_readwrite("boneName", &DemBonesModel::boneName, "The name of the bones")
        .def_readwrite("parent", &DemBonesModel::parent, "The indices of the parents of each bone")
        .def_readwrite("_bind", &DemBonesModel::bind, "The original bind pre-matrix")
        .def_readwrite("_preMulInv", &DemBonesModel::preMulInv, "Inverse Pre-mult matrices")
        .def_readwrite("_rotOrder", &DemBonesModel::rotOrder, "The rotation order for each bone")
        .def_readwrite("_orient", &DemBonesModel::orient, "The orientation of each bone")

        .def_readonly("_lr", &DemBonesModel::lr, "Output local rotations")
        .def_readonly("_lt", &DemBonesModel::lt, "Output local translations")
        .def_readonly("_gb", &DemBonesModel::gb, "Output BindMatrices")
        .def_readonly("_lbr", &DemBonesModel::lbr, "Output Local Bind rotations")
        .def_readonly("_lbt", &DemBonesModel::lbt, "Output Local Bind translations")

        .def_readwrite(
            "bindUpdate", &DemBonesModel::bindUpdate,
            "Bind transformation update\n"
            "    0=keep original\n"
            "    1=set translations to p-norm centroids and rotations to identity\n"
            "    2=do 1 and group joints"
        )

        .def_property_readonly(
            "rmse", &DemBonesModel::rmse, "Root mean squared reconstruction error"
        )

        // Functions
        .def(
            "compute", &DemBonesModel::compute,
            "Skinning decomposition of alternative updating weights and bone transformations\n"
        );
}
