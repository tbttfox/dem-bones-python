#include <Python.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "dbmodel.h"

namespace nb = nanobind;

NB_MODULE(_dem_bones_core, m) {
    nb::class_<DemBonesModel>(m, "DemBones")
        .def(nb::init<>())

        // Python behavior params
        .def_rw(
            "tolerance", &DemBonesModel::tolerance,
            "If the solver fails to converge faster than `tolerance` for `patience` iterations, "
            "bail out\n"
            "default = 1e-3"
        )
        .def_rw(
            "patience", &DemBonesModel::patience,
            "If the solver fails to converge faster than `tolerance` for `patience` iterations, "
            "bail out\n"
            "default = 3"
        )

        .def_rw("verbose", &DemBonesModel::verbose, "Whether to print solver updates")

        // Solver params
        .def_rw(
            "nIters", &DemBonesModel::nIters, "Number of global iterations, default = 30"
        )
        .def_rw(
            "nTransIters", &DemBonesModel::nTransIters,
            "Number of bone transformations update iterations per global iteration, default = 5"
        )
        .def_rw(
            "transAffine", &DemBonesModel::transAffine,
            "Translations affinity soft constraint, default = 10.0"
        )
        .def_rw(
            "transAffineNorm", &DemBonesModel::transAffineNorm,
            "p-norm for bone translations affinity soft constraint, default = 4.0"
        )
        .def_rw(
            "nWeightsIters", &DemBonesModel::nWeightsIters,
            "Number of weights update iterations per global iteration, default = 3"
        )
        .def_rw(
            "nnz", &DemBonesModel::nnz, "Number of non-zero weights per vertex, default = 8"
        )
        .def_rw(
            "weightsSmooth", &DemBonesModel::weightsSmooth,
            "Weights smoothness soft constraint, default = 1e-4"
        )
        .def_rw(
            "weightsSmoothStep", &DemBonesModel::weightsSmoothStep,
            "Step size for the weights smoothness soft constraint, default = 1.0"
        )
        .def_rw(
            "weightsEps", &DemBonesModel::weightEps, "Epsilon for weights solver, default = 1e-15"
        )

        // Array and vector data
        .def_rw("_u", &DemBonesModel::u, "Internal storage for rest verts")
        .def_rw("_v", &DemBonesModel::v, "Internal storage for animated verts")
        .def_rw("lockW", &DemBonesModel::lockW, "The lock percent of each vertex")
        .def_rw("_m", &DemBonesModel::m, "The bone transformations")
        .def_rw("lockM", &DemBonesModel::lockM, "The bone transformations lock control")
        .def_rw("fv", &DemBonesModel::fv, "The mesh topology")
        .def_rw("fTime", &DemBonesModel::fTime, "The timestamps per-frame")
        .def_rw("boneName", &DemBonesModel::boneName, "The name of the bones")
        .def_rw("parent", &DemBonesModel::parent, "The indices of the parents of each bone")
        .def_rw("_bind", &DemBonesModel::bind, "The original bind pre-matrix")
        .def_rw("_preMulInv", &DemBonesModel::preMulInv, "Inverse Pre-mult matrices")
        .def_rw("_rotOrder", &DemBonesModel::rotOrder, "The rotation order for each bone")
        .def_rw("_orient", &DemBonesModel::orient, "The orientation of each bone")

        .def_rw("_wvi", &DemBonesModel::wvi, "The vertex indices of the weight values")
        .def_rw("_wbi", &DemBonesModel::wbi, "The bone indices of the weight values")
        .def_rw("_wfv", &DemBonesModel::wfv, "The actual weight values")

        .def_ro("_lr", &DemBonesModel::lr, "Output local rotations")
        .def_ro("_lt", &DemBonesModel::lt, "Output local translations")
        .def_ro("_gb", &DemBonesModel::gb, "Output BindMatrices")
        .def_ro("_lbr", &DemBonesModel::lbr, "Output Local Bind rotations")
        .def_ro("_lbt", &DemBonesModel::lbt, "Output Local Bind translations")

        .def_rw(
            "bindUpdate", &DemBonesModel::bindUpdate,
            "Bind transformation update\n"
            "    0=keep original\n"
            "    1=set translations to p-norm centroids and rotations to identity\n"
            "    2=do 1 and group joints"
        )

        .def_prop_ro(
            "rmse", &DemBonesModel::rmse, "Root mean squared reconstruction error"
        )

        // Functions
        .def(
            "compute", &DemBonesModel::compute,
            "Skinning decomposition of alternative updating weights and bone transformations\n"
        );
}
