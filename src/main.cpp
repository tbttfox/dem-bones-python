#pragma once

#include <DemBones/DemBonesExt.h>
#include <DemBones/MatBlocks.h>
#include <Python.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

#define LOG(str)                       \
    {                                  \
        std::cout << str << std::endl; \
    }

typedef double Scalar;
typedef float AniMeshScalar;
typedef Dem::DemBonesExt<Scalar, AniMeshScalar> DBE;
typedef Eigen::Matrix4<Scalar> Matrix4;

class DemBonesModel : public DBE {
   public:
    double tolerance;
    int patience;
    bool lock_weights = false;
    bool lock_bones = false;
    DBE::MatrixX lr, lt, gb, lbr, lbt;

    DemBonesModel() : tolerance(1e-3), patience(3) {
        nIters = 30;
        clear();
    }

    void clear() {
        Dem::DemBonesExt<Scalar, AniMeshScalar>::clear();
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

    void validate() {
        nS = 1;  // Only one subject at a time in this context
        nV = u.cols();
        nB = boneName.size();
        nF = v.rows() / 3;

        if (lock_bones || lockM.size() == 0) {
            lockM.resize(nB);
            lockM *= 0;
            if (lock_bones) {
                lockM.array() += 1;
            }
        }

        if (lock_weights || lockW.size() == 0) {
            lockW.resize(nB);
            lockW *= 0;
            if (lock_weights) {
                lockW.array() += 1;
            }
        }

        // fill with identity if they're unset
        if (bind.cols() == 0) {
            bind.resize(4, 4 * nB);
            for (size_t j = 0; j < nB; ++j) {
                bind.blk4(0, j) = Matrix4::Identity();
            }
        }

        if (preMulInv.cols() == 0) {
            preMulInv.resize(4, 4 * nB);
            for (size_t j = 0; j < nB; ++j) {
                preMulInv.blk4(0, j) = Matrix4::Identity();
            }
        }
        if (m.cols() == 0 || m.rows() == 0) {
            m.resize(nF * 16, nB * 4);
            for (size_t j = 0; j < nB; ++j) {
                for (size_t k = 0; k < nF; ++k) {
                    m.blk4(k, j) = Matrix4::Identity();
                }
            }
        }

        // clang-format off
        // Double check that everything matches
        if (v.cols() != nV){throw std::length_error("The animation doesn't match the number of verts in the rest pose");}
        if (parent.size() != nB){throw std::length_error("The parent size doesn't match the boneName size");}
        if (rotOrder.cols() != nB){throw std::length_error("The rotOrder size doesn't match the boneName size");}
        if (orient.cols() != nB){throw std::length_error("the orient size doesn't match the boneName size");}
        if (lockM.size() != nB){throw std::length_error("The bone tranform lock size doesn't match the boneName size");}
        if (lockW.size() != nV){throw std::length_error("The weight lock size doesn't match the number of verts in the rest pose");}

        if (bind.cols() != nB * 4){throw std::length_error("The bind size doesn't match the boneName size");}
        if (preMulInv.cols() != nB * 4){throw std::length_error("The preMulInv size doesn't match the boneName size");}
        if (m.cols() != nB * 4){throw std::length_error("The m size doesn't match the boneName size");}
        if (m.rows() != nF * 16){throw std::length_error("The m size doesn't match the number of frames");}
        // clang-format on

        fStart.resize(nS + 1);
        fStart(0) = 0;
        fStart(1) = nF;
        subjectID.resize(nF);
        for (int s = 0; s < nS; s++) {
            for (int k = fStart(s); k < fStart(s + 1); k++) {
                subjectID(k) = s;
            }
        }
    }

    void compute() {
        validate();
        DBE::compute();
        DBE::MatrixX lr, lt, gb, lbr, lbt;
        bool degreeRot = false;
        computeRTB(0, lr, lt, gb, lbr, lbt, degreeRot);
    }

    void set_weights(const py::list& weights) {
        std::vector<Eigen::Triplet<Scalar>> trips;
        unsigned int vertIdx = 0;
        auto cweights = weights.cast<std::vector<std::unordered_map<int, Scalar>>>();
        for (const auto& vertDict : cweights) {
            for (const auto& weightItem : vertDict) {
                trips.push_back(Eigen::Triplet<Scalar>(weightItem.first, vertIdx, weightItem.second)
                );
            }
            vertIdx++;
        }
        w.setFromTriplets(trips.begin(), trips.end());
    }

    std::vector<std::unordered_map<int, Scalar>> get_weights() {
        std::vector<std::unordered_map<int, Scalar>> ret;
        for (int boneIdx = 0; boneIdx < w.outerSize(); ++boneIdx) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(w, boneIdx); it; ++it) {
                auto weight = it.value();
                auto vertIdx = it.row();
                if (vertIdx > ret.size() - 1) {
                    ret.resize(vertIdx);
                }
                ret[vertIdx][boneIdx] = weight;
            }
        }
        return ret;
    }

    void initialize_lock_weights(int numVerts) {
        if (lockW.size() == 0) {
            lockW = DBE::VectorX::Zero(numVerts);
            if (lock_weights) {
                lockW.array() += 1.0;
            }
        }
    }

    // Weight definition
    DBE::VectorX get_weight_locks() { return lockW; }
    void set_weight_locks(Eigen::Ref<DBE::VectorX> newW) { lockW = newW; }

   private:
    double prevErr;
    int patience_count;
};

PYBIND11_MODULE(_core, m) {
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
