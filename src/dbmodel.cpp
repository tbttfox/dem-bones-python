#include "dbmodel.h"

bool DemBonesModel::cbIterEnd() {
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

void DemBonesModel::validate() {
    nS = 1;  // Only one subject at a time in this context
    nV = u.cols();
    nB = boneName.size();
    nF = v.rows() / 3;

    // clang-format off
        if (nV == 0){throw std::length_error("No rest pose given");}
        if (nB == 0){throw std::length_error("No bones given");}
        if (nF == 0){throw std::length_error("No target animation given");}
    // clang-format on

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

void DemBonesModel::compute() {
    validate();
    DBE::compute();
    DBE::MatrixX lr, lt, gb, lbr, lbt;
    bool degreeRot = false;
    computeRTB(0, lr, lt, gb, lbr, lbt, degreeRot);
}

void DemBonesModel::set_weights(std::vector<std::unordered_map<int, Scalar>>& cweights) {
    std::vector<Eigen::Triplet<Scalar>> trips;
    unsigned int vertIdx = 0;
    for (const auto& vertDict : cweights) {
        for (const auto& weightItem : vertDict) {
            trips.push_back(Eigen::Triplet<Scalar>(weightItem.first, vertIdx, weightItem.second));
        }
        vertIdx++;
    }
    w.setFromTriplets(trips.begin(), trips.end());
}

std::vector<std::unordered_map<int, Scalar>> DemBonesModel::get_weights() {
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
