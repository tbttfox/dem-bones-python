#include "dbmodel.h"

#include <algorithm>  // std::sort std::stable_sort
#include <format>
#include <numeric>  // std::iota

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
}

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

    if (nV == 0) {
        throw std::length_error("No rest pose given");
    }
    if (nB == 0) {
        throw std::length_error("No bones given");
    }
    if (nF == 0) {
        throw std::length_error("No target animation given");
    }

    if (lock_bones || (lockM.size() == 0)) {
        lockM.resize(nB);
        if (lock_bones) {
            lockM.setConstant(1);
        } else {
            lockM.setConstant(0);
        }
    }

    if (lock_weights || (lockW.size() == 0)) {
        lockW.resize(nV);
        if (lock_weights) {
            lockW.setConstant(1);
        } else {
            lockW.setConstant(0);
        }
    }

    if (parent.size() == 0) {
        parent.resize(nB);
        parent.setConstant(-1);
    }

    if (rotOrder.cols() == 0) {
        rotOrder.resize(3, nB);
        rotOrder.row(0).setConstant(0);
        rotOrder.row(1).setConstant(1);
        rotOrder.row(2).setConstant(2);
    }

    if (orient.cols() == 0) {
        orient.resize(3, nB);
        orient.setConstant(0);
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
    if ((m.cols() == 0) || (m.rows() == 0)) {
        m.resize(nF * 4, nB * 4);
        for (size_t j = 0; j < nB; ++j) {
            for (size_t k = 0; k < nF; ++k) {
                m.blk4(k, j) = Matrix4::Identity();
            }
        }
    }

    if (fTime.size() == 0){
        fTime.resize(nF);
        for (size_t i = 0; i< nF; ++i){
            fTime(i) = (double)i;
        }
    }

    // Double check that everything matches
    if (v.cols() != nV) {
        throw std::length_error(std::format(
            "The animation vert count ({}) doesn't match the number of verts in the rest pose ({})",
            v.cols(), nV
        ));
    }
    if (parent.size() != nB) {
        throw std::length_error(std::format(
            "The parent size ({}) doesn't match the boneName size ({})", parent.size(), nB
        ));
    }
    if (rotOrder.cols() != nB) {
        throw std::length_error(std::format(
            "The rotOrder size ({}) doesn't match the boneName size ({})", rotOrder.cols(), nB
        ));
    }
    if (orient.cols() != nB) {
        throw std::length_error(std::format(
            "the orient size ({}) doesn't match the boneName size ({})", orient.cols(), nB
        ));
    }
    if (lockM.size() != nB) {
        throw std::length_error(std::format(
            "The bone tranform lock size ({}) doesn't match the boneName size ({})", lockM.size(),
            nB
        ));
    }
    if (lockW.size() != nV) {
        throw std::length_error(std::format(
            "The weight lock size ({}) doesn't match the number of verts in the rest pose ({})",
            lockW.size(), nV
        ));
    }
    if (bind.cols() != nB * 4) {
        throw std::length_error(std::format(
            "The bind size ({}) doesn't match the boneName size (4 * {})", bind.cols(), nB
        ));
    }
    if (preMulInv.cols() != nB * 4) {
        throw std::length_error(std::format(
            "The preMulInv size ({}) doesn't match the boneName size (4 * {})", preMulInv.cols(), nB
        ));
    }
    if (m.cols() != nB * 4) {
        throw std::length_error(std::format(
            "The m col size ({}) doesn't match the boneName size (4 * {})", m.cols(), nB
        ));
    }
    if (m.rows() != nF * 4) {
        throw std::length_error(std::format(
            "The m row size ({}) doesn't match the number of frames (4 * {})", m.rows(), nF
        ));
    }

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

void DemBonesModel::exposeWeights() {
    wvi.clear();
    wbi.clear();
    wfv.clear();

    for (int boneIdx = 0; boneIdx < w.outerSize(); ++boneIdx) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(w, boneIdx); it; ++it) {
            wbi.push_back(boneIdx);
            wvi.push_back(it.row());
            wfv.push_back(it.value());
        }
    }
}

void DemBonesModel::ingestWeights() {
    std::vector<Eigen::Triplet<Scalar>> trips;
    for (auto i : sort_indexes(wbi)) {
        trips.push_back(Eigen::Triplet<Scalar>(wbi[i], wvi[i], wfv[i]));
    }
    w = DBE::SparseMatrix();  // clear the matrix
    w.setFromTriplets(trips.begin(), trips.end());
}

void DemBonesModel::compute() {
    prevErr = -1;
    patience_count = patience;
    validate();
    ingestWeights();
    DBE::compute();
    bool degreeRot = false;
    computeRTB(0, lr, lt, gb, lbr, lbt, degreeRot);
    exposeWeights();
}
