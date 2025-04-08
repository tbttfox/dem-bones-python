#include "dbmodel.h"

#include <algorithm>  // std::sort std::stable_sort
#include <format>
#include <fstream>
#include <iostream>
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

    if (fTime.size() == 0) {
        fTime.resize(nF);
        for (size_t i = 0; i < nF; ++i) {
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
    w.resize(nB, nV);
    w.setFromTriplets(trips.begin(), trips.end());
}

void DemBonesModel::exportSolver() {
    std::string basename = "D:\\temp\\dembones_out\\mine\\";

    {
        std::string filename = basename + "nIters.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << nIters << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "nInitIters.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << nInitIters << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "nTransIters.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << nTransIters << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "transAffine.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << transAffine << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "transAffineNorm.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << transAffineNorm << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "nWeightsIters.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << nWeightsIters << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "nnz.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << nnz << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "weightsSmooth.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << weightsSmooth << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "weightsSmoothStep.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << weightsSmoothStep << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "weightEps.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << weightEps << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "nV.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << nV << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "nB.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << nB << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "nS.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << nS << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "nF.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << nF << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "fStart.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << fStart << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "subjectID.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << subjectID << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "u.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << u << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "w.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << w << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "lockW.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << lockW << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "m.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << m << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "lockM.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << lockM << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "v.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << v << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "fv.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            for (const auto& f : fv) {
                for (auto num : f) {
                    file << num << ",";
                }
                file << std::endl;
            }
            file.close();
        }
    }
    {
        std::string filename = basename + "fTime.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << fTime << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "boneName.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            for (const auto& bn : boneName) {
                file << bn << std::endl;
            }
            file.close();
        }
    }
    {
        std::string filename = basename + "parent.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << parent << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "bind.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << bind << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "preMulInv.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << preMulInv << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "rotOrder.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << rotOrder << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "orient.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << orient << std::endl;
            file.close();
        }
    }
    {
        std::string filename = basename + "bindUpdate.txt";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << bindUpdate << std::endl;
            file.close();
        }
    }
}

void DemBonesModel::compute() {
    prevErr = -1;
    patience_count = patience;
    validate();
    ingestWeights();

    //exportSolver();
    DBE::compute();

    bool degreeRot = false;
    computeRTB(0, lr, lt, gb, lbr, lbt, degreeRot);
    exposeWeights();
}
