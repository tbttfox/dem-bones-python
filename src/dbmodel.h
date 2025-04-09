#pragma once
#include <DemBones/DemBones.h>
#include <DemBones/DemBonesExt.h>
#include <DemBones/MatBlocks.h>
#include <nanobind/nanobind.h>

#include <Eigen/Dense>
#include <format>
#include <vector>
namespace nb = nanobind;

typedef double Scalar;
typedef float AniMeshScalar;
typedef Dem::DemBonesExt<Scalar, AniMeshScalar> DBE;

class DemBonesModel : public DBE {
   public:
    double tolerance;
    int patience;
    bool verbose = false;
    bool lock_weights = false;
    bool lock_bones = false;
    DBE::MatrixX lr, lt, gb, lbr, lbt;
    std::vector<int> wvi, wbi;
    std::vector<Scalar> wfv;

    DemBonesModel() : tolerance(1e-3), patience(3) {
        nIters = 30;
        clear();
    }

    void clear() { DBE::clear(); }

    void cbIterBegin() {
        if (verbose) {
            nb::print(std::format("  iteration #{}", iter).c_str());
        }
    }

    bool cbIterEnd();

    void cbInitSplitBegin() {}

    void cbInitSplitEnd() {}

    void cbWeightsBegin() {}

    void cbWeightsEnd() {
        if (verbose) {
            nb::print("    updated weights...");
        }
    }

    void cbTranformationsBegin() {}

    void cbTransformationsEnd() {
        if (verbose) {
            nb::print("    updated transforms...");
        }
    }

    bool cbTransformationsIterEnd() { return false; }

    bool cbWeightsIterEnd() { return false; }

    void validate();

    void compute();

    void exposeWeights();

    void ingestWeights();

    void exportSolver();

   private:
    double prevErr;
    int patience_count;
};
