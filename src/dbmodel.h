#include <DemBones/DemBonesExt.h>
#include <DemBones/MatBlocks.h>

#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>
#include <vector>

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
    std::vector<int> wvi, wbi;
    std::vector<Scalar> wfv;

    DemBonesModel() : tolerance(1e-3), patience(3) {
        nIters = 30;
        clear();
    }

    void clear() { Dem::DemBonesExt<Scalar, AniMeshScalar>::clear(); }

    void cbIterBegin() { LOG("  iteration #" << iter); }

    bool cbIterEnd();

    void cbInitSplitBegin() {}

    void cbInitSplitEnd() {}

    void cbWeightsBegin() {}

    void cbWeightsEnd() { LOG("    updated weights..."); }

    void cbTranformationsBegin() {}

    void cbTransformationsEnd() { LOG("    updated transforms..."); }

    bool cbTransformationsIterEnd() { return false; }

    bool cbWeightsIterEnd() { return false; }

    void validate();

    void compute();

    void exposeWeights();

    void ingestWeights();

   private:
    double prevErr;
    int patience_count;
};
