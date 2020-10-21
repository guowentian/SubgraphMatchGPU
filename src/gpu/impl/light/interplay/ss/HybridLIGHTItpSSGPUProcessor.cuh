#ifndef __HYBRID_GPU_LIGHT_INTERPLAY_SINGLE_SEQUENCE_PROCESSOR_CUH__
#define __HYBRID_GPU_LIGHT_INTERPLAY_SINGLE_SEQUENCE_PROCESSOR_CUH__

#include "HybridLIGHTItpGPUProcessor.cuh"

namespace Light {
class HybridLIGHTItpSSGPUProcessor : public HybridLIGHTItpGPUProcessor {
 public:
  HybridLIGHTItpSSGPUProcessor(LightItpWorkContext *wctx, InterPartTask *task)
      : HybridLIGHTItpGPUProcessor(wctx, task) {}

  virtual void CountInterPart(LightItpWorkContext *wctx,
                              size_t cur_exec_level) {
    auto &ans = *wctx->ans;
    size_t ret = ItpLIGHTCount(wctx, cur_exec_level);
    ans += ret;
  }

  virtual void ComputeCountInterPart(LightItpWorkContext *wctx,
                                     size_t cur_exec_level) {
    auto &ans = *wctx->ans;
    size_t ret = ItpComputeCount(wctx, cur_exec_level);
    ans += ret;
  }
};
}  // namespace Light

#endif