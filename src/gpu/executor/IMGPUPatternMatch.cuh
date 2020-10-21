#ifndef __IN_MEMORY_GPU_PATTERN_MATCH_CUH__
#define __IN_MEMORY_GPU_PATTERN_MATCH_CUH__

#include "HybridSeparatePatternMatch.cuh"

class IMGPUPatternMatch : public HybridSeparatePatternMatch {
 public:
  IMGPUPatternMatch(Plan *plan, TrackPartitionedGraph *cpu_rel,
                    HybridGPUComponent *gpu_comp, HybridCPUComponent *cpu_comp,
                    HybridGPUComponent *itp_gpu_comp, size_t thread_num)
      : HybridSeparatePatternMatch(plan, cpu_rel, gpu_comp, cpu_comp,
                                   itp_gpu_comp, thread_num) {}

  // To reuse the code, we consider the in-memory GPU setting as processing a
  // graph with a single partition in the hybrid mode. In this case, all
  // instances are intra-partition and processed by GPUs only.
  virtual void GPUCPUExecute(long long &ans) {
    assert(cpu_relation_->GetPartitionNum() == 1);
    GPUExecute(&ans);
  }

  virtual void InitGPUCPUComponent() { gpu_comp_->InitGPU(); }
};

#endif
