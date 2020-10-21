#ifndef __GPU_EXECUTOR_HYBRID_COPROCESSING_PATTERN_MATCH_CUH__
#define __GPU_EXECUTOR_HYBRID_COPROCESSING_PATTERN_MATCH_CUH__

#include "HybridInterplayPatternMatch.cuh"

class HybridCoprocessPatternMatch : public HybridInterplayPatternMatch {
 public:
  HybridCoprocessPatternMatch(Plan *plan, TrackPartitionedGraph *cpu_rel,
                              HybridGPUComponent *gpu_comp,
                              HybridCPUComponent *cpu_comp,
                              HybridGPUComponent *itp_gpu_comp,
                              size_t thread_num)
      : HybridInterplayPatternMatch(plan, cpu_rel, gpu_comp, cpu_comp,
                                    itp_gpu_comp, thread_num) {
    cpu_ans_lock_.Init();
  }

  virtual void InitGPUCPUComponent() {
    gpu_comp_->InitGPU();
    cpu_comp_->Init();
    itp_gpu_comp_->InitGPU();
  }

 protected:
  virtual void CPUExecute(long long *cpu_ans) {
#pragma omp parallel num_threads(thread_num_)
    {
      InterPartTask task;
      while (inter_scheduler_->GetTask(task, true)) {
        long long thread_cpu_ans = 0;
        task.ans_ = &thread_cpu_ans;
        cpu_comp_->ThreadExecute(&task);

        cpu_ans_lock_.Lock();
        *cpu_ans += thread_cpu_ans;
        cpu_ans_lock_.Unlock();
      }
    }
  }

  SpinLock cpu_ans_lock_;
};

#endif