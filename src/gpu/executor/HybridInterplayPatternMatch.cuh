#ifndef __GPU_EXECUTOR_HYBRID_INTERPLAY_PATTERN_MATCH_CUH__
#define __GPU_EXECUTOR_HYBRID_INTERPLAY_PATTERN_MATCH_CUH__

#include "HybridSeparatePatternMatch.cuh"

class HybridInterplayPatternMatch : public HybridSeparatePatternMatch {
 public:
  HybridInterplayPatternMatch(Plan *plan, TrackPartitionedGraph *cpu_rel,
                              HybridGPUComponent *gpu_comp,
                              HybridCPUComponent *cpu_comp,
                              HybridGPUComponent *itp_gpu_comp,
                              size_t thread_num)
      : HybridSeparatePatternMatch(plan, cpu_rel, gpu_comp, cpu_comp,
                                   itp_gpu_comp, thread_num) {}

  virtual void InitGPUCPUComponent() {
    gpu_comp_->InitGPU();
    itp_gpu_comp_->InitGPU();
  }

 protected:
  virtual void CPUExecute(long long *cpu_ans) {
    // CPU does not process inter-partition workload
  }

  virtual void GPUExecute(long long *gpu_ans) {
    std::cout << "GPU start interplay processing" << std::endl;
    size_t gpu_thread_num = plan_->GetDevPartitionNum();
    std::vector<long long> thread_intra_part_ans(gpu_thread_num, 0);
    std::vector<long long> thread_inter_part_ans(gpu_thread_num, 0);
#pragma omp parallel num_threads(gpu_thread_num)
    {
      size_t omp_thread_id = ParallelUtils::GetParallelThreadId();
      size_t dev_id = plan_->GetDevPartitionNum();
      if (dev_manager_->AcquireDevice(omp_thread_id, dev_id)) {
        std::cout << "thread_id=" << omp_thread_id << " acquire device "
                  << dev_id << std::endl;
        CUDA_ERROR(cudaSetDevice(dev_id));
        GPUThreadIntraPart(omp_thread_id, dev_id,
                           thread_intra_part_ans[omp_thread_id]);
        GPUThreadInterPart(omp_thread_id, dev_id,
                           thread_inter_part_ans[omp_thread_id]);
        dev_manager_->ReleaseDevice(omp_thread_id, dev_id);
      }
    }

    long long intra_part_gpu_ans = 0;
    for (size_t i = 0; i < gpu_thread_num; ++i) {
      *gpu_ans += thread_inter_part_ans[i] + thread_intra_part_ans[i];
      intra_part_gpu_ans += thread_intra_part_ans[i];
    }
    count_profiler_->AddCount("intra_partition_count", 0, intra_part_gpu_ans);
    count_profiler_->AddCount("gpu_find_count", 0, *gpu_ans);
  }

  virtual void GPUThreadInterPart(size_t omp_thread_id, size_t dev_id,
                                  long long &thread_ans) {
    InterPartTask inter_part_task;
    while (inter_scheduler_->GetTask(inter_part_task, false)) {
      std::cout << "thread_id=" << omp_thread_id << ",dev_id=" << dev_id
                << ",GPU start process inter_partition ["
                << inter_part_task.start_offset_ << ","
                << inter_part_task.end_offset_ << ")" << std::endl;
      inter_part_task.ans_ = &thread_ans;
      inter_part_task.d_partition_id_ = dev_id;

      itp_gpu_comp_->GPUThreadExecute(&inter_part_task);

      std::cout << "thread_id=" << omp_thread_id << ",dev_id=" << dev_id
                << ",finish process inter_partition ["
                << inter_part_task.start_offset_ << ","
                << inter_part_task.end_offset_ << ")" << std::endl;
    }
  }
};

#endif