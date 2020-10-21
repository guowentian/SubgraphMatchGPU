#ifndef __GPU_EXECUTOR_HYBRID_SEPARATE_PATTERN_MATCH_CUH__
#define __GPU_EXECUTOR_HYBRID_SEPARATE_PATTERN_MATCH_CUH__

#include <cstdio>
#include <future>
#include <thread>
#include <utility>
#include "DeviceManager.h"
#include "InterPartScheduler.h"
#include "IntraPartScheduler.h"
#include "ParallelUtils.h"
#include "PatternMatch.cuh"

class HybridSeparatePatternMatch : public PatternMatch {
 public:
  HybridSeparatePatternMatch(Plan *plan, TrackPartitionedGraph *cpu_rel,
                             HybridGPUComponent *gpu_comp,
                             HybridCPUComponent *cpu_comp,
                             HybridGPUComponent *itp_gpu_comp,
                             size_t thread_num)
      : PatternMatch(plan, cpu_rel, gpu_comp, cpu_comp, itp_gpu_comp,
                     thread_num) {
    dev_manager_ = new DeviceManager(plan->GetDevPartitionNum());
    intra_scheduler_ =
        new IntraPartScheduler(cpu_rel, plan->GetDevPartitionNum());
    inter_scheduler_ =
        new InterPartScheduler(cpu_rel, plan->GetDevPartitionNum(), thread_num);
  }

  ~HybridSeparatePatternMatch() {
    delete dev_manager_;
    dev_manager_ = NULL;
    delete intra_scheduler_;
    intra_scheduler_ = NULL;
    delete inter_scheduler_;
    inter_scheduler_ = NULL;
  }

  virtual void InitGPUCPUComponent() {
    gpu_comp_->InitGPU();
    cpu_comp_->Init();
  }

  virtual void GPUCPUExecute(long long &ans) {
    long long gpu_ans = 0;
    long long cpu_ans = 0;

    auto cpu_task =
        std::async(std::launch::async, [&]() { this->CPUExecute(&cpu_ans); });
    auto gpu_task = std::async(std::launch::async,
                               [&]() { return this->GPUExecute(&gpu_ans); });
    gpu_task.get();
    cpu_task.get();

    ans = gpu_ans + cpu_ans;
  }

 protected:
  virtual void CPUExecute(long long *cpu_ans) {
    InterPartTask task;
    while (inter_scheduler_->GetTask(task, true)) {
      //    std::cout << "CPU start process inter-partition [" <<
      //    task.start_offset_
      //              << "," << task.end_offset_ << "]" << std::endl;
      task.ans_ = cpu_ans;
      cpu_comp_->Execute(&task);

      //    std::cout << "CPU finish process inter-partition [" <<
      //    task.start_offset_
      //              << "," << task.end_offset_ << "]" << std::endl;
    }
    count_profiler_->AddCount("cpu_find_count", 0, *cpu_ans);
  }

  virtual void GPUExecute(long long *gpu_ans) {
    size_t gpu_thread_num = plan_->GetDevPartitionNum();
    std::vector<long long> thread_ans(gpu_thread_num, 0);
#pragma omp parallel num_threads(gpu_thread_num)
    {
      size_t omp_thread_id = ParallelUtils::GetParallelThreadId();
      size_t dev_id = plan_->GetDevPartitionNum();
      if (dev_manager_->AcquireDevice(omp_thread_id, dev_id)) {
        std::cout << "thread_id=" << omp_thread_id << " acquire device "
                  << dev_id << std::endl;
        CUDA_ERROR(cudaSetDevice(dev_id));
        GPUThreadIntraPart(omp_thread_id, dev_id, thread_ans[omp_thread_id]);
        dev_manager_->ReleaseDevice(omp_thread_id, dev_id);
      }
    }

    *gpu_ans = 0;
    for (size_t i = 0; i < gpu_thread_num; ++i) {
      *gpu_ans += thread_ans[i];
    }
    count_profiler_->AddCount("intra_partition_count", 0, *gpu_ans);
    count_profiler_->AddCount("gpu_find_count", 0, *gpu_ans);
  }

  virtual void GPUThreadIntraPart(size_t omp_thread_id, size_t dev_id,
                                  long long &thread_ans) {
    size_t h_partition_id = cpu_relation_->GetPartitionNum();
    while (intra_scheduler_->GetPartitionForTask(h_partition_id)) {
      // copy graph to device memory
      gpu_comp_->BuildDevicePartition(
          dev_id, cpu_relation_->GetPartition(h_partition_id));
      IntraPartTask task;
      // get next task in this partition
      while (intra_scheduler_->GetTask(h_partition_id, task)) {
        std::cout << "thread_id=" << omp_thread_id << ",dev_id=" << dev_id
                  << ",start process h_partition_id=" << h_partition_id
                  << ",vertex_ids=[" << task.start_offset_ << ","
                  << task.end_offset_ << "]" << std::endl;

        task.Set(INTRA_PARTITION, &thread_ans, dev_id,
                 cpu_relation_->GetPartition(h_partition_id),
                 task.start_offset_, task.end_offset_);

        gpu_comp_->GPUThreadExecute(&task);

        std::cout << "thread_id=" << omp_thread_id << ",dev_id=" << dev_id
                  << ",finish process h_partition_id=" << h_partition_id
                  << ",vertex_ids=[" << task.start_offset_ << ","
                  << task.end_offset_ << "]" << std::endl;
      }
      gpu_comp_->ReleaseDevicePartition(dev_id);

      if (intra_scheduler_->FinishPartition(h_partition_id)) {
        // When no more tasks in this partition, release the graph partition in
        // main memory as it is not needed any more.
        cpu_relation_->ReleasePartition(h_partition_id);
      }
    }
  }

 protected:
  DeviceManager *dev_manager_;
  IntraPartScheduler *intra_scheduler_;
  InterPartScheduler *inter_scheduler_;
};

#endif