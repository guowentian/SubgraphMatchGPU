#ifndef __PATTERN_MATCH_CUH__
#define __PATTERN_MATCH_CUH__

#include <cassert>
#include <iostream>
#include <map>
#include <vector>

#include <omp.h>
#include "CPUFilter.h"
#include "CPUIntersection.h"
#include "CountProfiler.h"
#include "HybridGPUComp.cuh"
#include "HybridPtGroupCPUComp.h"
#include "PhaseProfiler.h"
#include "TimeMeasurer.h"

class PatternMatch {
 public:
  PatternMatch(Plan *plan, TrackPartitionedGraph *cpu_rel,
               HybridGPUComponent *gpu_comp, HybridCPUComponent *cpu_comp,
               HybridGPUComponent *itp_gpu_comp, size_t thread_num)
      : plan_(plan),
        cpu_relation_(cpu_rel),
        gpu_comp_(gpu_comp),
        cpu_comp_(cpu_comp),
        itp_gpu_comp_(itp_gpu_comp),
        thread_num_(thread_num) {
    phase_profiler_ = new PhaseProfiler(thread_num_);
    phase_profiler_->AddPhase("gpu_time");
    phase_profiler_->AddPhase("cpu_time");
    phase_profiler_->AddPhase("intra_part_time");
    phase_profiler_->AddPhase("inter_part_time");

    count_profiler_ = new CountProfiler(thread_num_);
    count_profiler_->AddPhase("intra_partition_count");
    count_profiler_->AddPhase("gpu_find_count");
    count_profiler_->AddPhase("cpu_find_count");

    // Initialize
    // Execution configuration might be changed by the query plan
    execute_mode_ = HYBRID_CPU_GPU;
    variant_ = O2;
  }
  ~PatternMatch() {
    delete phase_profiler_;
    phase_profiler_ = NULL;
    delete count_profiler_;
    count_profiler_ = NULL;
  }

  void Execute() {
    double total_elapsed_time = 0;
    double init_elapsed_time = 0;
    omp_set_num_threads(thread_num_);

    std::cout << "start execute..." << std::endl;
    TimeMeasurer timer;
    timer.StartTimer();

    // 1. generate the plans for inter-partition workload
    // including the execute_mode and variant used, and
    // grouping of search sequences
    this->OptimizePlan();

    // 2. initialization of each component
    this->InitGPUCPUComponent();

    timer.EndTimer();
    init_elapsed_time += timer.GetElapsedMicroSeconds();
    total_elapsed_time += timer.GetElapsedMicroSeconds();
    plan_->Print();

    timer.StartTimer();

    // 3. execution
    GPUCPUExecute(total_match_count_);
    CUDA_ERROR(cudaDeviceSynchronize());

    timer.EndTimer();
    total_elapsed_time += timer.GetElapsedMicroSeconds();

    PrintStatistics();

    std::cout << "total_match_count=" << total_match_count_
              << ", elapsed_time=" << total_elapsed_time / 1000.0 << "ms"
              << std::endl;
  }

  virtual void OptimizePlan() {
    plan_->OptimizePlan();
    execute_mode_ = plan_->GetExecuteMode();
    variant_ = plan_->GetVariant();
  }

  virtual void InitGPUCPUComponent() = 0;

  virtual void GPUCPUExecute(long long &ans) = 0;

  void PrintStatistics() {
#if defined(PROFILE)
    gpu_comp_->ReportProfile();

    phase_profiler_->Report("cpu_time");
    double total_gpu_execute_time = phase_profiler_->AggregatePhase("gpu_time");
    double total_cpu_execute_time = phase_profiler_->AggregatePhase("cpu_time");
    double total_intra_part_time =
        phase_profiler_->AggregatePhase("intra_part_time");
    double total_inter_part_time =
        phase_profiler_->AggregatePhase("inter_part_time");
    long long total_intra_partition_count =
        count_profiler_->GetCount("intra_partition_count");

    // total_cpu_execute_time -= total_gpu_execute_time;
    std::cout << "intra_partition_count=" << total_intra_partition_count
              << ",intra_partition_rate="
              << total_intra_partition_count * 1.0 / total_match_count_
              << std::endl;
    std::cout << "gpu_execute_time="
              << total_gpu_execute_time / plan_->GetDevPartitionNum() / 1000.0
              << "ms (avg per GPU)"
              << ", cpu_execute_time="
              << total_cpu_execute_time / thread_num_ / 1000.0
              << "ms (avg per thread)" << std::endl;
    std::cout << "total_intra_part_time_=" << total_intra_part_time / 1000.0
              << "ms, total_inter_part_time=" << total_inter_part_time / 1000.0
              << "ms" << std::endl;

    if (itp_gpu_comp_) {
      itp_gpu_comp_->ReportProfile();
    }
    if (cpu_comp_) {
      cpu_comp_->ReportProfile();
    }
#endif
  }

  long long GetTotalMatchCount() const { return total_match_count_; }

  // For interplay, GPU would search inter-partition instances
  bool UseInterplay() const {
    return execute_mode_ == HYBRID_CPU_GPU &&
           (variant_ == O2 || variant_ == O3);
  }

 protected:
  Plan *plan_;
  TrackPartitionedGraph *cpu_relation_;
  size_t thread_num_;
  ExecuteMode execute_mode_;
  Variant variant_;

  HybridGPUComponent *gpu_comp_;
  HybridCPUComponent *cpu_comp_;
  HybridGPUComponent *itp_gpu_comp_;

  CountProfiler *count_profiler_;
  PhaseProfiler *phase_profiler_;

  long long total_match_count_;
};

#endif
