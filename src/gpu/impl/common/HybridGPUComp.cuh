#ifndef __HYBRID_GPU_COMPONENT_CUH__
#define __HYBRID_GPU_COMPONENT_CUH__

#include "CPUGraph.h"
#include "CountProfiler.h"
#include "CudaContext.cuh"
#include "CudaContextManager.cuh"
#include "DevGraph.cuh"
#include "DevGraphPartition.cuh"
#include "GPUProfiler.cuh"
#include "GPUTimer.cuh"
#include "GPUUtil.cuh"
#include "HybridGPUProcessor.cuh"
#include "ProfilerHelper.cuh"
#include "Task.h"
#include "WorkContext.cuh"

class HybridGPUComponent {
 public:
  HybridGPUComponent(Plan* plan, TrackPartitionedGraph* cpu_relation,
                     bool materialize_result, size_t thread_num)
      : cpu_relation_(cpu_relation),
        materialize_result_(materialize_result),
        thread_num_(thread_num) {
    size_t d_partition_num = plan->GetDevPartitionNum();
    gpu_profiler_ = new GPUProfiler(d_partition_num);
    gpu_profiler_->AddPhase("pcie_transfer_time");
    count_profiler_ = new CountProfiler(d_partition_num);

    cuda_contexts_.resize(d_partition_num);
    d_partitions_.resize(d_partition_num);

    for (size_t i = 0; i < d_partition_num; ++i) {
      CUDA_ERROR(cudaSetDevice(i));
      cuda_contexts_[i] =
          CudaContextManager::GetCudaContextManager()->GetCudaContext(i);
      d_partitions_[i] = new DevGraphPartition(cuda_contexts_[i]);
    }
  }
  ~HybridGPUComponent() {
    delete gpu_profiler_;
    gpu_profiler_ = NULL;
    delete count_profiler_;
    count_profiler_ = NULL;

    for (size_t i = 0; i < d_partitions_.size(); ++i) {
      CUDA_ERROR(cudaSetDevice(i));
      delete d_partitions_[i];
      d_partitions_[i] = NULL;
    }
  }

  virtual void GPUThreadExecute(Task* task) {
    size_t d_partition_id = task->d_partition_id_;
    WorkContext* wctx = this->CreateWorkContext(task);
    HybridGPUProcessor* processor = this->CreateGPUProcessor(wctx, task);

    CUDA_ERROR(cudaSetDevice(d_partition_id));
    gpu_profiler_->StartTimer("process_level_time", d_partition_id,
                              cuda_contexts_[d_partition_id]->Stream());
    processor->ProcessLevel(0);
    gpu_profiler_->EndTimer("process_level_time", d_partition_id,
                            cuda_contexts_[d_partition_id]->Stream());

    delete wctx;
    wctx = NULL;
    delete processor;
    processor = NULL;
  }

  virtual WorkContext* CreateWorkContext(Task* task) = 0;

  virtual HybridGPUProcessor* CreateGPUProcessor(WorkContext* wctx,
                                                 Task* task) = 0;

  virtual void InitGPU() = 0;

  virtual void ReportProfile() {
    gpu_profiler_->Report();
    count_profiler_->Report();
  }

  void ReleaseDevicePartition(size_t dev_id) {
    CUDA_ERROR(cudaSetDevice(dev_id));
    d_partitions_[dev_id]->Release();
  }
  void BuildDevicePartition(size_t d_partition_id,
                            GraphPartition* cpu_partition) {
    cudaStream_t stream = cuda_contexts_[d_partition_id]->Stream();
    CUDA_ERROR(cudaSetDevice(d_partition_id));
    gpu_profiler_->StartTimer("pcie_transfer_time", d_partition_id, stream);
    d_partitions_[d_partition_id]->BuildIntraPartition(cpu_partition);
    gpu_profiler_->EndTimer("pcie_transfer_time", d_partition_id, stream);
  }

 protected:
  TrackPartitionedGraph* cpu_relation_;

  std::vector<CudaContext*> cuda_contexts_;
  std::vector<DevGraphPartition*> d_partitions_;

  GPUProfiler* gpu_profiler_;
  CountProfiler* count_profiler_;

  bool materialize_result_;
  size_t thread_num_;
};

#endif
