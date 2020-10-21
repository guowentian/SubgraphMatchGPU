#ifndef __HYBRID_GPU_PROCESSOR_CUH__
#define __HYBRID_GPU_PROCESSOR_CUH__

#include "BatchData.h"
#include "BatchManager.cuh"
#include "Task.h"
#include "WorkContext.cuh"

class HybridGPUProcessor {
 public:
  HybridGPUProcessor(WorkContext *wctx, Task *task)
      : wctx_(wctx), task_(task) {}

  virtual void ProcessLevel(size_t cur_exec_level) {
    if (this->IsProcessOver(cur_exec_level)) {
      return;
    }
    auto gpu_context = wctx_->context;
    size_t batch_size = this->GetLevelBatchSize(cur_exec_level);
    BatchManager *batch_manager = new BatchManager(gpu_context, batch_size);

    this->OrganizeBatch(cur_exec_level, batch_manager);

    BatchData *im_batch_data = NULL;
    this->BeforeBatchProcess(cur_exec_level, im_batch_data);

    for (size_t batch_id = 0; batch_id < batch_manager->GetBatchNum();
         ++batch_id) {
      BatchSpec batch_spec = batch_manager->GetBatch(batch_id);
      this->PrintProgress(cur_exec_level, batch_id,
                          batch_manager->GetBatchNum(), batch_spec);

      this->PrepareBatch(cur_exec_level, im_batch_data, batch_spec);

      this->ExecuteBatch(cur_exec_level, batch_spec);

      this->CollectCount(cur_exec_level);

      if (this->NeedSearchNext(cur_exec_level)) {
        size_t next_exec_level = this->GetNextLevel(cur_exec_level);
        this->ProcessLevel(next_exec_level);
      }

      this->ReleaseBatch(cur_exec_level, im_batch_data, batch_spec);
    }

    this->AfterBatchProcess(cur_exec_level, im_batch_data);
    delete batch_manager;
    batch_manager = NULL;
  }

  virtual size_t GetNextLevel(size_t cur_exec_level) {
    return cur_exec_level + 1;
  }

  virtual bool IsProcessOver(size_t cur_exec_level) = 0;

  virtual size_t GetLevelBatchSize(size_t cur_exec_level) = 0;

  virtual void BeforeBatchProcess(size_t cur_exec_level,
                                  BatchData *&im_batch_data) = 0;
  virtual void AfterBatchProcess(size_t cur_exec_level,
                                 BatchData *&im_batch_data) = 0;
  virtual void PrepareBatch(size_t cur_exec_level, BatchData *im_batch_data,
                            BatchSpec batch_spec) = 0;
  virtual void ReleaseBatch(size_t cur_exec_level, BatchData *im_batch_data,
                            BatchSpec batch_spec) = 0;

  virtual bool NeedSearchNext(size_t cur_exec_level) = 0;

  virtual void ExecuteBatch(size_t cur_exec_level, BatchSpec batch_spec) = 0;

  virtual void OrganizeBatch(size_t cur_exec_level,
                             BatchManager *batch_manager) = 0;

  virtual void PrintProgress(size_t cur_exec_level, size_t batch_id,
                             size_t batch_num, BatchSpec batch_spec) = 0;

  virtual void CollectCount(size_t cur_exec_level) = 0;

 protected:
  WorkContext *wctx_;
  Task *task_;
};

#endif