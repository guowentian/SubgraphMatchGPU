#ifndef __GPU_TASK_ONE_PARTITION_SCHEDULER_CUH__
#define __GPU_TASK_ONE_PARTITION_SCHEDULER_CUH__

#include "IntraPartScheduler.h"

class VertexSetScheduler {
 public:
  VertexSetScheduler(uintV* vertices, size_t vertex_count, size_t partition_num)
      : vertices_(vertices),
        vertex_count_(vertex_count),
        partition_num_(partition_num),
        cur_offset_(0) {
    lock_.Init();
    PrepareSchedule();
  }
  bool GetTask(VertexSetTask& task) {
    lock_.Lock();
    size_t start_off = cur_offset_;
    size_t count = std::min(batch_size_, vertex_count_ - cur_offset_);
    cur_offset_ += count;
    lock_.Unlock();
    task.Set(VERTEX_SET, NULL, vertices_ + start_off, count, 0xffffffff);
    return count > 0;
  }

 protected:
  void PrepareSchedule() {
    // Suppose each GPU has 5000 threads, ensure each GPU thread to take 10
    // edges to increase the utility for one task
    batch_size_ = 5000 * 20;
    size_t min_partition_num = 2 * partition_num_;
    size_t per_partition_size =
        (vertex_count_ + min_partition_num - 1) / min_partition_num;
    batch_size_ = std::min(batch_size_, per_partition_size);
  }

  SpinLock lock_;
  // workload
  uintV* vertices_;
  size_t vertex_count_;

  // schedule
  size_t batch_size_;
  size_t partition_num_;  // gpu/cpu thread number
  size_t cur_offset_;
};

#endif