#ifndef __GPU_TASK_INTER_PARTITION_SCHEDULER_H__
#define __GPU_TASK_INTER_PARTITION_SCHEDULER_H__

#include <cassert>
#include <vector>

#include "CPUGraph.h"
#include "Meta.h"
#include "SpinLock.h"
#include "Task.h"

class InterPartScheduler {
 public:
  InterPartScheduler(TrackPartitionedGraph* cpu_rel, size_t dev_num,
                     size_t cpu_thread_num)
      : cpu_relation_(cpu_rel),
        dev_num_(dev_num),
        cpu_thread_num_(cpu_thread_num) {
    lock_.Init();
    total_edges_count_ = cpu_relation_->GetInterPartitionEdgesCount();
    cur_edge_offset_ = 0;

    PrepareSchedule();
  }

  bool GetTask(InterPartTask& task, bool is_cpu) {
    size_t batch_size = is_cpu ? cpu_batch_size_ : gpu_batch_size_;
    lock_.Lock();
    uintE start_offset = cur_edge_offset_;
    size_t count = std::min(batch_size, total_edges_count_ - cur_edge_offset_);
    cur_edge_offset_ += count;
    lock_.Unlock();
    task.Set(INTER_PARTITION, NULL, start_offset, (uintE)(start_offset + count),
             0xffffffff);
    return count > 0;
  }

 protected:
  void PrepareSchedule() {
    // Ensure each task can feed 8 edges for each CPU thread
    cpu_batch_size_ = 4;

    // Suppose each GPU has 5000 threads, ensure each GPU thread to take 10
    // edges to increase the utility for one task
    gpu_batch_size_ = 5000 * 20;

    // In case gpu_batch_size_ is set too big, ensure the workload is divided
    // into at least min_partition_num tasks
    size_t min_partition_num = 2 * dev_num_;
    size_t per_partition_size =
        (total_edges_count_ + min_partition_num - 1) / min_partition_num;
    gpu_batch_size_ = std::min(gpu_batch_size_, per_partition_size);
  }

 protected:
  TrackPartitionedGraph* cpu_relation_;
  size_t dev_num_;
  size_t cpu_thread_num_;

  SpinLock lock_;
  size_t total_edges_count_;
  size_t cur_edge_offset_;
  size_t gpu_batch_size_;
  size_t cpu_batch_size_;
};

#endif