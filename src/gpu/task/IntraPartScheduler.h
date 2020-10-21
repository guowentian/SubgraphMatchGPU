#ifndef __GPU_TASK_INTRA_PARTITION_SCHEDULER_CUH__
#define __GPU_TASK_INTRA_PARTITION_SCHEDULER_CUH__

#include <cassert>
#include <vector>

#include "CPUGraph.h"
#include "Meta.h"
#include "SpinLock.h"
#include "Task.h"

class IntraPartScheduler {
 public:
  IntraPartScheduler(PartitionedGraph* graph, size_t dev_num)
      : graph_(graph), dev_num_(dev_num) {
    partition_num_ = graph_->GetPartitionNum();

    lock_.Init();
    intra_part_start_off_.resize(partition_num_, 0);
    intra_part_end_off_.resize(partition_num_, 0);
    intra_part_batch_size_.resize(partition_num_, 0);
    intra_part_ongoing_count_.resize(partition_num_, 0);

    PrepareSchedule();
  }

  // Get the partition that has unfinished tasks
  // Return true if such a partition exist
  bool GetPartitionForTask(size_t& partition_id) {
    bool ret = false;
    lock_.Lock();
    for (size_t p = 0; p < partition_num_; ++p) {
      if (intra_part_start_off_[p] < intra_part_end_off_[p]) {
        intra_part_ongoing_count_[p]++;
        partition_id = p;
        ret = true;
        break;
      }
    }
    lock_.Unlock();
    return ret;
  }
  // get a task from the given h_partition_id
  bool GetTask(size_t h_partition_id, IntraPartTask& task) {
    bool ret = false;
    lock_.Lock();
    if (intra_part_start_off_[h_partition_id] <
        intra_part_end_off_[h_partition_id]) {
      ret = true;
      task.start_offset_ = intra_part_start_off_[h_partition_id];
      task.end_offset_ = std::min(
          intra_part_end_off_[h_partition_id],
          (uintV)(task.start_offset_ + intra_part_batch_size_[h_partition_id]));
      intra_part_start_off_[h_partition_id] = task.end_offset_;
    }
    lock_.Unlock();
    return ret;
  }

  // Signify that this thread has finished tasks in h_partition_id.
  // Return true if no threads are processing tasks in h_partition_id.
  bool FinishPartition(size_t h_partition_id) {
    bool finish_all = false;
    lock_.Lock();
    intra_part_ongoing_count_[h_partition_id]--;
    if (intra_part_ongoing_count_[h_partition_id] == 0) {
      finish_all = true;
    }
    lock_.Unlock();
    return finish_all;
  }

  size_t GetPartitionNum() const { return partition_num_; }
  size_t GetDeviceNum() const { return dev_num_; }

 protected:
  void PrepareSchedule() {
    bool big_graph =
        graph_->GetVertexCount() > 1000000 || graph_->GetEdgeCount() > 5000000;
    // partition_units is the #work units each partition has
    size_t partition_units = 0;
    if (!big_graph) {
      // if it is a small graph_
      // set small partition units for each partition
      if (dev_num_ * 2 < partition_num_) {
        // each GPU has enough fine grained tasks at this time
        partition_units = 1;
      } else {
        // divide each graph_ partition for 2 gpus
        partition_units = 2;
      }
    } else {
      partition_units = 2 * dev_num_;
    }
    for (size_t p = 0; p < partition_num_; ++p) {
      size_t partition_vertex_count = graph_->GetPartition(p)->GetVertexCount();
      intra_part_end_off_[p] = partition_vertex_count;
      intra_part_batch_size_[p] =
          (partition_vertex_count + partition_units - 1) / partition_units;
    }
  }

 protected:
  PartitionedGraph* graph_;
  size_t dev_num_;
  size_t partition_num_;  // partition number of the CPU graph

  SpinLock lock_;
  std::vector<uintV> intra_part_start_off_;
  std::vector<uintV> intra_part_end_off_;
  std::vector<size_t> intra_part_batch_size_;
  std::vector<size_t> intra_part_ongoing_count_;
};

#endif