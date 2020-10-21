#ifndef __GPU_COPROCESSING_TASK_H__
#define __GPU_COPROCESSING_TASK_H__

#include <atomic>
#include "GraphPartition.h"
#include "Meta.h"

enum TaskType { INTRA_PARTITION, INTER_PARTITION, VERTEX_SET, kTaskType };

struct Task {
  void Set(TaskType task_type, long long* ans, size_t d_partition_id) {
    task_type_ = task_type;
    ans_ = ans;
    d_partition_id_ = d_partition_id;
  }
  TaskType task_type_;
  // result
  long long* ans_;
  // gpu id
  size_t d_partition_id_;
};

struct IntraPartTask : Task {
  void Set(TaskType task_type, long long* ans, size_t d_partition_id,
           GraphPartition* cpu_partition, uintV start_offset,
           uintV end_offset) {
    Task::Set(task_type, ans, d_partition_id);
    ans_ = ans;

    d_partition_id_ = d_partition_id;
    cpu_partition_ = cpu_partition;

    start_offset_ = start_offset;
    end_offset_ = end_offset;
  }
  size_t GetVertexCount() const { return end_offset_ - start_offset_; }

  // workload
  // the range of vertex ids in cpu_partition_ [start_offset_, end_offset_)
  GraphPartition* cpu_partition_;
  uintV start_offset_;
  uintV end_offset_;
};

struct InterPartTask : Task {
  void Set(TaskType task_type, long long* ans, uintE start_offset,
           uintE end_offset, size_t d_partition_id) {
    Task::Set(task_type, ans, d_partition_id);
    ans_ = ans;
    start_offset_ = start_offset;
    end_offset_ = end_offset;
    d_partition_id_ = d_partition_id;
  }
  size_t GetEdgeCount() const { return end_offset_ - start_offset_; }

  // workload: the range of inter-partition edges [start_offset_,end_offset_)
  uintE start_offset_;
  uintE end_offset_;
};

struct VertexSetTask : Task {
  void Set(TaskType task_type, long long* ans, uintV* vertices,
           size_t vertex_count, size_t d_partition_id) {
    Task::Set(task_type, ans, d_partition_id);
    ans_ = ans;
    d_partition_id_ = d_partition_id;
    vertices_ = vertices;
    vertex_count_ = vertex_count;
  }
  size_t GetVertexCount() const { return vertex_count_; }

  // the vertex set
  uintV* vertices_;
  size_t vertex_count_;
};

#endif
