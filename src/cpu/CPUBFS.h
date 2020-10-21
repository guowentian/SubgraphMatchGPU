#ifndef __CPU_BFS_H__
#define __CPU_BFS_H__

#include "CPURBFS.h"

// as a comparison for CPUReuseBFS
// without reusing the cached intersection result
class CPUBfs : public CPUReuseBFS {
 public:
  CPUBfs(ReuseTraversalPlan* plan, Graph* graph, size_t thread_num,
         size_t buffer_limit)
      : CPUReuseBFS(plan, graph, thread_num, buffer_limit),
        plan_(plan),
        graph_(graph) {}
  virtual ~CPUBfs() {}

 protected:
  virtual void Process(size_t level, unsigned long long& ans) {
    if (level == plan_->GetVertexCount()) {
      size_t last_level = level - 1;
      size_t path_num = instances_[last_level].size();
      ans += path_num;
      return;
    }
    size_t vertex_count = graph_->GetVertexCount();
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();

    size_t remaining_buffer = buffer_limit_ - buffer_size_;
    size_t remaining_levels_num = plan_->GetVertexCount() - level;
    size_t buffer_budget = remaining_buffer / remaining_levels_num;

    if (level == 0) {
      // memory allocate
      instances_[level].Init(memory_allocator_, vertex_count);
      next_offsets_[level].Init(memory_allocator_, vertex_count + 1);
      buffer_size_ += instances_[level].GetMemoryCost() +
                      next_offsets_[level].GetMemoryCost();

      // assign
      parallel_for(uintV u = 0; u < vertex_count; ++u) {
        instances_[level][u] = u;
      }

      // next level
      Process(level + 1, ans);

      // clear up
      buffer_size_ -= instances_[level].GetMemoryCost() +
                      next_offsets_[level].GetMemoryCost();
      next_offsets_[level].Release(memory_allocator_);
      instances_[level].Release(memory_allocator_);

    } else {
      size_t path_num = instances_[level - 1].size();
      if (path_num == 0) {
        return;
      }

      phase_profiler_->StartTimer("organize_batch", 0);
      std::vector<BatchSpec> batches;
      // 2*sizeof(size_t): batch_children_offset + parent_indices_
      // sizeof(uintV): instances_
      size_t per_child_consume = sizeof(uintV) + 2 * sizeof(size_t);
      OrganizeBatch(level, buffer_budget, per_child_consume, batches);
      phase_profiler_->EndTimer("organize_batch", 0);

      size_t last_level = level - 1;
      for (size_t batch_id = 0; batch_id < batches.size(); ++batch_id) {
        size_t batch_size = batches[batch_id].GetSize();
#if defined(BATCH_PROFILE)
        std::cout << "level=" << level << ",batch_id=" << batch_id
                  << ",batch_num=" << batches.size()
                  << ",buffer_size=" << buffer_size_ / 1000.0 / 1000.0 / 1000.0
                  << "GB" << std::endl;
#endif

        phase_profiler_->StartTimer("get_children_offset", 0);
        // batch_children_offset indicates the upper bound of children count of
        // each partial instance
        Array<size_t> batch_children_count;
        Array<size_t> batch_children_offset;
        batch_children_count.Init(memory_allocator_, batch_size + 1);
        batch_children_offset.Init(memory_allocator_, batch_size + 1);
        GetLevelBatchChildrenOffset(level, batches[batch_id],
                                    batch_children_count.data(),
                                    batch_children_offset.data());
        phase_profiler_->EndTimer("get_children_offset", 0);

        phase_profiler_->StartTimer("memory_manage", 0);
        auto batch_intersect_result = new SegmentedArray<uintV>();
        auto batch_filter_result = new SegmentedArray<uintV>();
        batch_intersect_result->Init(memory_allocator_,
                                     batch_children_offset.data(), batch_size);
        batch_filter_result->Init(memory_allocator_,
                                  batch_children_offset.data(), batch_size);
        phase_profiler_->EndTimer("memory_manage", 0);

        phase_profiler_->StartTimer("check_constraints", 0);
        CheckConstraints(level, batches[batch_id], *batch_intersect_result,
                         *batch_filter_result);
        phase_profiler_->EndTimer("check_constraints", 0);

        phase_profiler_->StartTimer("materialize", 0);
        Materialize(level, batches[batch_id], batch_children_count.data(),
                    *batch_filter_result);
        phase_profiler_->EndTimer("materialize", 0);

        // update memory infor.
        size_t total_batch_memory_size =
            batch_children_count.GetMemoryCost() +
            batch_children_offset.GetMemoryCost() +
            batch_filter_result->GetMemoryCost() +
            batch_intersect_result->GetMemoryCost() +
            instances_[level].GetMemoryCost() +
            parent_indices_[level].GetMemoryCost();
        buffer_size_ += total_batch_memory_size;

        // next level
        Process(level + 1, ans);

        phase_profiler_->StartTimer("memory_manage", 0);
        // clear up
        buffer_size_ -= total_batch_memory_size;
        parent_indices_[level].Release(memory_allocator_);
        instances_[level].Release(memory_allocator_);
        batch_filter_result->Release(memory_allocator_);
        delete batch_filter_result;
        batch_filter_result = NULL;
        batch_intersect_result->Release(memory_allocator_);
        delete batch_intersect_result;
        batch_intersect_result = NULL;

        batch_children_offset.Release(memory_allocator_);
        batch_children_count.Release(memory_allocator_);
        phase_profiler_->EndTimer("memory_manage", 0);
      }
    }
  }

  virtual void CheckConstraints(size_t level, BatchSpec& batch_spec,
                                SegmentedArray<uintV>& batch_intersect_result,
                                SegmentedArray<uintV>& batch_filter_result) {
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();

    size_t levels_num = plan_->GetVertexCount();
    size_t last_level = level - 1;

    size_t batch_start = batch_spec.GetStart();
    size_t batch_end = batch_spec.GetEnd();
    size_t batch_size = batch_spec.GetSize();

    parallel_for(size_t batch_pos = batch_start; batch_pos < batch_end;
                 ++batch_pos) {
      std::vector<uintV> M(levels_num, kMaxuintV);
      RecoverInstance(last_level, batch_pos, instances_, parent_indices_, M);

      std::vector<uintV*> intersect_arrays;
      std::vector<size_t> intersect_arrays_size;
      for (auto u : ordered_conn_[level]) {
        auto v = M[u];
        intersect_arrays.push_back(cols + row_ptrs[v]);
        intersect_arrays_size.push_back(row_ptrs[v + 1] - row_ptrs[v]);
      }
      size_t intersect_num = intersect_arrays.size();
      // By right, we can just use a thread-local vector to store the
      // intersection result, which could be more efficient. But that would be
      // unfair comparison for GPU_RBFS, as GPU_RBFS always store the
      // intersection result into a global memory region
      auto intersect_result =
          batch_intersect_result.GetArray(batch_pos - batch_start);
      size_t intersect_result_size = 0;
      MWayIntersect<CPUIntersectMethod::HOME_MADE>(
          intersect_arrays.data(), intersect_arrays_size.data(), intersect_num,
          intersect_result, intersect_result_size);

      auto thread_filter_result =
          batch_filter_result.GetArray(batch_pos - batch_start);
      size_t thread_filter_result_size = 0;
      for (size_t i = 0; i < intersect_result_size; ++i) {
        auto candidate = intersect_result[i];
        if (CheckCondition(M.data(), candidate, ordered_cond_[level]) &&
            !CheckEquality(M.data(), level, candidate)) {
          thread_filter_result[thread_filter_result_size++] = candidate;
        }
      }
      batch_filter_result.SetSize(batch_pos - batch_start,
                                  thread_filter_result_size);
      assert(thread_filter_result_size <=
             batch_filter_result.GetSizeBound(batch_pos - batch_start));
    }
  }

  virtual void Materialize(size_t level, BatchSpec& batch_spec,
                           size_t* compact_batch_offset,
                           SegmentedArray<uintV>& batch_result) {
    size_t last_level = level - 1;
    size_t batch_start = batch_spec.GetStart();
    size_t batch_end = batch_spec.GetEnd();
    size_t batch_size = batch_spec.GetSize();

    compact_batch_offset[batch_size] = ParallelUtils::ParallelPlusScan(
        batch_result.GetSizes().data(), compact_batch_offset, batch_size);

    // allocate memory
    size_t total_instances_count = compact_batch_offset[batch_size];
    instances_[level].Init(memory_allocator_, total_instances_count);
    parent_indices_[level].Init(memory_allocator_, total_instances_count);

    // assign
    parallel_for(size_t i = 0; i < batch_size; ++i) {
      for (size_t j = compact_batch_offset[i]; j < compact_batch_offset[i + 1];
           ++j) {
        instances_[level][j] =
            batch_result.GetArray(i)[j - compact_batch_offset[i]];
        parent_indices_[level][j] = batch_start + i;
      }
    }
  }

 private:
  ReuseTraversalPlan* plan_;
  Graph* graph_;
};

#endif
