#ifndef __CPU_REUSEABLE_BFS_H__
#define __CPU_REUSEABLE_BFS_H__

#include <omp.h>
#include "CPUBFSUtils.h"
#include "CPUFilter.h"
#include "CPUGraph.h"
#include "CPUIntersection.h"
#include "CPUPatternMatch.h"
#include "CountProfiler.h"
#include "LinearMemoryAllocator.h"
#include "PhaseProfiler.h"
#include "ReuseTraversalPlan.h"
#include "TimeMeasurer.h"

class CPUReuseBFS : public CPUPatternMatch {
  public:
  CPUReuseBFS(ReuseTraversalPlan* plan, Graph* graph, size_t thread_num,
              size_t buffer_limit)
      : CPUPatternMatch(thread_num), plan_(plan), graph_(graph) {
    buffer_limit_ = buffer_limit;
    buffer_size_ = 0;

    plan_->GetOrderedConnectivity(ordered_conn_);
    plan_->GetOrderedOrdering(ordered_cond_);

    memory_allocator_ = new LinearMemoryAllocator(buffer_limit);

    size_t levels_num = plan_->GetVertexCount();
    instances_.resize(levels_num);
    parent_indices_.resize(levels_num);
    next_offsets_.resize(levels_num);
    level_batch_.resize(levels_num);
    cached_result_.resize(levels_num);

    phase_profiler_ = new PhaseProfiler(1);
    phase_profiler_->AddPhase("organize_batch");
    phase_profiler_->AddPhase("check_constraints");
    phase_profiler_->AddPhase("check_connectivity");
    phase_profiler_->AddPhase("intersect_time");
    phase_profiler_->AddPhase("check_ordering_non_equality");
    phase_profiler_->AddPhase("materialize");
    phase_profiler_->AddPhase("get_children_offset");
    phase_profiler_->AddPhase("memory_manage");

    count_profiler_ = new CountProfiler(thread_num_);
    count_profiler_->AddPhase("intersect_count");
    count_profiler_->AddPhase("reuse_count");
    count_profiler_->AddPhase("total_intersect_result_size");
    count_profiler_->AddPhase("reuse_intersect_result_size");
    count_profiler_->AddPhase("reuse_intersect_candidates_size");
  }
  virtual ~CPUReuseBFS() {
    assert(buffer_size_ == 0);
    delete memory_allocator_;
    memory_allocator_ = NULL;
    delete phase_profiler_;
    phase_profiler_ = NULL;
    delete count_profiler_;
    count_profiler_ = NULL;
  }

  virtual void Execute() {
    omp_set_num_threads(thread_num_);

    TimeMeasurer timer;
    timer.StartTimer();

    unsigned long long total_match_count = 0;
    this->Process(0, total_match_count);

    timer.EndTimer();

    this->SetTotalMatchCount(total_match_count);
    std::cout << "total_match_count=" << total_match_count
              << ", elapsed_time=" << timer.GetElapsedMicroSeconds() / 1000.0
              << "ms" << std::endl;
    phase_profiler_->Report((size_t)0);

    count_profiler_->ReportAgg("intersect_count");
    count_profiler_->ReportAgg("reuse_count");
    count_profiler_->ReportAgg("total_intersect_result_size");
    count_profiler_->ReportAgg("reuse_intersect_result_size");
    count_profiler_->ReportAgg("reuse_intersect_candidates_size");
#if defined(REUSE_PROFILE)
    std::cout << "reuse ratio="
              << count_profiler_->GetCount("reuse_count") * 1.0 /
                     count_profiler_->GetCount("intersect_count")
              << ", average result size per intersect="
              << count_profiler_->GetCount("total_intersect_result_size") *
                     1.0 / count_profiler_->GetCount("intersect_count")
              << std::endl;
    std::cout << "reuse intersect result size versus candidates size="
              << count_profiler_->GetCount("reuse_intersect_result_size") *
                     1.0 /
                     count_profiler_->GetCount(
                         "reuse_intersect_candidates_size")
              << ", average reuse intersect result size="
              << count_profiler_->GetCount("reuse_intersect_result_size") *
                     1.0 / count_profiler_->GetCount("reuse_count")
              << ", average reuse intersect candidates size="
              << count_profiler_->GetCount("reuse_intersect_candidates_size") *
                     1.0 / count_profiler_->GetCount("reuse_count")
              << std::endl;
#endif
  }

  protected:
  virtual void Process(size_t level, unsigned long long& ans) {
    if (level == plan_->GetVertexCount()) {
      size_t path_num = instances_[level - 1].size();
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
      // 4*sizeof(size_t): parent_indices_ + next_offsets_ +
      // batch_children_offset + batch_children_count
      // 3*sizeof(uintV): instances_ + batch_intersect_result +
      // batch_filter_result
      size_t per_child_consume = 3 * sizeof(uintV) + 4 * sizeof(size_t);
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
        batch_children_count.Init(memory_allocator_, batch_size);
        batch_children_offset.Init(memory_allocator_, batch_size + 1);
        GetLevelBatchChildrenOffset(level, batches[batch_id],
                                    batch_children_count.data(),
                                    batch_children_offset.data());
        phase_profiler_->EndTimer("get_children_offset", 0);

        phase_profiler_->StartTimer("memory_manage", 0);
        // batch_intersect_result is memory buffer to write the intermediate
        // result of intersection;
        // batch_filter_result is memory buffer to write the intermediate
        // result after checking non-equality and ordering constraint
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
        Materialize(level, batches[batch_id], *batch_intersect_result,
                    *batch_filter_result);
        phase_profiler_->EndTimer("materialize", 0);

        // update memory infor.
        size_t total_batch_memory_size =
            batch_children_count.GetMemoryCost() +
            batch_children_offset.GetMemoryCost() +
            batch_intersect_result->GetMemoryCost() +
            batch_filter_result->GetMemoryCost() +
            instances_[level].GetMemoryCost() +
            parent_indices_[level].GetMemoryCost() +
            next_offsets_[level].GetMemoryCost();
        buffer_size_ += total_batch_memory_size;

        // process next level
        Process(level + 1, ans);

        phase_profiler_->StartTimer("memory_manage", 0);
        // clear up
        buffer_size_ -= total_batch_memory_size;
        cached_result_[level] = NULL;

        next_offsets_[level].Release(memory_allocator_);
        parent_indices_[level].Release(memory_allocator_);
        instances_[level].Release(memory_allocator_);

        batch_filter_result->Release(memory_allocator_);
        batch_intersect_result->Release(memory_allocator_);
        delete batch_filter_result;
        batch_filter_result = NULL;
        delete batch_intersect_result;
        batch_intersect_result = NULL;

        batch_children_offset.Release(memory_allocator_);
        batch_children_count.Release(memory_allocator_);

        phase_profiler_->EndTimer("memory_manage", 0);
      }
    }
  }

  virtual void Materialize(size_t level, BatchSpec& batch_spec,
                           SegmentedArray<uintV>& batch_intersect_result,
                           SegmentedArray<uintV>& batch_filter_result) {
    size_t last_level = level - 1;
    size_t batch_start = batch_spec.GetStart();
    size_t batch_end = batch_spec.GetEnd();
    size_t batch_size = batch_spec.GetSize();

    size_t* batch_offset = next_offsets_[last_level].data() + batch_start;
    assert(batch_end <= next_offsets_[last_level].size());
    batch_offset[batch_size] = ParallelUtils::ParallelPlusScan(
        batch_filter_result.GetSizes().data(), batch_offset, batch_size);

    // allocate memory
    size_t total_instances_count = batch_offset[batch_size];
    instances_[level].Init(memory_allocator_, total_instances_count);
    parent_indices_[level].Init(memory_allocator_, total_instances_count);
    next_offsets_[level].Init(memory_allocator_, total_instances_count + 1);

    // assign
    parallel_for(size_t i = 0; i < batch_size; ++i) {
      for (size_t j = batch_offset[i]; j < batch_offset[i + 1]; ++j) {
        instances_[level][j] =
            batch_filter_result.GetArray(i)[j - batch_offset[i]];
        parent_indices_[level][j] = batch_start + i;
      }
    }
    level_batch_[last_level] = batch_spec;
    cached_result_[level] = &batch_intersect_result;
  }

  void CheckConstraints(size_t level, BatchSpec& batch_spec,
                        SegmentedArray<uintV>& batch_intersect_result,
                        SegmentedArray<uintV>& batch_filter_result) {
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();

    size_t levels_num = plan_->GetVertexCount();
    size_t last_level = level - 1;

    size_t batch_start = batch_spec.GetStart();
    size_t batch_end = batch_spec.GetEnd();
    size_t batch_size = batch_spec.GetSize();

    auto& reuse_conn_meta =
        plan_->GetLevelReuseIntersectPlan()[level].GetReuseConnectivityMeta();
    auto& separate_conn =
        plan_->GetLevelReuseIntersectPlan()[level].GetSeparateConnectivity();
    bool cache_required = plan_->GetCacheRequired(level);

    parallel_for(size_t batch_pos = batch_start; batch_pos < batch_end;
                 ++batch_pos) {
      std::vector<uintV> M(levels_num, kMaxuintV);
      RecoverInstance(last_level, batch_pos, instances_, parent_indices_, M);

      std::vector<uintV*> intersect_arrays;
      std::vector<size_t> intersect_arrays_size;
      bool find_cached_result;

      if (reuse_conn_meta.size() == 0) {
        // There is no reuse
        for (size_t i = 0; i < separate_conn.size(); ++i) {
          auto u = separate_conn[i];
          auto v = M[u];
          intersect_arrays.push_back(cols + row_ptrs[v]);
          intersect_arrays_size.push_back(row_ptrs[v + 1] - row_ptrs[v]);
        }
      } else {
        // It is possible to have reuse
        // If we can, find from the cached intersection result
        find_cached_result =
            FindCachedResult(level, batch_pos, M, reuse_conn_meta,
                             intersect_arrays, intersect_arrays_size);

        // for debug
        // bool find_cached_result = false;

        if (find_cached_result) {
          // from adjacent lists of separate vertices
          for (size_t i = 0; i < separate_conn.size(); ++i) {
            auto u = separate_conn[i];
            auto v = M[u];
            intersect_arrays.push_back(cols + row_ptrs[v]);
            intersect_arrays_size.push_back(row_ptrs[v + 1] - row_ptrs[v]);
          }
        } else {
          // if cannot find from the cached result,
          // we can only intersect each adjacent list
          for (auto u : ordered_conn_[level]) {
            auto v = M[u];
            intersect_arrays.push_back(cols + row_ptrs[v]);
            intersect_arrays_size.push_back(row_ptrs[v + 1] - row_ptrs[v]);
          }
        }
      }

      auto thread_intersect_result =
          batch_intersect_result.GetArray(batch_pos - batch_start);
      size_t thread_intersect_result_size = 0;

      // When cache_required=false and intersect_num=1, we can actually avoid
      // materialization. We do not use such an optimization to have fair
      // comparison with GPU_BFS, because GPU_BFS will always write intersection
      // result into global memory
      MWayIntersect<CPUIntersectMethod::HOME_MADE>(
          intersect_arrays.data(), intersect_arrays_size.data(),
          intersect_arrays.size(), thread_intersect_result,
          thread_intersect_result_size);
      batch_intersect_result.SetSize(batch_pos - batch_start,
                                     thread_intersect_result_size);
      assert(thread_intersect_result_size <=
             batch_intersect_result.GetSizeBound(batch_pos - batch_start));

#if defined(REUSE_PROFILE)
      size_t thread_id = ParallelUtils::GetParallelThreadId();
      count_profiler_->AddCount("intersect_count", thread_id, 1);
      count_profiler_->AddCount("total_intersect_result_size", thread_id,
                                thread_intersect_result_size);
      if (reuse_conn_meta.size() > 0) {
        if (find_cached_result) {
          count_profiler_->AddCount("reuse_count", thread_id, 1);
          count_profiler_->AddCount("reuse_intersect_result_size", thread_id,
                                    thread_intersect_result_size);
          size_t reuse_intersect_candidates_size = kMaxsize_t;
          for (auto u : ordered_conn_[level]) {
            auto v = M[u];
            size_t candidates_size = row_ptrs[v + 1] - row_ptrs[v];
            reuse_intersect_candidates_size =
                std::min(reuse_intersect_candidates_size, candidates_size);
          }
          count_profiler_->AddCount("reuse_intersect_candidates_size",
                                    thread_id, reuse_intersect_candidates_size);
        }
      }

#endif

      auto thread_filter_result =
          batch_filter_result.GetArray(batch_pos - batch_start);
      size_t thread_filter_result_size = 0;

      // check non-equality and ordering
      for (size_t i = 0; i < thread_intersect_result_size; ++i) {
        auto candidate = thread_intersect_result[i];
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

  // Given partial instance M, at level,
  // find the cached intersection result according to reuse_conn_meta
  bool FindCachedResult(size_t level, size_t batch_pos, std::vector<uintV>& M,
                        std::vector<ReuseConnMeta>& reuse_conn_meta,
                        std::vector<uintV*>& intersect_arrays,
                        std::vector<size_t>& intersect_arrays_size) {
    size_t last_level = level - 1;
    for (size_t i = 0; i < reuse_conn_meta.size(); ++i) {
      std::vector<uintV> new_M(M);
      for (auto conn_u : reuse_conn_meta[i].GetConnectivity()) {
        auto source_conn_u = reuse_conn_meta[i].GetInvertedMapping()[conn_u];
        new_M[source_conn_u] = M[conn_u];
      }

      // Given new_M, track the intersection result
      uintV* intersect_array = NULL;
      size_t intersect_array_size = 0;
      // LocateCache(new_M, reuse_conn_meta[i].GetSourceVertex(),
      // intersect_array,
      //            intersect_array_size);

      assert(reuse_conn_meta[i].GetAlignedVertex() <=
             reuse_conn_meta[i].GetSourceVertex());
      FastLocateCache(new_M, batch_pos, last_level,
                      reuse_conn_meta[i].GetAlignedVertex(),
                      reuse_conn_meta[i].GetSourceVertex(), intersect_array,
                      intersect_array_size);

      // because of pipelining, some cached result may not be available
      if (intersect_array == NULL) {
        intersect_arrays.clear();
        intersect_arrays_size.clear();
        return false;
      }

#if defined(DEBUG)
      VertexMapping& inverted_mapping = reuse_conn_meta[i].GetInvertedMapping();
      bool equal = true;
      for (auto conn_u : reuse_conn_meta[i].GetConnectivity()) {
        auto source_conn_u = inverted_mapping[conn_u];
        if (conn_u != source_conn_u) {
          equal = false;
        }
      }
      // must find
      if (equal) {
        assert(intersect_array);
      }
      // check whether the cached result is correct
      auto row_ptrs = graph_->GetRowPtrs();
      auto cols = graph_->GetCols();
      std::vector<uintV> intersect_result;
      MWayIntersect<CPUIntersectMethod::HOME_MADE>(
          new_M.data(), row_ptrs, cols,
          ordered_conn_[reuse_conn_meta[i].GetSourceVertex()],
          intersect_result);
      if (intersect_result.size() != intersect_array_size) {
        printf("intersect_result.size()=%d, intersect_array_size=%d\n",
               (int)intersect_result.size(), (int)intersect_array_size);
      }
      assert(intersect_result.size() == intersect_array_size);
      for (size_t j = 0; j < intersect_array_size; ++j) {
        assert(intersect_array[j] == intersect_result[j]);
      }

#endif

      intersect_arrays.push_back(intersect_array);
      intersect_arrays_size.push_back(intersect_array_size);
    }
    return true;
  }

  // memory_budget: the amount of memory reserved for this level
  // per_child_consume: the memory cost for each child
  virtual size_t OrganizeBatch(size_t level, size_t memory_budget,
                               size_t per_child_consume,
                               size_t* children_offset,
                               std::vector<BatchSpec>& batches) {
    size_t last_level = level - 1;
    size_t path_num = instances_[last_level].size();
    size_t children_count_per_batch = memory_budget / per_child_consume;
    size_t total_children_count = children_offset[path_num];
    size_t batch_num = (total_children_count + children_count_per_batch - 1) /
                       children_count_per_batch;
    for (size_t batch_id = 0, cur_parent_id = 0; batch_id < batch_num;
         ++batch_id) {
      size_t batch_end = (batch_id + 1) * children_count_per_batch;
      size_t p = std::lower_bound(children_offset + cur_parent_id,
                                  children_offset + path_num, batch_end) -
                 (children_offset + cur_parent_id);
      assert(p);
      batches.push_back(BatchSpec(cur_parent_id, cur_parent_id + p));
      cur_parent_id += p;
    }
    assert(batches[batch_num - 1].GetEnd() == path_num);
  }

  virtual size_t OrganizeBatch(size_t level, size_t memory_budget,
                               size_t per_child_consume,
                               std::vector<BatchSpec>& batches) {
    size_t last_level = level - 1;
    size_t path_num = instances_[last_level].size();

    Array<size_t> children_count;
    Array<size_t> children_offset;
    children_count.Init(memory_allocator_, path_num);
    children_offset.Init(memory_allocator_, path_num + 1);
    BatchSpec batch_spec(0, path_num);

    GetLevelBatchChildrenOffset(level, batch_spec, children_count.data(),
                                children_offset.data());

    OrganizeBatch(level, memory_budget, per_child_consume,
                  children_offset.data(), batches);

    children_offset.Release(memory_allocator_);
    children_count.Release(memory_allocator_);
  }

  void GetLevelBatchChildrenOffset(size_t level, BatchSpec& batch_spec,
                                   size_t* children_count,
                                   size_t* children_offset) {
    size_t last_level = level - 1;
    size_t levels_num = plan_->GetVertexCount();
    size_t path_num = batch_spec.GetSize();
    auto row_ptrs = graph_->GetRowPtrs();

    parallel_for(size_t i = 0; i < path_num; ++i) {
      std::vector<uintV> M(levels_num, kMaxuintV);
      RecoverInstance(last_level, batch_spec.GetStart() + i, instances_,
                      parent_indices_, M);
      uintE width = kMaxuintE;
      for (auto& u : ordered_conn_[level]) {
        auto v = M[u];
        width = std::min(width, row_ptrs[v + 1] - row_ptrs[v]);
      }
      children_count[i] = width;
    }

    children_offset[path_num] = ParallelUtils::ParallelPlusScan(
        children_count, children_offset, path_num);
  }

  template <typename T>
  static void RecoverInstance(size_t level, size_t pos,
                              LayeredArray<T>& instances,
                              LayeredArray<size_t>& parent_indices,
                              std::vector<T>& M) {
    M[level] = instances[level][pos];
    size_t cur_level = level;
    size_t cur_pos = pos;
    while (cur_level) {
      size_t prev_pos = parent_indices[cur_level][cur_pos];
      --cur_level;
      cur_pos = prev_pos;
      M[cur_level] = instances[cur_level][cur_pos];
    }
  }

 private:
  void FastLocateCache(std::vector<uintV>& M, size_t cur_index,
                       size_t cur_level, size_t aligned_level,
                       size_t target_level, uintV*& res_array,
                       size_t& res_size) {
    // go from cur_level to aligned_level
    while (cur_level > aligned_level) {
      cur_index = parent_indices_[cur_level][cur_index];
      --cur_level;
    }

    size_t search_array_start_index;
    size_t search_array_size;
    if (cur_level == 0) {
      search_array_start_index = 0;
      search_array_size = instances_[0].size();
    } else {
      cur_index = parent_indices_[cur_level][cur_index];
      search_array_start_index = next_offsets_[cur_level - 1][cur_index];
      search_array_size = next_offsets_[cur_level - 1][cur_index + 1] -
                          next_offsets_[cur_level - 1][cur_index];
    }

    assert(target_level > 0);
    assert(aligned_level <= target_level);
    assert(cur_level >= aligned_level);

    // go from aligned_level to cur_level
    size_t prev_find_index = cur_index;
    while (cur_level < target_level) {
      uintV* search_array =
          instances_[cur_level].data() + search_array_start_index;
      size_t cur_pos =
          std::lower_bound(search_array, search_array + search_array_size,
                           M[cur_level]) -
          search_array;
      size_t cur_index = search_array_start_index + cur_pos;

      bool f = false;
      if (cur_pos < search_array_size &&
          search_array[cur_pos] == M[cur_level]) {
        if (level_batch_[cur_level].GetStart() <= cur_index &&
            cur_index < level_batch_[cur_level].GetEnd()) {
          f = true;
        }
      }

      if (f) {
        search_array_start_index = next_offsets_[cur_level][cur_index];
        search_array_size = next_offsets_[cur_level][cur_index + 1] -
                            next_offsets_[cur_level][cur_index];
        prev_find_index = cur_index;

      } else {
        // fail to find, return res_array as NULL
        res_array = NULL;
        res_size = 0;
        return;
      }

      ++cur_level;
    }

    assert(prev_find_index >= level_batch_[target_level - 1].GetStart());
    size_t cached_offset =
        prev_find_index - level_batch_[target_level - 1].GetStart();
    assert(cached_offset <= level_batch_[target_level - 1].GetSize());
    res_array = cached_result_[target_level]->GetArray(cached_offset);
    res_size = cached_result_[target_level]->GetSize(cached_offset);
  }
  // starting from level 0, move to target_level, by the partial instances
  // indicated in M
  void LocateCache(std::vector<uintV>& M, size_t target_level,
                   uintV*& res_array, size_t& res_size) {
    assert(target_level > 0);
    size_t search_array_size = instances_[0].size();
    // The index of which search_array resides in the corresponding array.
    // Need to track this as it may lead to inactive batch
    size_t search_array_start_index = 0;
    size_t prev_find_index = 0;
    for (size_t cur_level = 0; cur_level < target_level; ++cur_level) {
      auto search_array =
          instances_[cur_level].data() + search_array_start_index;
      size_t cur_pos =
          std::lower_bound(search_array, search_array + search_array_size,
                           M[cur_level]) -
          search_array;
      size_t cur_index = search_array_start_index + cur_pos;

      bool f = false;
      // find the partial instance
      if (cur_pos < search_array_size &&
          search_array[cur_pos] == M[cur_level]) {
        // in the range of active batch
        if (level_batch_[cur_level].GetStart() <= cur_index &&
            cur_index < level_batch_[cur_level].GetEnd()) {
          f = true;
        }
      }

      if (f) {
        search_array_start_index = next_offsets_[cur_level][cur_index];
        search_array_size = next_offsets_[cur_level][cur_index + 1] -
                            next_offsets_[cur_level][cur_index];
        prev_find_index = cur_index;
      } else {
        // fail to find, return res_array as NULL
        res_array = NULL;
        res_size = 0;
        return;
      }
    }

    assert(prev_find_index >= level_batch_[target_level - 1].GetStart());
    size_t cached_offset =
        prev_find_index - level_batch_[target_level - 1].GetStart();
    assert(cached_offset <= level_batch_[target_level - 1].GetSize());
    res_array = cached_result_[target_level]->GetArray(cached_offset);
    res_size = cached_result_[target_level]->GetSize(cached_offset);
  }

  private:
  ReuseTraversalPlan* plan_;
  Graph* graph_;

  protected:
  size_t buffer_limit_;
  size_t buffer_size_;

  LinearMemoryAllocator* memory_allocator_;

  LayeredArray<uintV> instances_;

  // For each instance in each level, record the position of the parent.
  // This can be used to track the path and recover the partial instances.
  LayeredArray<size_t> parent_indices_;

  // For each instance in each level, record the range of expanded children in
  // the next level. This is used to LocateCache
  LayeredArray<size_t> next_offsets_;
  // For each level, record the currently active batch that is expanded.
  // This is necessary because for the inactive batch, the corresponding entries
  // in next_offsets_ are invalid
  std::vector<BatchSpec> level_batch_;

  // The intersection result for level is aligned with the path number of
  // level-1. That is cached_result_[level].size() = instances_[level-1].size().
  std::vector<SegmentedArray<uintV>*> cached_result_;

  AllConnType ordered_conn_;
  AllCondType ordered_cond_;

  PhaseProfiler* phase_profiler_;
  CountProfiler* count_profiler_;
};

#endif
