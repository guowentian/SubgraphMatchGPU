#ifndef __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_VERIFY_COMPUTE_CUH__
#define __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_VERIFY_COMPUTE_CUH__

#include "CPUFilter.h"
#include "CPUGraph.h"
#include "CPUIntersection.h"

namespace Light {
static void VerifyItpCompute(
    size_t d_partition_id, size_t dfs_id, size_t cur_exec_level,
    CudaContext *context, ImData *im_data, ImDataDevHolder *im_data_holder,
    DevLazyTraversalPlan *dev_plan, DevGraphPartition *graph_partition,
    LazyTraversalPlan *plan, GraphDevTracker *graph_dev_tracker,
    TrackPartitionedGraph *cpu_relation, InterPartTask *task, long long &ans) {
  auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];
  auto op = exec_seq[cur_exec_level].first;
  uintV cur_level = exec_seq[cur_exec_level].second;

  auto &materialized_vertices =
      plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];
  auto &computed_unmaterialized_vertices =
      plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                             [cur_exec_level];
  auto &d_instances = im_data->GetInstances();
  size_t path_num = d_instances[materialized_vertices[0]]->GetSize();

  size_t pattern_vertex_count = plan->GetVertexCount();
  std::vector<uintV *> h_instances(pattern_vertex_count, NULL);
  for (auto u : materialized_vertices) {
    h_instances[u] = new uintV[path_num];
    DToH(h_instances[u], d_instances[u]->GetArray(), path_num);
  }

  uintP *h_partition_ids = cpu_relation->GetVertexPartitionMap();
  DevConnType *h_backward_conn = new DevConnType[pattern_vertex_count];
  DevCondArrayType *h_computed_order =
      new DevCondArrayType[pattern_vertex_count];

  DToH(h_backward_conn, dev_plan->GetBackwardConnectivity()->GetArray(),
       pattern_vertex_count);
  DToH(h_computed_order, dev_plan->GetComputedOrdering()->GetArray(),
       pattern_vertex_count);

  DevConnType *conn = h_backward_conn + cur_level;
  DevCondArrayType *cond = h_computed_order + cur_level;
  std::cout << "conn:";
  for (size_t i = 0; i < conn->GetCount(); ++i) {
    uintV u = conn->Get(i);
    std::cout << " " << u;
  }
  std::cout << std::endl;
  std::cout << "cond:";
  for (size_t i = 0; i < cond->GetCount(); ++i) {
    auto p = cond->Get(i);
    std::cout << " (" << p.GetOperator() << "," << p.GetOperand() << ")";
  }
  std::cout << std::endl;

  uintE *row_ptrs = cpu_relation->GetRowPtrs();
  uintV *cols = cpu_relation->GetCols();

  std::vector<size_t> cur_candidates_counts;
  std::vector<uintV> cur_candidates;
  for (size_t i = 0; i < path_num; ++i) {
    uintV M[kMaxQueryVerticesNum];
    for (auto u : materialized_vertices) {
      M[u] = h_instances[u][i];
    }

    std::vector<uintV> intersect_levels;
    for (size_t j = 0; j < conn->GetCount(); ++j) {
      uintV u = conn->Get(j);
      intersect_levels.push_back(u);
    }

    std::vector<uintV> intersect_result;
    MWayIntersect<HOME_MADE>(M, row_ptrs, cols, intersect_levels,
                             intersect_result);

    std::vector<uintV> cur_path_candidates;
    for (auto v : intersect_result) {
      bool valid = CheckDuplicate(M, v, intersect_levels, h_partition_ids);

      // condition
      if (valid) {
        for (size_t i = 0; i < cond->GetCount() && valid; ++i) {
          uintV u = cond->Get(i).GetOperand();
          CondOperator op = cond->Get(i).GetOperator();
          if (op == LESS_THAN) {
            valid = (v < M[u]);
          } else if (op == LARGER_THAN) {
            valid = (v > M[u]);
          } else {
            valid = (v != M[u]);
          }
        }
      }

      if (valid) {
        cur_path_candidates.push_back(v);
      }
    }

    cur_candidates_counts.push_back(cur_path_candidates.size());
    cur_candidates.insert(cur_candidates.end(), cur_path_candidates.begin(),
                          cur_path_candidates.end());
  }

  // prefix scan
  std::vector<size_t> cur_candidates_offsets(path_num + 1);
  for (size_t i = 0, sum = 0; i <= path_num; ++i) {
    cur_candidates_offsets[i] = sum;
    if (i < path_num) sum += cur_candidates_counts[i];
  }

  // verify
  {
    auto &d_candidates = im_data->GetCandidates();
    auto &d_candidates_offsets = im_data->GetCandidatesOffsets();

    assert(d_candidates[cur_level]->GetSize() == cur_candidates.size());
    uintV *h_cur_candidates = new uintV[cur_candidates.size()];
    size_t *h_cur_candidates_offsets = new size_t[path_num + 1];

    DToH(h_cur_candidates_offsets, d_candidates_offsets[cur_level]->GetArray(),
         path_num + 1);
    DToH(h_cur_candidates, d_candidates[cur_level]->GetArray(),
         cur_candidates.size());

    for (size_t i = 0; i <= path_num; ++i) {
      if (cur_candidates_offsets[i] != h_cur_candidates_offsets[i]) {
        std::cout << cur_candidates_offsets[i] << ","
                  << h_cur_candidates_offsets[i] << std::endl;
      }
      assert(cur_candidates_offsets[i] == h_cur_candidates_offsets[i]);
    }
    for (size_t j = 0; j < cur_candidates.size(); ++j) {
      assert(cur_candidates[j] == h_cur_candidates[j]);
    }

    delete[] h_cur_candidates;
    h_cur_candidates = NULL;
    delete[] h_cur_candidates_offsets;
    h_cur_candidates_offsets = NULL;
  }

  delete[] h_backward_conn;
  h_backward_conn = NULL;
  delete[] h_computed_order;
  h_computed_order = NULL;
  for (auto u : materialized_vertices) {
    delete[] h_instances[u];
    h_instances[u] = NULL;
  }
}
}  // namespace Light

#endif