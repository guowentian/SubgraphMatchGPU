#ifndef __LIGHT_ORGANIZE_BATCH_CUH__
#define __LIGHT_ORGANIZE_BATCH_CUH__

#include "BatchManager.cuh"
#include "LIGHTCommon.cuh"
#include "LIGHTDirectCount.cuh"

namespace Light {
// for each path, estimate the expanded children count after
// the intersection.
static void BuildIntersectChildrenCount(LightWorkContext *wctx, uintV cur_level,
                                        size_t path_num,
                                        DeviceArray<size_t> *&children_count) {
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;
  auto graph_partition = wctx->graph_partition;

  im_data_holder->GatherImData(im_data, context);
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();

  auto &d_instances = im_data->GetInstances();
  uintE *row_ptrs = graph_partition->GetRowPtrs()->GetArray();
  auto backward_conn = dev_plan->GetBackwardConnectivity()->GetArray();

  children_count = new DeviceArray<size_t>(path_num, context);
  size_t *children_count_data = children_count->GetArray();

  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        uintV M[kMaxQueryVerticesNum] = {kMaxuintV};
        auto &conn = backward_conn[cur_level];
        for (size_t i = 0; i < conn.GetCount(); ++i) {
          uintV u = conn.Get(i);
          M[u] = d_seq_instances[u][index];
        }
        uintV pivot_level = ThreadChoosePivotLevel(conn, M, row_ptrs);
        uintV pivot_vertex = M[pivot_level];
        size_t count = row_ptrs[pivot_vertex + 1] - row_ptrs[pivot_vertex];
        children_count_data[index] = count;
      },
      path_num, context);
}

/// for organize batch.
// estimate the children count given the previous candidate set
static void BuildGatherChildrenCount(LightWorkContext *wctx,
                                     uintV materialize_level, size_t path_num,
                                     DeviceArray<size_t> *&children_count) {
  auto im_data = wctx->im_data;
  auto context = wctx->context;
  size_t *cur_candidates_indices =
      im_data->GetCandidatesIndices()[materialize_level]->GetArray();
  size_t *cur_candidates_offsets =
      im_data->GetCandidatesOffsets()[materialize_level]->GetArray();

  children_count = new DeviceArray<size_t>(path_num, context);
  size_t *children_count_data = children_count->GetArray();

  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        size_t p = cur_candidates_indices[index];
        size_t count =
            cur_candidates_offsets[p + 1] - cur_candidates_offsets[p];
        children_count_data[index] = count;
      },
      path_num, context);
}

static void EstimateCountMemoryCost(LightWorkContext *wctx,
                                    size_t cur_exec_level,
                                    DeviceArray<size_t> *&children_count,
                                    size_t &parent_factor,
                                    size_t &children_factor) {
  auto context = wctx->context;
  auto plan = wctx->plan;
  auto im_data = wctx->im_data;

  auto &exec_seq = plan->GetExecuteOperations();
  auto &materialized_vertices = plan->GetMaterializedVertices()[cur_exec_level];
  auto &computed_unmaterialized_vertices =
      plan->GetComputedUnmaterializedVertices()[cur_exec_level];
  size_t path_num =
      im_data->GetInstances()[materialized_vertices[0]]->GetSize();

  bool allow_direct_count = plan->GetIntraPartitionPlanCompressLevel() ==
                            LazyTraversalCompressLevel::COMPRESS_LEVEL_SPECIAL;
  QueryType query_type = plan->GetQuery()->GetQueryType();
  if (allow_direct_count && CanDirectCount(query_type)) {
    children_count = new DeviceArray<size_t>(path_num, context);
    parent_factor = sizeof(size_t);
    children_factor = 0;
  } else {
    if (computed_unmaterialized_vertices.size() == 1) {
      // TODO: children_count is actually not needed here
      children_count = new DeviceArray<size_t>(path_num, context);
      parent_factor = sizeof(size_t);
      children_factor = 0;
    } else {
      BuildGatherChildrenCount(wctx, computed_unmaterialized_vertices[0],
                               path_num, children_count);
      parent_factor = sizeof(size_t);
      children_factor = sizeof(size_t);
    }
  }
}

static void EstimateComputeMemoryCost(std::vector<LazyTraversalEntry> &exec_seq,
                                      size_t cur_exec_level,
                                      size_t &parent_factor,
                                      size_t &children_factor) {
  // parent_factor
  // d_candidates_offsets, d_candidates_indices
  //
  // children_factor:
  // d_candidates
  //
  parent_factor = sizeof(size_t) * 2;
  children_factor = sizeof(uintV);

  // temporary memory cost for Intersect
  size_t temporary_parent_factor = sizeof(size_t) * 2;
  size_t temporary_children_factor = sizeof(bool) + sizeof(uintV);

  // organizeBatch in next level
  temporary_parent_factor += sizeof(size_t) * 3;

  size_t remaining_exec_levels = exec_seq.size() - cur_exec_level;
  parent_factor +=
      std::ceil(1.0 * temporary_parent_factor / remaining_exec_levels);
  children_factor +=
      std::ceil(1.0 * temporary_children_factor / remaining_exec_levels);
}

static void EstimateMaterializeMemoryCost(
    std::vector<LazyTraversalEntry> &exec_seq, VTGroup &materialized_vertices,
    VTGroup &computed_unmaterialized_vertices, size_t cur_exec_level,
    size_t &parent_factor, size_t &children_factor) {
  // parent_factor:
  //
  // children_factor:
  // children, materialized_vertices.size(), unmaterialized_vertices.size()
  parent_factor = 0;
  children_factor =
      sizeof(uintV) * (materialized_vertices.size() +
                       computed_unmaterialized_vertices.size() + 1);

  // path_offsets,
  size_t temporary_parent_factor = sizeof(size_t);
  // parents_indices
  // compact_output (output)
  // bitmaps
  // organizeBatch in next level
  size_t temporary_children_factor =
      sizeof(size_t) * 2 + sizeof(bool) + sizeof(size_t) * 3;

  size_t remaining_levels_num = exec_seq.size() - cur_exec_level;
  parent_factor +=
      std::ceil(1.0 * temporary_parent_factor / remaining_levels_num);
  children_factor +=
      std::ceil(1.0 * temporary_children_factor / remaining_levels_num);
}

static void EstimateFilterComputeMemoryCost(
    std::vector<LazyTraversalEntry> &exec_seq, size_t cur_exec_level,
    size_t &parent_factor, size_t &children_factor) {
  // parent_factor:
  // candidates_offsets
  //
  // children_factor:
  // candidates
  parent_factor = sizeof(size_t);
  children_factor = sizeof(uintV);

  // temporary_parent_factor:
  // path_offsets
  // organizeBatch in next level
  size_t temporary_parent_factor = sizeof(size_t) + sizeof(size_t) * 3;
  // temporary_children_factor:
  // bitmaps, children,
  size_t temporary_children_factor = sizeof(bool) + sizeof(uintV);

  size_t remaining_levels_num = exec_seq.size() - cur_exec_level;
  parent_factor +=
      std::ceil(1.0 * temporary_parent_factor / remaining_levels_num);
  children_factor +=
      std::ceil(1.0 * temporary_children_factor / remaining_levels_num);
}

static void EstimateComputeCountMemoryCost(
    std::vector<LazyTraversalEntry> &exec_seq, size_t cur_exec_level,
    size_t &parent_factor, size_t &children_factor) {
  // temporary_parent_factor
  // output_count
  //
  assert(cur_exec_level == exec_seq.size() - 1);
  parent_factor = sizeof(size_t);
  children_factor = 0;
}

static void EstimateComputePathCountMemoryCost(
    std::vector<LazyTraversalEntry> &exec_seq, size_t cur_exec_level,
    size_t &parent_factor, size_t &children_factor) {
  // parent_factor
  // children_offset
  //
  // temporary_parent_factor
  // output_count
  //
  size_t remaining_levels_num = exec_seq.size() - cur_exec_level;
  parent_factor =
      sizeof(size_t) + std::ceil(sizeof(size_t) * 1.0 / remaining_levels_num);
}
}  // namespace Light

#endif