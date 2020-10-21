#ifndef __HYBRID_LIGHT_GPU_PIPELINE_COMPONENT_DIRECT_COUNT_CUH__
#define __HYBRID_LIGHT_GPU_PIPELINE_COMPONENT_DIRECT_COUNT_CUH__
#include "DevLazyTraversalPlan.cuh"
#include "GPUUtil.cuh"

#include "GPUBinSearch.cuh"
#include "Intersect.cuh"

#include "LazyTraversalPlan.h"
#include "Meta.h"

namespace Light {
HOST_DEVICE bool CanDirectCount(const QueryType query_type) {
  if (Q0 <= query_type && query_type <= Q11 && query_type != Q4 &&
      query_type != Q9) {
    return true;
  }
  return false;
}

DEVICE size_t ThreadDirectCount(const size_t* buds_instances_count,
                                const QueryType query_type) {
  size_t ret = 0;
  switch (query_type) {
    case Q0:
      ret = buds_instances_count[2];
      break;
    case Q1:
      ret = buds_instances_count[3];
      break;
    case Q2:
      assert(buds_instances_count[1] == buds_instances_count[3]);
      if (buds_instances_count[1] > 0) {
        ret =
            (size_t)buds_instances_count[1] * (buds_instances_count[1] - 1) / 2;
      }
      break;
    case Q3:
      ret = buds_instances_count[3];
      break;
    case Q5:
      assert(buds_instances_count[3] <= buds_instances_count[1] &&
             buds_instances_count[3] <= buds_instances_count[5]);
      ret = (size_t)buds_instances_count[1] * buds_instances_count[3] *
            buds_instances_count[5];
      if (ret > 0) {
        ret = ret -
              (size_t)buds_instances_count[3] *
                  (buds_instances_count[1] - 1 + buds_instances_count[5] - 1) -
              (size_t)buds_instances_count[3] * buds_instances_count[3];
      }
      break;
    case Q6:
      if (buds_instances_count[4] > 0 && buds_instances_count[2] > 0) {
        ret = (size_t)buds_instances_count[4] * buds_instances_count[2] -
              buds_instances_count[2];
      }
      break;
    case Q7:
      ret = buds_instances_count[4];
      break;
    case Q8:
      ret = buds_instances_count[4];
      break;
    case Q10:
      assert(buds_instances_count[2] == buds_instances_count[4]);
      if (buds_instances_count[2] > 0) {
        size_t n = buds_instances_count[2];
        ret = n * (n - 1) / 2;
      }
      break;
    case Q11:
      ret = buds_instances_count[5];
      break;
    default:
      assert(false);
  }
  return ret;
}
static size_t LIGHTDirectCount(LightWorkContext* wctx, size_t cur_exec_level) {
  auto context = wctx->context;
  auto plan = wctx->plan;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;

  auto& d_instances = im_data->GetInstances();
  im_data_holder->GatherImData(im_data, context);
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();
  auto d_seq_candidates = im_data_holder->GetSeqCandidates()->GetArray();
  auto d_seq_candidates_indices =
      im_data_holder->GetSeqCandidatesIndices()->GetArray();
  auto d_seq_candidates_offsets =
      im_data_holder->GetSeqCandidatesOffsets()->GetArray();

  auto d_unmaterialized_vertices =
      dev_plan->GetComputedUnmaterializedVertices()->GetArray() +
      cur_exec_level;

  auto& materialized_vertices = plan->GetMaterializedVertices()[cur_exec_level];
  size_t path_num = d_instances[materialized_vertices[0]]->GetSize();
  DeviceArray<size_t> total_count(1, context);
  QueryType query_type = plan->GetQuery()->GetQueryType();

  auto k = [=] DEVICE(int index) {
    size_t candidates_counts[kMaxQueryVerticesNum] = {0};
    for (size_t i = 0; i < d_unmaterialized_vertices->GetCount(); ++i) {
      uintV u = d_unmaterialized_vertices->Get(i);
      size_t p = d_seq_candidates_indices[u][index];
      size_t count =
          d_seq_candidates_offsets[u][p + 1] - d_seq_candidates_offsets[u][p];
      candidates_counts[u] = count;
    }
    return ThreadDirectCount(candidates_counts, query_type);
  };

  GpuUtils::Reduce::TransformReduce(k, path_num, total_count.GetArray(),
                                    context);
  size_t ret;
  DToH(&ret, total_count.GetArray(), 1);
  return ret;
}

}  // namespace Light

#endif