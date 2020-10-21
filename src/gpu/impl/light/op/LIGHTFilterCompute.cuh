#ifndef __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_FILTER_COMPUTE_CUH__
#define __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_FILTER_COMPUTE_CUH__

#include "LIGHTCommon.cuh"

#include "DevGraphPartition.cuh"
#include "Intersect.cuh"
#include "LazyTraversalPlan.h"

#include "CountProfiler.h"
#include "GPUProfiler.cuh"
#include "Iterator.cuh"
#include "SegmentReduce.cuh"

namespace Light {

// apply ordering constraints between filter_level and the materialized
// vertices. The result is written to <new_candidates_offsets, new_candidates>.
// path_iterator[i] returns the path id at the position i
// cond: the ordering to apply
template <typename PathIdIterator>
static void FilterCandidate(LightWorkContext* wctx, uintV filter_level,
                            PathIdIterator path_iterator, size_t path_num,
                            DevCondArrayType* cond,
                            DeviceArray<size_t>*& new_candidates_offsets,
                            DeviceArray<uintV>*& new_candidates) {
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;

  auto& d_instances = im_data->GetInstances();
  uintV* cur_candidates = im_data->GetCandidates()[filter_level]->GetArray();
  size_t* cur_candidates_offsets =
      im_data->GetCandidatesOffsets()[filter_level]->GetArray();
  size_t* cur_candidates_indices =
      im_data->GetCandidatesIndices()[filter_level]->GetArray();

  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();

  DeviceArray<size_t>* path_offsets =
      new DeviceArray<size_t>(path_num + 1, context);
  GpuUtils::Scan::TransformScan(
      [=] DEVICE(int index) {
        size_t path_id = path_iterator[index];
        size_t pos = cur_candidates_indices[path_id];
        size_t count =
            cur_candidates_offsets[pos + 1] - cur_candidates_offsets[pos];
        return count;
      },
      path_num, path_offsets->GetArray(), path_offsets->GetArray() + path_num,
      context);

  size_t total_children_count;
  DToH(&total_children_count, path_offsets->GetArray() + path_num, 1);
  assert((int)total_children_count == total_children_count);

  if (total_children_count == 0) {
    new_candidates = new DeviceArray<uintV>(0, context);
    new_candidates_offsets = new DeviceArray<size_t>(0, context);
  } else {
    DeviceArray<uintV>* children =
        new DeviceArray<uintV>(total_children_count, context);
    DeviceArray<bool>* bitmaps =
        new DeviceArray<bool>(total_children_count, context);
    uintV* children_data = children->GetArray();
    bool* bitmaps_data = bitmaps->GetArray();

    // LBS expand to enforce ordering constraints
    GpuUtils::LoadBalance::LBSTransform<MGPULaunchBoxVT1>(
        [=] DEVICE(int index, int seg, int rank) {
          size_t path_id = path_iterator[seg];
          size_t pos = cur_candidates_indices[path_id];
          uintV candidate = cur_candidates[cur_candidates_offsets[pos] + rank];

          // gather materialized pattern vertices
          uintV M[kMaxQueryVerticesNum] = {kMaxuintV};
          for (size_t i = 0; i < cond->GetCount(); ++i) {
            uintV u = cond->Get(i).GetOperand();
            M[u] = d_seq_instances[u][path_id];
          }

          // check conditions for candidate
          bool valid = ThreadCheckCondition(*cond, M, candidate);

          children_data[index] = candidate;
          bitmaps_data[index] = valid;
        },
        total_children_count, path_offsets->GetArray(), path_num, context);

    // filter, compact
    new_candidates = NULL;
    int compact_output_count;
    GpuUtils::Compact::Compact(children, total_children_count,
                               bitmaps->GetArray(), new_candidates,
                               compact_output_count, context);
    delete children;
    children = NULL;
    // CAUTION: it is possible that compact_output_count = 0

    DeviceArray<size_t>* new_candidates_count =
        new DeviceArray<size_t>(path_num, context);
    GpuUtils::SegReduce::TransformSegReduce(
        [=] DEVICE(int index) { return bitmaps_data[index] ? 1 : 0; },
        total_children_count, path_offsets->GetArray(), path_num,
        new_candidates_count->GetArray(), (size_t)0, context);
    delete bitmaps;
    bitmaps = NULL;

    new_candidates_offsets = new DeviceArray<size_t>(path_num + 1, context);
    GpuUtils::Scan::ExclusiveSum(new_candidates_count->GetArray(), path_num,
                                 new_candidates_offsets->GetArray(),
                                 new_candidates_offsets->GetArray() + path_num,
                                 context);
    delete new_candidates_count;
    new_candidates_count = NULL;
  }

  delete path_offsets;
  path_offsets = NULL;
}

static void FilterCompute(LightWorkContext* wctx, uintV filter_level) {
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;

  auto& d_instances = im_data->GetInstances();
  auto& d_candidates = im_data->GetCandidates();
  auto& d_candidates_offsets = im_data->GetCandidatesOffsets();
  auto& d_candidates_indices = im_data->GetCandidatesIndices();

  size_t path_num = d_candidates_indices[filter_level]->GetSize();
  assert(d_candidates[filter_level] != NULL);
  im_data_holder->GatherImData(im_data, context);

  auto filter_order = dev_plan->GetFilterOrdering()->GetArray() + filter_level;
  DeviceArray<size_t>* new_candidates_offsets = NULL;
  DeviceArray<uintV>* new_candidates = NULL;
  FilterCandidate(wctx, filter_level,
                  GpuUtils::Iterator::CountingIterator<size_t>(0), path_num,
                  filter_order, new_candidates_offsets, new_candidates);

  ReleaseIfExists(d_candidates[filter_level]);
  ReleaseIfExists(d_candidates_offsets[filter_level]);
  d_candidates[filter_level] = new_candidates;
  d_candidates_offsets[filter_level] = new_candidates_offsets;

  // CAUTION: Due to batching, d_candidates_indices[filter_level] now holds
  // the pointer to the whole original array.
  /// This write can destroy the content of the whole original array.
  // As the whole original array is no more used, this write should be fine.
  GpuUtils::Transform::Sequence(d_candidates_indices[filter_level]->GetArray(),
                                path_num, (size_t)0, context);
}

}  // namespace Light
#endif