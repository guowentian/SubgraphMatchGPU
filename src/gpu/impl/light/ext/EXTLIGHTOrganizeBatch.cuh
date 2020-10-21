#ifndef __EXTERNAL_LIGHT_ORGANIZE_BATCH_CUH__
#define __EXTERNAL_LIGHT_ORGANIZE_BATCH_CUH__

#include "LIGHTOrganizeBatch.cuh"

namespace Light {
// for each path to perform intersection,
// estimate the cost by taking the memory cost of the subgraph needed,
// plus the cost to compute the new level (including the temporary memory
// and so on)
static void EXTEstimateIntersectCost(LightWorkContext *wctx, uintV cur_level,
                                     size_t path_num, size_t parent_factor,
                                     size_t children_factor,
                                     DeviceArray<size_t> *&children_count,
                                     DeviceArray<size_t> *&children_cost) {
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;
  auto graph_dev_tracker = wctx->graph_dev_tracker;

  im_data_holder->GatherImData(im_data, context);
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();

  uintE *row_ptrs = graph_dev_tracker->GetGraphRowPtrs()->GetArray();
  auto backward_conn = dev_plan->GetBackwardConnectivity()->GetArray();

  children_count = new DeviceArray<size_t>(path_num, context);
  children_cost = new DeviceArray<size_t>(path_num, context);
  size_t *children_count_data = children_count->GetArray();
  size_t *children_cost_data = children_cost->GetArray();

  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        // collect partial instance
        uintV M[kMaxQueryVerticesNum] = {kMaxuintV};
        auto &conn = backward_conn[cur_level];
        for (size_t i = 0; i < conn.GetCount(); ++i) {
          uintV u = conn.Get(i);
          M[u] = d_seq_instances[u][index];
        }

        size_t max_children_count =
            ThreadEstimatePathCount<MIN>(conn, M, row_ptrs);
        size_t subgraph_size = ThreadEstimatePathCount<ADD>(conn, M, row_ptrs);

        children_count_data[index] = max_children_count;
        children_cost_data[index] = sizeof(uintV) * subgraph_size +
                                    parent_factor +
                                    children_factor * max_children_count;
      },
      path_num, context);
}

}  // namespace Light

#endif