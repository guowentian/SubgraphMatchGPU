#ifndef __GPU_COMMON_LOAD_SUBGRAPH_H__
#define __GPU_COMMON_LOAD_SUBGRAPH_H__

#include <future>
#include <thread>

#include "CPUGraph.h"
#include "Task.h"

static uintV* LoadSubgraph(Graph* cpu_relation, size_t load_vertex_count,
                           size_t load_edge_count, uintV* load_vertex_ids,
                           uintE* load_row_ptrs, size_t thread_num) {
  uintE* graph_row_ptrs = cpu_relation->GetRowPtrs();
  uintV* graph_cols = cpu_relation->GetCols();

  uintV* load_cols = new uintV[load_edge_count];
  if (thread_num == 1) {
    size_t off = 0;
    for (size_t i = 0; i < load_vertex_count; ++i) {
      uintV u = load_vertex_ids[i];
      for (uintE j = graph_row_ptrs[u]; j < graph_row_ptrs[u + 1]; ++j) {
        uintV v = graph_cols[j];
        load_cols[off] = v;
        ++off;
      }
    }
  } else {
#pragma omp parallel for num_threads(thread_num)
    for (size_t e = 0; e < load_edge_count; ++e) {
      size_t index =
          std::upper_bound(load_row_ptrs, load_row_ptrs + load_vertex_count + 1,
                           e) -
          load_row_ptrs;
      --index;
      uintV u = load_vertex_ids[index];
      size_t v_off = e - load_row_ptrs[index];
      uintV v = graph_cols[graph_row_ptrs[u] + v_off];
      load_cols[e] = v;
    }
  }
  return load_cols;
}

#endif