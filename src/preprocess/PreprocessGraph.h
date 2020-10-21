#ifndef __PREPROCESS_GRAPH_H__
#define __PREPROCESS_GRAPH_H__

#include "CPUGraph.h"
#include "Meta.h"
#include "TimeMeasurer.h"

class PreprocessGraph {
 public:
  PreprocessGraph(Graph* graph) : graph_(graph) {}
  ~PreprocessGraph() {}

  void Preprocess() {
    // remove dangling nodes
    // self loops
    // parallel edges
    TimeMeasurer timer;
    timer.StartTimer();
    std::cout << "start preprocess..." << std::endl;
    size_t vertex_count = graph_->GetVertexCount();
    size_t edge_count = graph_->GetEdgeCount();
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();

    auto vertex_ecnt = new uintE[vertex_count + 1];
    memset(vertex_ecnt, 0, sizeof(uintE) * (vertex_count + 1));
    for (uintV u = 0; u < vertex_count; ++u) {
      for (auto j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        auto v = cols[j];
        bool parallel_edge = (j > row_ptrs[u] && v == cols[j - 1]);
        bool self_loop = u == v;
        if (!parallel_edge && !self_loop) {
          vertex_ecnt[u]++;
        }
      }
    }
    auto nrow_ptrs = new uintE[vertex_count + 1];
    uintE nedge_count = 0;
    for (uintV u = 0; u < vertex_count; ++u) {
      nrow_ptrs[u] = nedge_count;
      nedge_count += vertex_ecnt[u];
    }
    nrow_ptrs[vertex_count] = nedge_count;
    delete[] vertex_ecnt;
    vertex_ecnt = NULL;

    auto ncols = new uintV[nedge_count];
    for (uintV u = 0; u < vertex_count; ++u) {
      auto uoff = nrow_ptrs[u];
      for (uintE j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        auto v = cols[j];
        bool parallel_edge = j > row_ptrs[u] && v == cols[j - 1];
        bool self_loop = u == v;
        if (!parallel_edge && !self_loop) {
          ncols[uoff++] = v;
        }
      }
    }
    edge_count = nedge_count;
    std::swap(row_ptrs, nrow_ptrs);
    std::swap(cols, ncols);
    delete[] nrow_ptrs;
    nrow_ptrs = NULL;
    delete[] ncols;
    ncols = NULL;

    auto new_vertex_ids = new uintV[vertex_count];
    uintV max_vertex_id = 0;
    for (uintV u = 0; u < vertex_count; ++u) {
      if (row_ptrs[u] == row_ptrs[u + 1]) {
        new_vertex_ids[u] = vertex_count;
      } else {
        new_vertex_ids[u] = max_vertex_id++;
        row_ptrs[new_vertex_ids[u]] = row_ptrs[u];
      }
    }
    for (uintE j = 0; j < edge_count; ++j) {
      cols[j] = new_vertex_ids[cols[j]];
    }
    delete[] new_vertex_ids;
    new_vertex_ids = NULL;
    vertex_count = max_vertex_id;
    row_ptrs[vertex_count] = edge_count;

    timer.EndTimer();
    std::cout << "finish preprocess, time="
              << timer.GetElapsedMicroSeconds() / 1000.0 << "ms"
              << ", now vertex_count=" << vertex_count
              << ",edge_count=" << edge_count << std::endl;

    graph_->SetVertexCount(vertex_count);
    graph_->SetEdgeCount(edge_count);
    graph_->SetRowPtrs(row_ptrs);
    graph_->SetCols(cols);
  }

 private:
  Graph* graph_;
};

#endif
