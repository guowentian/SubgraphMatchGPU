#ifndef __GRAPH_PARTITION_H__
#define __GRAPH_PARTITION_H__

#include "Meta.h"
#include "ParallelUtils.h"

class GraphPartition {
 public:
  GraphPartition(uintP* vertex_partition_map, uintP partition_id,
                 size_t g_vertex_count, uintE* g_row_ptrs, uintV* g_cols)
      : partition_id_(partition_id), g_vertex_ids_(NULL) {
    BuildIntraPartition(vertex_partition_map, partition_id, g_vertex_count,
                        g_row_ptrs, g_cols);
  }
  GraphPartition(bool* exists, size_t g_vertex_count, uintE* g_row_ptrs,
                 uintV* g_cols, uintP partition_id, bool parallel = false)
      : partition_id_(partition_id), g_vertex_ids_(NULL) {
    if (parallel) {
      ParallelBuildSubgraphPartition(exists, g_vertex_count, g_row_ptrs,
                                     g_cols);
    } else {
      BuildSubgraphPartition(exists, g_vertex_count, g_row_ptrs, g_cols);
    }
  }

  ~GraphPartition() {
    delete[] row_ptrs_;
    row_ptrs_ = NULL;
    delete[] cols_;
    cols_ = NULL;
    if (g_vertex_ids_) {
      delete[] g_vertex_ids_;
      g_vertex_ids_ = NULL;
    }
  }

  void BuildIntraPartition(uintP* vertex_partition_map, uintP partition_id,
                           size_t g_vertex_count, uintE* g_row_ptrs,
                           uintV* g_cols) {
    // only keep the edges (u,v) where both u and v are in this partition
    // vertex_partition_map: [vertex_id, partition_id]
    auto relabel = new uintV[g_vertex_count];
    auto intra_edge_count = new uintE[g_vertex_count + 1];
    size_t intra_vertex_count = 0;
    size_t total_intra_edge_count = 0;
    for (uintV u = 0; u < g_vertex_count; ++u) {
      if (vertex_partition_map[u] == partition_id) {
        relabel[u] = intra_vertex_count++;
        auto new_u = relabel[u];
        uintE u_intra_edge_count = 0;
        for (uintE j = g_row_ptrs[u]; j < g_row_ptrs[u + 1]; ++j) {
          auto v = g_cols[j];
          if (vertex_partition_map[v] == partition_id) {
            // intra-partition edge
            u_intra_edge_count++;
          }
        }
        intra_edge_count[new_u] = u_intra_edge_count;
        total_intra_edge_count += u_intra_edge_count;
      }
    }

    row_ptrs_ = new uintE[intra_vertex_count + 1];
    uintE prefix = 0;
    for (uintV u = 0; u < g_vertex_count; ++u) {
      if (vertex_partition_map[u] == partition_id) {
        // as new_u is relabeled in sequence
        auto new_u = relabel[u];
        row_ptrs_[new_u] = prefix;
        prefix += intra_edge_count[new_u];
      }
    }
    row_ptrs_[intra_vertex_count] = prefix;

    cols_ = new uintV[total_intra_edge_count];
    for (uintV u = 0; u < g_vertex_count; ++u) {
      if (vertex_partition_map[u] == partition_id) {
        auto new_u = relabel[u];
        auto offset_start = row_ptrs_[new_u];
        uintE u_intra_edge_count = 0;
        for (uintE j = g_row_ptrs[u]; j < g_row_ptrs[u + 1]; ++j) {
          auto v = g_cols[j];
          if (vertex_partition_map[v] == partition_id) {
            // intra-partition edge
            uintV new_v = relabel[v];
            cols_[offset_start + u_intra_edge_count] = new_v;
            u_intra_edge_count++;
          }
        }
      }
    }
    vertex_count_ = intra_vertex_count;
    edge_count_ = total_intra_edge_count;

    delete[] intra_edge_count;
    intra_edge_count = NULL;

    g_vertex_ids_ = new uintV[intra_vertex_count];
    for (uintV v = 0; v < g_vertex_count; ++v) {
      if (vertex_partition_map[v] == partition_id) {
        g_vertex_ids_[relabel[v]] = v;
      }
    }

    delete[] relabel;
    relabel = NULL;
  }

  void BuildSubgraphPartition(bool* exists, size_t g_vertex_count,
                              uintE* g_row_ptrs, uintV* g_cols) {
    // keep the edges (u,v) where u is in current partition
    // keep all global vertices
    // exists: indicates a set of vertices in current partition
    auto intra_edge_count = new uintE[g_vertex_count + 1];
    size_t total_intra_edge_count = 0;
    for (uintV u = 0; u < g_vertex_count; ++u) {
      uintE u_intra_edge_count =
          exists[u] ? g_row_ptrs[u + 1] - g_row_ptrs[u] : 0;
      intra_edge_count[u] = u_intra_edge_count;
      total_intra_edge_count += u_intra_edge_count;
    }
    vertex_count_ = g_vertex_count;
    edge_count_ = total_intra_edge_count;

    row_ptrs_ = intra_edge_count;
    uintE prefix = 0;
    for (uintV u = 0; u <= vertex_count_; ++u) {
      auto tmp = row_ptrs_[u];
      row_ptrs_[u] = prefix;
      prefix += tmp;
    }
    cols_ = new uintV[edge_count_];
    for (uintV u = 0; u < vertex_count_; ++u) {
      if (exists[u]) {
        memcpy(cols_ + row_ptrs_[u], g_cols + g_row_ptrs[u],
               sizeof(uintV) * (g_row_ptrs[u + 1] - g_row_ptrs[u]));
      }
    }
  }

  // build in parallel
  void ParallelBuildSubgraphPartition(bool* exists, size_t g_vertex_count,
                                      uintE* g_row_ptrs, uintV* g_cols) {
    // keep the edges (u,v) where u is in current partition
    // keep all global vertices
    // exists: indicates a set of vertices in current partition
    auto intra_edge_count = new uintE[g_vertex_count + 1];
    parallel_for(uintV u = 0; u < g_vertex_count; ++u) {
      uintE u_intra_edge_count =
          exists[u] ? g_row_ptrs[u + 1] - g_row_ptrs[u] : 0;
      intra_edge_count[u] = u_intra_edge_count;
    }
    intra_edge_count[g_vertex_count] = 0;

    vertex_count_ = g_vertex_count;
    row_ptrs_ = new uintE[vertex_count_ + 1];
    ParallelUtils::ParallelPlusScan(intra_edge_count, row_ptrs_,
                                    vertex_count_ + 1);

    delete[] intra_edge_count;
    intra_edge_count = NULL;

    edge_count_ = row_ptrs_[vertex_count_];
    cols_ = new uintV[edge_count_];
    parallel_for(uintV u = 0; u < vertex_count_; ++u) {
      if (exists[u]) {
        memcpy(cols_ + row_ptrs_[u], g_cols + g_row_ptrs[u],
               sizeof(uintV) * (g_row_ptrs[u + 1] - g_row_ptrs[u]));
      }
    }
  }

  uintP GetPartitionId() const { return partition_id_; }
  size_t GetVertexCount() const { return vertex_count_; }
  size_t GetEdgeCount() const { return edge_count_; }
  uintE* GetRowPtrs() const { return row_ptrs_; }
  uintV* GetCols() const { return cols_; }
  uintV* GetGlobalVertexIds() const { return g_vertex_ids_; }
  void SetVertexCount(size_t vertex_count) { vertex_count_ = vertex_count; }
  void SetEdgeCount(size_t edge_count) { edge_count_ = edge_count; }
  void SetRowPtrs(uintE* row_ptrs) { row_ptrs_ = row_ptrs; }
  void SetCols(uintV* cols) { cols_ = cols; }

 protected:
  uintP partition_id_;
  size_t vertex_count_;
  size_t edge_count_;
  uintE* row_ptrs_;
  uintV* cols_;
  uintV*
      g_vertex_ids_;  // the mapping from local vertex ids to global vertex ids
};
#endif
