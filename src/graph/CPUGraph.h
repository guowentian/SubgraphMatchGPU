#ifndef __CPU_GRAPH_H__
#define __CPU_GRAPH_H__

#include <vector>
#include "AbstractGraph.h"
#include "GraphIO.h"
#include "GraphPartition.h"

class Graph : public AbstractGraph {
 public:
  Graph(std::string& filename, bool directed) : AbstractGraph(directed) {
    GraphIO::ReadDataFile(filename, directed, vertex_count_, edge_count_,
                          row_ptrs_, cols_);
  }
  // for testing
  Graph(std::vector<std::vector<uintV>>& data) : AbstractGraph(false) {
    GraphIO::ReadFromVector(data, vertex_count_, edge_count_, row_ptrs_, cols_);
  }
  ~Graph() {}

  uintE* GetRowPtrs() const { return row_ptrs_; }
  uintV* GetCols() const { return cols_; }

  void SetRowPtrs(uintE* row_ptrs) { row_ptrs_ = row_ptrs; }
  void SetCols(uintV* cols) { cols_ = cols; }

 protected:
  uintE* row_ptrs_;
  uintV* cols_;
};

class PartitionedGraph : public Graph {
 public:
  PartitionedGraph(std::string& data_filename, bool directed,
                   std::string& partition_filename, size_t partition_num)
      : Graph(data_filename, directed),
        partition_num_(partition_num),
        local_vertex_ids_(NULL) {
    vertex_partition_map_ = new uintP[vertex_count_];
    GraphIO::ReadPartitionFile(partition_filename, partition_num, vertex_count_,
                               vertex_partition_map_);
    CreatePartitions();
#if defined(PROFILE)
    size_t intra_partition_edges_count = 0;
    for (size_t pid = 0; pid < partitions_.size(); ++pid) {
      auto p = partitions_[pid];
      std::cout << "partition_id=" << pid
                << ",vertex_count=" << p->GetVertexCount()
                << ",edge_count=" << p->GetEdgeCount() << std::endl;
      intra_partition_edges_count += p->GetEdgeCount();
    }
    std::cout << "intra_partition_edge_rate="
              << intra_partition_edges_count * 1.0 / edge_count_ << std::endl;
#endif
  }
  // for testing
  PartitionedGraph(std::vector<std::vector<uintV>>& data,
                   std::vector<uintP>& partition_maps, size_t partition_num)
      : Graph(data), partition_num_(partition_num), local_vertex_ids_(NULL) {
    vertex_partition_map_ = new uintP[vertex_count_];
    for (uintV u = 0; u < vertex_count_; ++u) {
      vertex_partition_map_[u] = partition_maps[u];
      assert(partition_maps[u] < partition_num_);
    }
    CreatePartitions();
  }

  ~PartitionedGraph() {
    delete[] vertex_partition_map_;
    vertex_partition_map_ = NULL;
    for (size_t p = 0; p < partitions_.size(); ++p) {
      if (partitions_[p]) {
        delete partitions_[p];
        partitions_[p] = NULL;
      }
    }
    if (local_vertex_ids_) {
      delete[] local_vertex_ids_;
      local_vertex_ids_ = NULL;
    }
  }

  void ReleasePartition(size_t partition_id) {
    delete partitions_[partition_id];
    partitions_[partition_id] = NULL;
  }

  uintP* GetVertexPartitionMap() const { return vertex_partition_map_; }
  size_t GetPartitionNum() const { return partition_num_; }
  GraphPartition* GetPartition(size_t p) { return partitions_[p]; }
  uintP GetVertexToPartitionId(uintV u) {
    return u < vertex_count_ ? vertex_partition_map_[u] : partition_num_;
  }
  uintV* GetLocalVertexIds() const { return local_vertex_ids_; }

 protected:
  size_t partition_num_;
  uintP* vertex_partition_map_;
  std::vector<GraphPartition*> partitions_;
  // The mapping from global vertex ids to local vertex ids in each intra
  // partition
  uintV* local_vertex_ids_;

 private:
  void CreatePartitions() {
    assert(vertex_partition_map_ != NULL);
    partitions_.resize(partition_num_);
    for (size_t p = 0; p < partition_num_; ++p) {
      partitions_[p] = new GraphPartition(vertex_partition_map_, p,
                                          vertex_count_, row_ptrs_, cols_);
    }

    local_vertex_ids_ = new uintV[vertex_count_];
    for (uintP p = 0; p < partition_num_; ++p) {
      uintV* g_vertex_ids = partitions_[p]->GetGlobalVertexIds();
      size_t intra_vertex_count = partitions_[p]->GetVertexCount();
      for (uintV i = 0; i < intra_vertex_count; ++i) {
        local_vertex_ids_[g_vertex_ids[i]] = i;
      }
    }
  }
};

class TrackPartitionedGraph : public PartitionedGraph {
 public:
  TrackPartitionedGraph(std::string& data_filename, bool directed,
                        std::string& partition_filename, size_t partition_num)
      : PartitionedGraph(data_filename, directed, partition_filename,
                         partition_num) {
    TrackInterPartitionEdges();
  }
  // for testing
  TrackPartitionedGraph(std::vector<std::vector<uintV>>& data,
                        std::vector<uintP>& partition_maps,
                        size_t partition_num)
      : PartitionedGraph(data, partition_maps, partition_num) {
    TrackInterPartitionEdges();
  }

  ~TrackPartitionedGraph() {
    delete[] inter_row_ptrs_;
    inter_row_ptrs_ = NULL;
    delete[] inter_cols_;
    inter_cols_ = NULL;
  }

  size_t GetInterPartitionEdgesCount() const { return inter_edges_count_; }
  uintE* GetInterRowPtrs() const { return inter_row_ptrs_; }
  uintV* GetInterCols() const { return inter_cols_; }

  double GetCrossPartitionEdgeRate() const {
    return 1.0 * inter_edges_count_ / this->GetEdgeCount();
  }

 protected:
  uintE* inter_row_ptrs_;
  uintV* inter_cols_;
  size_t inter_edges_count_;

 private:
  void TrackInterPartitionEdges() {
    assert(this->GetVertexPartitionMap() != NULL);
    auto offset = new uintE[vertex_count_ + 1];
    memset(offset, 0, sizeof(uintE) * (vertex_count_ + 1));
    for (uintV u = 0; u < vertex_count_; ++u) {
      auto u_pid = GetVertexToPartitionId(u);
      for (uintE j = row_ptrs_[u]; j < row_ptrs_[u + 1]; ++j) {
        auto v = cols_[j];
        auto v_pid = GetVertexToPartitionId(v);
        if (u_pid != v_pid) {
          offset[u]++;
        }
      }
    }
    uintE inter_sum = 0;
    for (uintV i = 0; i <= vertex_count_; ++i) {
      auto tmp = offset[i];
      offset[i] = inter_sum;
      inter_sum += tmp;
    }
    inter_row_ptrs_ = new uintE[vertex_count_ + 1];
    memcpy(inter_row_ptrs_, offset, sizeof(uintE) * (vertex_count_ + 1));
    inter_cols_ = new uintV[inter_sum];
    for (uintV u = 0; u < vertex_count_; ++u) {
      auto u_pid = GetVertexToPartitionId(u);
      for (uintE j = row_ptrs_[u]; j < row_ptrs_[u + 1]; ++j) {
        auto v = cols_[j];
        auto v_pid = GetVertexToPartitionId(v);
        if (u_pid != v_pid) {
          inter_cols_[offset[u]++] = v;
        }
      }
    }
    inter_edges_count_ = inter_sum;

    delete[] offset;
    offset = NULL;
  }
};

#endif
