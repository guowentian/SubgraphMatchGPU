#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <unordered_set>

#include "CPUGraph.h"
#include "GraphIO.h"
#include "GraphPartition.h"
#include "PreprocessGraph.h"

TEST(GraphTest, CSRBinGraphTest) {
  std::string filename("../../data/com-dblp.ungraph.txt.bin");
  Graph* graph = new Graph(filename, false);
  auto vertex_count = graph->GetVertexCount();
  auto edge_count = graph->GetEdgeCount();
  auto row_ptrs = graph->GetRowPtrs();
  auto cols = graph->GetCols();
  ASSERT_EQ(317080, vertex_count);
  ASSERT_EQ(2099732, edge_count);
  delete graph;
  graph = NULL;
}

TEST(GraphTest, SNAPGraphTest) {
  std::string filename("../../data/com-dblp.ungraph.txt");
  Graph* graph = new Graph(filename, false);
  auto vertex_count = graph->GetVertexCount();
  auto edge_count = graph->GetEdgeCount();
  auto row_ptrs = graph->GetRowPtrs();
  auto cols = graph->GetCols();
  ASSERT_EQ(425957, vertex_count);
  ASSERT_EQ(2099732, edge_count);
  for (uintV u = 0; u < 100; ++u) {
    for (uintE j = row_ptrs[u] + 1; j < row_ptrs[u + 1]; ++j) {
      ASSERT_LE(cols[j - 1], cols[j]);
    }
  }
  delete graph;
  graph = NULL;
}

TEST(GraphTest, PreprocessGraphTest) {
  std::string filename("../../data/simple_graph1.txt");
  Graph* graph = new Graph(filename, false);
  PreprocessGraph* preprocess = new PreprocessGraph(graph);
  preprocess->Preprocess();
  delete preprocess;
  preprocess = NULL;
  auto vertex_count = graph->GetVertexCount();
  auto edge_count = graph->GetEdgeCount();
  auto row_ptrs = graph->GetRowPtrs();
  auto cols = graph->GetCols();

  ASSERT_EQ(5, vertex_count);
  ASSERT_EQ(12, edge_count);
  size_t eq_row_ptrs[] = {0, 3, 5, 8, 10, 12};
  uintV eq_cols[] = {1, 2, 3, 0, 2, 0, 1, 4, 0, 4, 2, 3};
  for (uintV u = 0; u <= vertex_count; ++u) {
    ASSERT_EQ(eq_row_ptrs[u], row_ptrs[u]);
  }
  for (size_t j = 0; j < edge_count; ++j) {
    ASSERT_EQ(eq_cols[j], cols[j]);
  }

  delete graph;
  graph = NULL;
}

TEST(GraphTest, CPUGraphTest) {
  std::string data_filename("../../data/simple_graph2.txt");
  std::string partition_filename("../../data/simple_graph2_partition.txt");
  size_t partition_num = 2;
  TrackPartitionedGraph* graph = new TrackPartitionedGraph(
      data_filename, false, partition_filename, partition_num);

  // test GraphPartition
  {
    // partition 0: (0,1)
    GraphPartition* partition = graph->GetPartition(0);
    ASSERT_EQ(2, partition->GetVertexCount());
    ASSERT_EQ(2, partition->GetEdgeCount());
    uintV eq_cols[] = {1, 0};
    size_t eq_row_ptrs[] = {0, 1, 2};
    for (uintV u = 0; u <= 2; ++u) {
      ASSERT_EQ(eq_row_ptrs[u], partition->GetRowPtrs()[u]);
    }
    for (size_t j = 0; j < 2; ++j) {
      ASSERT_EQ(eq_cols[j], partition->GetCols()[j]);
    }
  }
  {
    // partition 1: (2,4) (3,4)
    GraphPartition* partition = graph->GetPartition(1);
    ASSERT_EQ(3, partition->GetVertexCount());
    ASSERT_EQ(4, partition->GetEdgeCount());
    size_t eq_row_ptrs[] = {0, 1, 2, 4};
    uintV eq_cols[] = {2, 2, 0, 1};
    for (uintV u = 0; u <= 3; ++u) {
      ASSERT_EQ(eq_row_ptrs[u], partition->GetRowPtrs()[u]);
    }
    for (size_t j = 0; j < 4; ++j) {
      ASSERT_EQ(eq_cols[j], partition->GetCols()[j]);
    }
  }
  // test TrackPartitionedGraph
  auto inter_row_ptrs = graph->GetInterRowPtrs();
  auto inter_cols = graph->GetInterCols();
  size_t eq_inter_row_ptrs[] = {0, 2, 3, 5, 6, 6};
  uintV eq_inter_cols[] = {2, 3, 2, 0, 1, 0};
  ASSERT_EQ(5, graph->GetVertexCount());
  ASSERT_EQ(6, graph->GetInterPartitionEdgesCount());
  for (uintV u = 0; u <= 5; ++u) {
    ASSERT_EQ(eq_inter_row_ptrs[u], inter_row_ptrs[u]);
  }
  for (size_t j = 0; j < 6; ++j) {
    ASSERT_EQ(eq_inter_cols[j], inter_cols[j]);
  }

  delete graph;
  graph = NULL;
}

TEST(GraphTest, GraphFromVectorTest) {
  std::vector<std::vector<uintV>> graph_data;
  std::vector<uintP> partition_ids;
  const size_t vertex_count = 6;
  graph_data.resize(vertex_count);
  for (uintV u = 0; u < vertex_count; ++u) {
    for (uintV v = 0; v < vertex_count; ++v) {
      if (v != u) graph_data[u].push_back(v);
    }
  }
  for (uintV u = 0; u < vertex_count; ++u) {
    partition_ids.push_back(0);
  }

  TrackPartitionedGraph* cpu_graph =
      new TrackPartitionedGraph(graph_data, partition_ids, 1);

  {
    // verify
    auto row_ptrs = cpu_graph->GetRowPtrs();
    auto cols = cpu_graph->GetCols();
    auto partition_map = cpu_graph->GetVertexPartitionMap();
    size_t per_len = 5;
    size_t prefix = 0;
    for (size_t i = 0; i < 6; ++i) {
      ASSERT_EQ(row_ptrs[i], prefix);
      prefix += per_len;
    }

    for (uintV u = 0; u < 6; ++u) {
      size_t cur = 0;
      for (size_t off = 0; off < 5; ++off) {
        if (cur == u) ++cur;
        ASSERT_EQ(cols[row_ptrs[u] + off], cur);
        ++cur;
      }
    }
    ASSERT_EQ(row_ptrs[6], cpu_graph->GetEdgeCount());

    for (size_t i = 0; i < 6; ++i) {
      ASSERT_EQ(partition_map[i], 0);
    }
  }

  delete cpu_graph;
  cpu_graph = NULL;
}

template <typename IndexType, typename DataType>
static bool Isomorphic(size_t vertex_count, size_t edge_count,
                       std::vector<IndexType>& old_row_ptrs,
                       std::vector<DataType>& old_cols,
                       std::vector<IndexType>& new_row_ptrs,
                       std::vector<DataType>& new_cols, DataType* mapping,
                       bool* used, size_t level) {
  if (level == vertex_count) return true;
  for (DataType u = 0; u < vertex_count; ++u) {
    if (!used[u]) {
      mapping[level] = u;
      used[u] = true;

      if (old_row_ptrs[level + 1] - old_row_ptrs[level] ==
          new_row_ptrs[u + 1] - new_row_ptrs[u]) {
        // degree equal
        bool valid = true;
        for (IndexType old_j = old_row_ptrs[level];
             old_j < old_row_ptrs[level + 1]; ++old_j) {
          DataType old_v = old_cols[old_j];
          if (old_v < level) {
            // backward edge
            DataType new_v = mapping[old_v];

            // find whetehr new_v exists in the edge list
            bool exists = false;
            for (IndexType new_j = new_row_ptrs[u]; new_j < new_row_ptrs[u + 1];
                 ++new_j) {
              DataType v = new_cols[new_j];
              if (v == new_v) {
                exists = true;
              }
            }
            if (!exists) {
              valid = false;
              break;
            }
          }
        }
        if (valid) {
          if (Isomorphic(vertex_count, edge_count, old_row_ptrs, old_cols,
                         new_row_ptrs, new_cols, mapping, used, level + 1)) {
            return true;
          }
        }
      }

      used[u] = false;
    }
  }
  return false;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
