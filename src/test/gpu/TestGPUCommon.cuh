#include <gtest/gtest.h>
#include <vector>
#include "CPUGraph.h"
#include "DeviceArray.cuh"
#include "GPUUtil.cuh"
#include "Meta.h"

// the utilities for testing framework
//
template <typename DataType>
DeviceArray<DataType> *CreateDeviceArray(DataType *h_data, size_t size,
                                         CudaContext *context) {
  DeviceArray<DataType> *ret = new DeviceArray<DataType>(size, context);
  CUDA_ERROR(cudaMemcpy(ret->GetArray(), h_data, sizeof(DataType) * size,
                        cudaMemcpyHostToDevice));
  return ret;
}

template <typename DataType>
void Verify(DeviceArray<DataType> *actual, DataType *expect, size_t size) {
  if (size == 0) return;
  DataType *h_actual = new DataType[size];
  CUDA_ERROR(cudaMemcpy(h_actual, actual->GetArray(), sizeof(DataType) * size,
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < size; ++i) {
    ASSERT_EQ(h_actual[i], expect[i]);
  }
  delete[] h_actual;
  h_actual = NULL;
}

static TrackPartitionedGraph *GenerateNClique(const int N) {
  std::vector<std::vector<uintV> > graph_data;
  std::vector<uintP> partition_ids;
  const size_t vertex_count = N;
  graph_data.resize(vertex_count);
  for (uintV u = 0; u < vertex_count; ++u) {
    for (uintV v = 0; v < vertex_count; ++v) {
      if (v != u) graph_data[u].push_back(v);
    }
  }
  for (uintV u = 0; u < vertex_count; ++u) {
    partition_ids.push_back(0);
  }

  TrackPartitionedGraph *cpu_graph =
      new TrackPartitionedGraph(graph_data, partition_ids, 1);
  return cpu_graph;
}
static TrackPartitionedGraph *Generate6Clique() { return GenerateNClique(6); }
