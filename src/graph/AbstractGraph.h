#ifndef __ABSTRACT_GRAPH_H__
#define __ABSTRACT_GRAPH_H__

#include "Meta.h"

class AbstractGraph {
 public:
  AbstractGraph(bool directed) : directed_(directed) {
    vertex_count_ = edge_count_ = 0;
  }
  ~AbstractGraph() {}

  bool GetDirected() const { return directed_; }
  size_t GetEdgeCount() const { return edge_count_; }
  size_t GetVertexCount() const { return vertex_count_; }
  void SetVertexCount(size_t vertex_count) { vertex_count_ = vertex_count; }
  void SetEdgeCount(size_t edge_count) { edge_count_ = edge_count; }

 protected:
  const bool directed_;
  size_t vertex_count_;
  size_t edge_count_;
};

#endif
