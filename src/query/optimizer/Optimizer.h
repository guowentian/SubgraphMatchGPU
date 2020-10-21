#ifndef __QUERY_OPTIMIZER_H__
#define __QUERY_OPTIMIZER_H__

#include "QueryCommon.h"

class Optimizer {
 public:
  Optimizer(QueryType query_type, size_t vertex_count, AllConnType &con,
            AllCondType &order)
      : query_type_(query_type),
        vertex_count_(vertex_count),
        con_(con),
        order_(order) {}

  // intra-partition: one search sequence
  virtual void GenerateOrder() {}

  // inter-partition: multiple search sequences
  virtual void GenerateInterPartitionOrder() {}

 protected:
  QueryType query_type_;
  size_t vertex_count_;
  AllConnType con_;
  AllCondType order_;
};

#endif
