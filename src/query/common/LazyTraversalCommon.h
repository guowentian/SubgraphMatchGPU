#ifndef __QUERY_COMMON_LAZY_TRAVERSAL_COMMON_H__
#define __QUERY_COMMON_LAZY_TRAVERSAL_COMMON_H__

#include "Meta.h"
#include "QueryCommon.h"

#include <iostream>
#include <string>
#include <vector>

enum LazyTraversalOperation {
  // compute the candidate set for a pattern vertex,
  // given the current partial instances
  COMPUTE,
  // execute COMPUTE step and do not materialize the candidate set,
  // but count the number for each path
  // only used in special treatment for fast counting, e.g., Q6
  COMPUTE_PATH_COUNT,

  // expand the candidate set and materialize the partial instances
  MATERIALIZE,
  // given the current (new) partial instances,
  // compute the candidate set again by enforcing new ordering constraints
  FILTER_COMPUTE,

  // count the number (under the compressed form)
  COUNT,
  // execute COMPUTE step and do not materialize the candidate set ,
  // but count the total number
  COMPUTE_COUNT,

};

// Since existing methods use hardcoded implementation or compressed output
// technique to count the instances in a fast way,
// to compare with them, we provide different variants of implementations
// to support these techniques.
enum LazyTraversalCompressLevel {
  // The most direct and intuitive method to count instances
  // without any specific optimization.
  // Each instance is generated and then counted.
  COMPRESS_LEVEL_MATERIALIZE,
  // A general optimization technique for counting that can apply to any
  // pattern graph.
  // The fast counting is achieved by avoiding unnecessary materialization.
  // For 1 unmaterialized vertex, directly count the size of candidate set
  // for each partial instance;
  // for 2 unmaterialized vertices, u1 and u2,
  // iterate each element in candidate_set(u_1), count the number of elements
  // in candidate_set(u2) can form the instances.
  COMPRESS_LEVEL_NON_MATERIALIZE,
  // A further optimization over COMPRESS_LEVEL_NON_MATERIALIZE that can also
  // apply to any pattern graph.
  // In this variant, the execute sequences have COMPUTE_COUNT and
  // COMPUTE_PATH_COUNT,
  // while COMPRESS_LEVEL_NON_MATERIALIZE does not have.
  // COMPUTE_COUNT and COMPUTE_PATH_COUNT can calculate the size of candidate
  // set
  // without materializing each element in the candidate set.
  // They are needed because in some cases, we don't need to materialize the
  // whole
  // candidate set in order for counting.
  COMPRESS_LEVEL_NON_MATERIALIZE_OPT,
  // The fastest and ad-hoc method for counting, which cannot apply to general
  // pattern graph.
  // For some pattern graphs, we can hardcode a specific way to quickly count
  // the
  // instances, given the candidate sets of some pattern vertices.
  // For other patterns that we cannot hardcode, we simply fall back to
  // OPT_LEVEL_EXT_NO_MATERIALIZE.
  // This hardcode technique to count the instances over the compressed form for
  // specialized patterns is used by many existing methods, but this is not
  // scalable
  // nor general.
  COMPRESS_LEVEL_SPECIAL
};

typedef std::pair<LazyTraversalOperation, uintV> LazyTraversalEntry;

static std::string GetLazyTraversalOperationString(LazyTraversalOperation op) {
  if (op == COMPUTE) {
    return "COMPUTE";
  } else if (op == MATERIALIZE) {
    return "MATERIALIZE";
  } else if (op == FILTER_COMPUTE) {
    return "FILTER_COMPUTE";
  } else if (op == COMPUTE_COUNT) {
    return "COMPUTE_COUNT";
  } else if (op == COMPUTE_PATH_COUNT) {
    return "COMPUTE_PATH_COUNT";
  } else if (op == COUNT) {
    return "COUNT";
  } else {
    return "";
  }
}

static void PrintLazyTraversalPlanData(
    const std::vector<LazyTraversalEntry>& exec_seq_,
    const MultiVTGroup& materialized_vertices_,
    const MultiVTGroup& computed_unmaterialized_vertices_) {
  for (auto p : exec_seq_) {
    std::cout << "(" << GetLazyTraversalOperationString(p.first);
    std::cout << "," << p.second << "),";
  }
  std::cout << std::endl;

  for (size_t i = 0; i < exec_seq_.size(); ++i) {
    std::cout << "materialized vertices:(";
    bool first = true;
    for (auto u : materialized_vertices_[i]) {
      if (first) {
        first = false;
      } else {
        std::cout << ",";
      }
      std::cout << u;
    }
    std::cout << ")" << std::endl;

    std::cout << "computed_unmaterialized_vertices:(";
    first = true;
    for (auto u : computed_unmaterialized_vertices_[i]) {
      if (first) {
        first = false;
      } else {
        std::cout << ",";
      }
      std::cout << u;
    }
    std::cout << ")" << std::endl;
  }
}

#endif
