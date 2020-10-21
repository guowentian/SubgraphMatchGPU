#ifndef __QUERY_OPTIMIZER_LAZY_TRAVERSAL_PLAN_HARD_CODE_H__
#define __QUERY_OPTIMIZER_LAZY_TRAVERSAL_PLAN_HARD_CODE_H__

#include "Plan.h"

class LazyTraversalPlanHardcode {
 public:
  static void GenerateSearchSequence(SearchSequence& seq, const Query* query) {
    size_t vertex_count = query->GetVertexCount();
    switch (query->GetQueryType()) {
      case Q0:
        Q0LazyPlan(seq, vertex_count);
        break;
      case Q1:
        Q1LazyPlan(seq, vertex_count);
        break;
      case Q2:
        Q2LazyPlan(seq, vertex_count);
        break;
      case Q3:
        Q3LazyPlan(seq, vertex_count);
        break;
      case Q4:
        Q4LazyPlan(seq, vertex_count);
        break;
      case Q5:
        Q5LazyPlan(seq, vertex_count);
        break;
      case Q6:
        Q6LazyPlan(seq, vertex_count);
        break;
      case Q7:
        Q7LazyPlan(seq, vertex_count);
        break;
      case Q8:
        Q8LazyPlan(seq, vertex_count);
        break;
      case Q9:
        Q9LazyPlan(seq, vertex_count);
        break;
      case Q10:
        Q10LazyPlan(seq, vertex_count);
        break;
      case Q11:
        Q11LazyPlan(seq, vertex_count);
        break;
      case Q12:
        Q12LazyPlan(seq, vertex_count);
        break;
      case Q13:
        Q13LazyPlan(seq, vertex_count);
        break;
      default:
        assert(false);
        break;
    }
  }

  static void Q0LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 2};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q1LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 2, 3};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q2LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 2, 1, 3};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q3LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 2, 3};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q4LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {1, 2, 3, 0, 4};
    seq.assign(ord, ord + vertex_count);
  }

  static void Q5LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 2, 4, 1, 3, 5};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q6LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 3, 2, 4};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q7LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 2, 3, 4};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q8LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 2, 3, 4};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q9LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 2, 3, 4};
    seq.assign(ord, ord + vertex_count);
  }

  static void Q10LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 3, 2, 4};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q11LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 2, 3, 4, 5};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q12LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 3, 2, 4};
    seq.assign(ord, ord + vertex_count);
  }
  static void Q13LazyPlan(SearchSequence& seq, size_t vertex_count) {
    const uintV ord[] = {0, 1, 2, 3, 4, 5};
    seq.assign(ord, ord + vertex_count);
  }
};

#endif
