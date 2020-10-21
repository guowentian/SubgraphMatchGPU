#ifndef __CPU_INTERSECTION_H__
#define __CPU_INTERSECTION_H__

#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <vector>
#include "Meta.h"

enum CPUIntersectMethod { HOME_MADE, STANDARD_LIB };

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
static size_t SortedIntersection(InputIterator1 it1_start,
                                 InputIterator1 it1_end,
                                 InputIterator2 it2_start,
                                 InputIterator2 it2_end,
                                 OutputIterator out_it) {
  size_t output_size = 0;
  while (it1_start != it1_end && it2_start != it2_end) {
    if (*it1_start < *it2_start) {
      ++it1_start;
    } else if (*it1_start > *it2_start) {
      ++it2_start;
    } else {
      *out_it = *it1_start;
      ++it1_start;
      ++it2_start;
      ++out_it;
      ++output_size;
    }
  }
  return output_size;
}
template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
static size_t StdSortedIntersection(InputIterator1 it1_start,
                                    InputIterator1 it1_end,
                                    InputIterator2 it2_start,
                                    InputIterator2 it2_end,
                                    OutputIterator out_it) {
  size_t output_size =
      std::set_intersection(it1_start, it1_end, it2_start, it2_end, out_it) -
      out_it;
  return output_size;
}
template <typename InputIterator1, typename T, typename OutputIterator>
static size_t HashIntersection(InputIterator1 it1_start, InputIterator1 it1_end,
                               std::unordered_set<T> *adj_set,
                               OutputIterator out_it) {
  OutputIterator orig_out_it = out_it;
  while (it1_start != it1_end) {
    if (adj_set->find(*it1_start) != adj_set->end()) {
      *out_it = *it1_start;
      ++out_it;
    }
    ++it1_start;
  }
  return out_it - orig_out_it;
}

// ========= API ===========
// m-way intersect in batch
template <CPUIntersectMethod method>
static void MWayIntersect(uintV *path, uintE *row_ptrs, uintV *cols,
                          std::vector<uintV> &intersect_levels,
                          std::vector<uintV> &result) {
  std::vector<uintV> res[2];
  for (size_t j = 0; j < intersect_levels.size(); ++j) {
    auto p2 = intersect_levels[j];
    auto first = path[p2];
    auto first_begin = &cols[row_ptrs[first]];
    auto first_end = &cols[row_ptrs[first + 1]];

    if (j == 0) {
      res[j % 2].assign(first_begin, first_end);
    } else {
      size_t max_size =
          std::min((size_t)(first_end - first_begin), res[(j + 1) % 2].size());
      res[j % 2].resize(max_size);
      size_t res_size = 0;
      if (method == HOME_MADE) {
        res_size =
            SortedIntersection(first_begin, first_end, res[(j + 1) % 2].begin(),
                               res[(j + 1) % 2].end(), res[j % 2].begin());
      } else if (method == STANDARD_LIB) {
        res_size = StdSortedIntersection(
            first_begin, first_end, res[(j + 1) % 2].begin(),
            res[(j + 1) % 2].end(), res[j % 2].begin());
      } else {
        assert(false);
      }
      assert(res_size <= max_size);
      res[j % 2].resize(res_size);
    }
  }
  size_t res_idx = (intersect_levels.size() + 1) % 2;
  result.swap(res[res_idx]);
}
// Note that intersect_arrays and intersect_arrays_size can be changed
template <CPUIntersectMethod method>
static void MWayIntersect(uintV **intersect_arrays,
                          size_t *intersect_arrays_size, size_t intersect_num,
                          uintV *intersect_result,
                          size_t &intersect_result_size) {
  if (intersect_num == 1) {
    for (size_t i = 0; i < intersect_arrays_size[0]; ++i) {
      intersect_result[i] = intersect_arrays[0][i];
    }
    intersect_result_size = intersect_arrays_size[0];
  } else {
    // reorder
    for (size_t i = 0; i < intersect_num; ++i) {
      for (size_t j = i + 1; j < intersect_num; ++j) {
        if (intersect_arrays_size[i] > intersect_arrays_size[j]) {
          std::swap(intersect_arrays_size[i], intersect_arrays_size[j]);
          std::swap(intersect_arrays[i], intersect_arrays[j]);
        }
      }
    }
    std::vector<uintV> res[2];
    res[0].assign(intersect_arrays[0],
                  intersect_arrays[0] + intersect_arrays_size[0]);
    for (size_t i = 1; i < intersect_num; ++i) {
      size_t max_size =
          std::min(res[(i + 1) % 2].size(), intersect_arrays_size[i]);
      res[i % 2].resize(max_size);
      size_t res_size = 0;
      if (i + 1 == intersect_num) {
        if (method == HOME_MADE) {
          res_size = SortedIntersection(
              res[(i + 1) % 2].begin(), res[(i + 1) % 2].end(),
              intersect_arrays[i],
              intersect_arrays[i] + intersect_arrays_size[i], intersect_result);
        } else {
          res_size = StdSortedIntersection<>(
              res[(i + 1) % 2].begin(), res[(i + 1) % 2].end(),
              intersect_arrays[i],
              intersect_arrays[i] + intersect_arrays_size[i], intersect_result);
        }
        intersect_result_size = res_size;
      } else {
        if (method == HOME_MADE) {
          res_size =
              SortedIntersection(res[(i + 1) % 2].begin(),
                                 res[(i + 1) % 2].end(), intersect_arrays[i],
                                 intersect_arrays[i] + intersect_arrays_size[i],
                                 res[i % 2].begin());
        } else {
          res_size = StdSortedIntersection<>(
              res[(i + 1) % 2].begin(), res[(i + 1) % 2].end(),
              intersect_arrays[i],
              intersect_arrays[i] + intersect_arrays_size[i],
              res[i % 2].begin());
        }
        res[i % 2].resize(res_size);
      }
    }
  }
}

// called before mway intersect in pipeline
static void PrepareMWayIntersect(uintV *path, uintE *row_ptrs, uintV *cols,
                                 std::vector<uintV> &intersect_indices,
                                 std::vector<uintV *> &intersect_edges_begins,
                                 std::vector<uintV *> &intersect_edges_ends,
                                 uintV &pivot_pos) {
  assert(intersect_indices.size() > 0);
  intersect_edges_begins.resize(intersect_indices.size());
  intersect_edges_ends.resize(intersect_indices.size());
  for (size_t j = 0; j < intersect_indices.size(); ++j) {
    auto plevel_index = intersect_indices[j];
    intersect_edges_begins[j] = cols + row_ptrs[path[plevel_index]];
    intersect_edges_ends[j] = cols + row_ptrs[path[plevel_index] + 1];
  }
  pivot_pos = 0;
  size_t min_ngr_num = intersect_edges_ends[0] - intersect_edges_begins[0];
  for (size_t j = 1; j < intersect_indices.size(); ++j) {
    size_t len = intersect_edges_ends[j] - intersect_edges_begins[j];
    if (min_ngr_num > len) {
      min_ngr_num = len;
      pivot_pos = j;
    }
  }
}

// ensure the partial instance cross partitions
static void PrepareMWayIntersectInterPartition(
    uintV *path, uintE *row_ptrs, uintV *cols, uintE *inter_row_ptrs,
    uintV *inter_cols, uintV curr_index, std::vector<uintV> &intersect_indices,
    std::vector<uintV *> &intersect_edges_begins,
    std::vector<uintV *> &intersect_edges_ends, uintV &pivot_pos,
    uintV prime_edge_0 = 0, uintV prime_edge_1 = 1) {
  assert(intersect_indices.size() > 0);
  intersect_edges_begins.resize(intersect_indices.size());
  intersect_edges_ends.resize(intersect_indices.size());
  for (size_t j = 0; j < intersect_indices.size(); ++j) {
    auto plevel_index = intersect_indices[j];
    if ((curr_index == prime_edge_0 && plevel_index == prime_edge_1) ||
        (plevel_index == prime_edge_0 && curr_index == prime_edge_1)) {
      intersect_edges_begins[j] =
          inter_cols + inter_row_ptrs[path[plevel_index]];
      intersect_edges_ends[j] =
          inter_cols + inter_row_ptrs[path[plevel_index] + 1];
    } else {
      intersect_edges_begins[j] = cols + row_ptrs[path[plevel_index]];
      intersect_edges_ends[j] = cols + row_ptrs[path[plevel_index] + 1];
    }
  }
  pivot_pos = 0;
  size_t min_ngr_num = intersect_edges_ends[0] - intersect_edges_begins[0];
  for (size_t j = 1; j < intersect_indices.size(); ++j) {
    size_t len = intersect_edges_ends[j] - intersect_edges_begins[j];
    if (min_ngr_num > len) {
      min_ngr_num = len;
      pivot_pos = j;
    }
  }
}

// mway intersect in pipeline
static bool MWayIntersect(uintV *path, uintV candidate,
                          std::vector<uintV> &intersect_indices,
                          std::vector<uintV *> &intersect_edges_begins,
                          std::vector<uintV *> &intersect_edges_ends,
                          uintV pivot_pos) {
  for (size_t j = 0; j < intersect_indices.size(); ++j) {
    // skip j as candidate is enumerated from the level pivot_pos
    if (j == pivot_pos) continue;
    if (intersect_edges_begins[j] == intersect_edges_ends[j]) {
      return false;
    }
    auto match_ptr = std::lower_bound(intersect_edges_begins[j],
                                      intersect_edges_ends[j], candidate);
    // when we search the next time, can start from match_ptr
    intersect_edges_begins[j] = match_ptr;
    if (match_ptr == intersect_edges_ends[j] || (*match_ptr) != candidate) {
      return false;
    }
  }
  return true;
}

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename InputIndexIterator1,
          typename OutputIndexIterator>
static size_t SortedIntersection(InputIterator1 it1_start,
                                 InputIterator1 it1_end,
                                 InputIndexIterator1 indices_it1_start,
                                 InputIndexIterator1 indices_it1_end,
                                 InputIterator2 it2_start,
                                 InputIterator2 it2_end, OutputIterator out_it,
                                 OutputIndexIterator indices_out_it) {
  size_t output_size = 0;
  while (it1_start != it1_end && it2_start != it2_end) {
    if (*it1_start < *it2_start) {
      ++it1_start;
      ++indices_it1_start;
    } else if (*it1_start > *it2_start) {
      ++it2_start;
    } else {
      *out_it = *it1_start;
      *indices_out_it = *indices_it1_start;
      ++it1_start;
      ++indices_it1_start;
      ++it2_start;
      ++out_it;
      ++indices_out_it;
      ++output_size;
    }
  }
  return output_size;
}

////// hash set based set intersection
static void MWayIntersect(uintV *path, uintE *row_ptrs, uintV *cols,
                          std::unordered_set<uintV> *adj_sets,
                          std::vector<uintV> &intersect_levels,
                          std::vector<uintV> &result) {
  std::vector<uintV> res[2];
  for (size_t j = 0; j < intersect_levels.size(); ++j) {
    auto p2 = intersect_levels[j];
    auto v = path[p2];
    if (j == 0) {
      res[j % 2].assign(&cols[row_ptrs[v]], &cols[row_ptrs[v + 1]]);
    } else {
      res[j % 2].resize(row_ptrs[v + 1] - row_ptrs[v], res[(j + 1) % 2].size());
      size_t res_size =
          HashIntersection(res[(j + 1) % 2].begin(), res[(j + 1) % 2].end(),
                           &adj_sets[v], res[j % 2].begin());
      res[j % 2].resize(res_size);
    }
  }
  size_t res_idx = (intersect_levels.size() + 1) % 2;
  result.swap(res[res_idx]);
}

#endif
