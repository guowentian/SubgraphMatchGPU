#ifndef __QUERY_COMMON_TYPES_H__
#define __QUERY_COMMON_TYPES_H__

#include "Meta.h"

#include <string>
#include <vector>

typedef std::vector<uintV> ConnType;
// connectivity among all vertices
typedef std::vector<std::vector<uintV>> AllConnType;
// condition between two vertex
typedef std::pair<CondOperator, uintV> CondType;
// conditions among all vertices
typedef std::vector<std::vector<CondType>> AllCondType;

// a sequence of vertex ids that determine the search order
typedef std::vector<uintV> SearchSequence;
// a group of dfs ids
typedef std::vector<uintV> DfsIdGroup;
// multiple groups of dfs ids
typedef std::vector<DfsIdGroup> MultiDfsIdGroup;
// Each level has multiple groups, and each group is a set of dfs ids
typedef std::vector<MultiDfsIdGroup> LevelMultiDfsIdGroup;

// a group of vertices
typedef std::vector<uintV> VTGroup;

typedef std::vector<VTGroup> MultiVTGroup;

typedef std::vector<MultiVTGroup> LevelMultiVTGroup;

static std::string GetString(CondOperator op) {
  std::string ret = "";
  switch (op) {
    case LESS_THAN:
      ret = "LESS_THAN";
      break;
    case LARGER_THAN:
      ret = "LAGER_THAN";
      break;
    case NON_EQUAL:
      ret = "NON_EQUAL";
      break;
    default:
      break;
  }
  return ret;
}

#endif
