#ifndef __CPU_PATTERN_MATCH_H__
#define __CPU_PATTERN_MATCH_H__

#include <vector>

class CPUPatternMatch {
 public:
  CPUPatternMatch(size_t thread_num) : thread_num_(thread_num) {
#if defined(PROFILE)
    thread_time_.resize(thread_num_, 0);
#endif
  }
  virtual ~CPUPatternMatch() {}

  virtual void Execute() = 0;

  long long GetTotalMatchCount() const { return total_match_count_; }
  void SetTotalMatchCount(long long c) { total_match_count_ = c; }

 protected:
  size_t thread_num_;
  long long total_match_count_;
#if defined(PROFILE)
  std::vector<long long> thread_time_;
#endif
};

#endif
