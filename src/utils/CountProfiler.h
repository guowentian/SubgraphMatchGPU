#ifndef __UTILS_COUNT_PROFILER_H__
#define __UTILS_COUNT_PROFILER_H__

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

class CountProfiler {
 public:
  typedef size_t CountType;

  CountProfiler(size_t thread_count) : thread_num_(thread_count) {
    phase_num_ = 0;
    accum_counts_.resize(thread_num_);
  }

  void Clear() {
    for (size_t i = 0; i < thread_num_; ++i) {
      for (size_t j = 0; j < phase_num_; ++j) {
        accum_counts_[i][j] = 0;
      }
    }
  }
  void AddPhase(const char *phase_name) {
    assert(!PhaseExists(phase_name));
    phase_names_.push_back(std::string(phase_name));
    phase_num_++;
    for (size_t i = 0; i < thread_num_; ++i) {
      accum_counts_[i].push_back(0);
    }
  }
  void AddCount(const char *phase_name, size_t thread_id, CountType c) {
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    accum_counts_[thread_id][pos] += c;
#endif
  }
  CountType GetCount(size_t thread_id, const char *phase_name) {
    CountType ret = 0;
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    ret = accum_counts_[thread_id][pos];
#endif
    return ret;
  }
  CountType GetCount(const char *phase_name) {
    CountType ret = 0;
#if defined(PROFILE)
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      ret += GetCount(thread_id, phase_name);
    }
#endif
    return ret;
  }

  void Report() {
#if defined(PROFILE)
    for (size_t i = 0; i < thread_num_; ++i) {
      Report(i);
    }
#endif
  }
  void Report(size_t thread_id) {
#if defined(PROFILE)
    if (phase_num_) {
      for (size_t j = 0; j < phase_num_; ++j) {
        if (j > 0) std::cout << ",";
        std::cout << phase_names_[j] << "=" << accum_counts_[thread_id][j];
      }
      std::cout << std::endl;
    }
#endif
  }
  void Report(const char *phase_name) {
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    std::cout << "phase_name=" << phase_name;
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      std::cout << ",(thread_id=" << thread_id << ","
                << accum_counts_[thread_id][pos] << "ms)";
    }
    std::cout << std::endl;
#endif
  }
  void ReportAgg(const char *phase_name) {
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    CountType total = this->GetCount(phase_name);
    std::cout << "phase_name=" << phase_name << ",count=" << total;
    std::cout << std::endl;
#endif
  }

  size_t GetThreadNum() const { return thread_num_; }
  size_t GetPhaseNum() const { return phase_num_; }

 private:
  size_t FindPhaseIndex(const char *phase_name) {
    std::string name(phase_name);
    size_t pos = phase_num_;
    for (size_t i = 0; i < phase_num_; ++i) {
      if (name == phase_names_[i]) {
        pos = i;
        break;
      }
    }
    assert(pos < phase_num_);
    return pos;
  }
  bool PhaseExists(const char *phase_name) {
    std::string name(phase_name);
    for (size_t i = 0; i < phase_names_.size(); ++i) {
      if (name == phase_names_[i]) {
        return true;
      }
    }
    return false;
  }

  size_t thread_num_;
  size_t phase_num_;
  std::vector<std::string> phase_names_;
  std::vector<std::vector<CountType> > accum_counts_;
};

#endif
