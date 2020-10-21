#ifndef __GPU_EXECUTOR_DEVICE_MANAGER_CUH__
#define __GPU_EXECUTOR_DEVICE_MANAGER_CUH__

#include <cassert>
#include <vector>

#include "Meta.h"
#include "Plan.h"
#include "SpinLock.h"

class DeviceManager {
 public:
  DeviceManager(size_t dev_num) {
    dev_num_ = dev_num;
    lock_.Init();
    available_.resize(dev_num_, true);
    occupy_thread_ids_.resize(dev_num_);
  }
  bool AcquireDevice(size_t thread_id, size_t& dev_id) {
    bool ret = false;
    lock_.Lock();
    for (size_t i = 0; i < dev_num_; ++i) {
      if (available_[i]) {
        available_[i] = false;
        occupy_thread_ids_[i] = thread_id;
        dev_id = i;
        ret = true;
        break;
      }
    }
    lock_.Unlock();
    return ret;
  }
  void ReleaseDevice(size_t thread_id, size_t dev_id) {
    lock_.Lock();
    assert(available_[dev_id] == false);
    assert(occupy_thread_ids_[dev_id] == thread_id);
    available_[dev_id] = true;
    occupy_thread_ids_[dev_id] = dev_num_;
    lock_.Unlock();
  }
  size_t GetDeviceOccupyingThreadId(size_t dev_id) {
    lock_.Lock();
    size_t ret = occupy_thread_ids_[dev_id];
    lock_.Unlock();
    return ret;
  }
  bool GetAvailable(size_t dev_id) {
    lock_.Lock();
    bool ret = available_[dev_id];
    lock_.Unlock();
    return ret;
  }
  size_t GetDeviceNum() const { return dev_num_; }

 protected:
  size_t dev_num_;
  SpinLock lock_;
  std::vector<bool> available_;
  std::vector<size_t> occupy_thread_ids_;
};

#endif