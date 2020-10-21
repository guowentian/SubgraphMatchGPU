#ifndef __TIME_MEASURER_H__
#define __TIME_MEASURER_H__

#include <stdlib.h>
#include <sys/time.h>

class TimeMeasurer {
 public:
  TimeMeasurer() {}
  ~TimeMeasurer() {}

  void StartTimer() { start_timestamp = wtime(); }
  void EndTimer() { end_timestamp = wtime(); }

  double GetElapsedMicroSeconds() const {
    return end_timestamp - start_timestamp;
  }

  double wtime() {
    double time[2];
    struct timeval time1;
    gettimeofday(&time1, NULL);

    time[0] = time1.tv_sec;
    time[1] = time1.tv_usec;

    return time[0] * (1.0e6) + time[1];
  }

 private:
  double start_timestamp;
  double end_timestamp;
};

#endif
