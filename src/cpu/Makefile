##BATCH_PROFILE: print info when generating new batch
##DEBUG: do some validation after searching the cached intermediate result for reusable bfs
##PROFILE: profile the execution time of each phase

NVCC = g++
PROJ_DIR = ..
INCLUDE = $(shell find ${PROJ_DIR} -type d | sed s/^/-I/)
LIBS = 
OPEN_FLAGS=-DOPENMP -fopenmp -lgomp
CFLAGS += -O3 -std=c++11 ${OPEN_FLAGS} -DBUILTIN_LOCK
					#-DPROFILE 
					#-DREUSE_PROFILE
					#-DBATCH_PROFILE
					#-DDEBUG 
					#-DVALIDATE#-DRUNTIME_LIMIT

all: patternmatchcpu 
build: patternmatchcpu 

patternmatchcpu: PatternMatchCPU.cpp
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf patternmatchcpu *.o
