NVCC = g++
PROJ_DIR = ..
INCLUDE = $(shell find ${PROJ_DIR} -type d | sed s/^/-I/)
LIBS = 
OPEN_FLAGS=-DOPENMP -fopenmp -lgomp
CFLAGS += -O3 -std=c++11 ${OPEN_FLAGS} -DBUILTIN_LOCK -DDEBUG
					#-DPROFILE 
					#-DREUSE_PROFILE
					#-DBATCH_PROFILE
					#-DDEBUG 
					#-DVALIDATE#-DRUNTIME_LIMIT

all: test_plan 
build: test_plan 

test_plan: TestPlan.cpp
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf test_plan *.o
