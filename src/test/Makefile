GTEST_DIR = /home/wentian/programs/googletest/googletest
CC = g++
PROJ_DIR = ..
INCLUDE = $(shell find ${PROJ_DIR} -type d | sed s/^/-I/)
LIBS = $(GTEST_DIR)/libgtest.a -lpthread
OPEN_FLAGS=-DOPENMP -fopenmp -lgomp
FLAGS += -O3 -std=c++11 $(OPEN_FLAGS) -DBUILTIN_LOCK -DVALIDATE -DPROFILE

all: test_graph test_pattern_match_cpu test_query test_utils
build: test_graph test_pattern_match_cpu test_query test_utils

test_graph: TestGraph.cpp
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

test_pattern_match_cpu: TestPatternMatchCPU.cpp
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

test_query: TestQuery.cpp
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

test_utils: TestUtils.cpp
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf test_graph test_pattern_match_cpu test_query test_utils *.o
