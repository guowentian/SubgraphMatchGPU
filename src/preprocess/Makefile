GCC = g++
PROJ_DIR = ..
PROJ_INCLUDE = $(shell find ${PROJ_DIR} -type d | sed s/^/-I/)
INCLUDE = $(PROJ_INCLUDE) 
LIBS = -lm 
OPENMP_FLAGS = -DOPENMP -fopenmp -lgomp
CFLAGS += -std=c++11 -O3 -DBUILTIN_LOCK 

all: preprocess_graph motif_cut
build: preprocess_graph motif_cut

OBJ = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

%.o: %.cpp 
	$(GCC) $< $(CFLAGS) $(INCLUDE) -c -o $@
	
preprocess_graph: PreprocessGraph.cpp
	$(GCC) $(OPENMP_FLAGS) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

motif_cut: MotifCut.cpp
	$(GCC) $(OPENMP_FLAGS) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf preprocess_graph motif_cut *.o
