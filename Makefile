CC=nvcc 
# this g++-9 and -arch=sm_35 is only for it to work on university machine
# the last flag disables warning error (it seems like sm_35 is deprecated)
C_FLAGS=-ccbin /usr/bin/g++-9 -arch=sm_35 -Wno-deprecated-gpu-targets
SRC=src/main.cu
TARGET=k_means_clustering

all: ${TARGET}

${TARGET}:
	${CC} ${C_FLAGS} -o ${TARGET} ${SRC}

clean:
	rm -f ${TARGET}

.PHONY: clean
