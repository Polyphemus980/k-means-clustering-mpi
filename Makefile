CC=nvcc
# this g++-9 and -arch=sm_35 is only for it to work on university machine (GPUNODE2)
# the last flag disables warning error (it seems like sm_35 is deprecated)
C_FLAGS_NODE2=-ccbin /usr/bin/g++-9 -arch=sm_35 -Wno-deprecated-gpu-targets
C_FLAGS_NODE3=--std=c++20
C_FLAGS_NODE3_OPTIMIZED=--std=c++20 -O3
SRC=src/main.cu src/utils.cu src/file_io.cu
TARGET=KMeans
TARGET_NODE2=${TARGET}-NODE2
TARGET_NODE3=${TARGET}-NODE3
TARGET_NODE3_OPT=${TARGET_NODE3}_OPT

node2: ${TARGET_NODE2}
node3: ${TARGET_NODE3}
node3_opt: ${TARGET_NODE3_OPT}

${TARGET_NODE2}:
	${CC} ${C_FLAGS_NODE2} -o ${TARGET} ${SRC}

${TARGET_NODE3}:
	${CC} ${C_FLAGS_NODE3} -o ${TARGET} ${SRC}

${TARGET_NODE3_OPT}:
	${CC} ${C_FLAGS_NODE3_OPTIMIZED} -o ${TARGET} ${SRC}

clean:
	rm -f ${TARGET}

.PHONY: clean
