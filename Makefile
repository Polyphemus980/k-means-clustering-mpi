MAKEFLAGS += --jobs=$(shell nproc)
CC := nvcc

# Compiler flags for different nodes
C_FLAGS_NODE2 := -ccbin /usr/bin/g++-9 -arch=sm_35 -Wno-deprecated-gpu-targets --extended-lambda
C_FLAGS_NODE3 := --std=c++20 --extended-lambda
C_FLAGS_NODE3_OPTIMIZED := --std=c++20 -O3 --extended-lambda

# Directory structure
SRC_DIR := src
OBJ_DIR := obj
TARGET := KMeans

# Find all source files
SRCS := $(wildcard $(SRC_DIR)/*.cu)
# Generate corresponding object file names
OBJS := $(SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Create object directory
$(OBJ_DIR):
	mkdir -p $@

# Compile each source file into an object file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link all object files into final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Phony targets
.PHONY: clean node2 node3 node3_opt

# Different build configurations
node2: CFLAGS = $(C_FLAGS_NODE2)
node2: $(TARGET)

node3: CFLAGS = $(C_FLAGS_NODE3)
node3: $(TARGET)

node3_opt: CFLAGS = $(C_FLAGS_NODE3_OPTIMIZED)
node3_opt: $(TARGET)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)