# K-Means Clustering using Cuda

## Running program

To run the program, first compile it using the provided `CMakeLists.txt` file.
Having binary `KMeans` just run: 
```shell
./KMeans data_format computation_method input_file output_file
```

where:
 - `data_format` is either `bin` or `txt` (look at `Loading data` section)
 - `computation_method` is one of `cpu`, `gpu1` and `gpu2`

## Loading data

There are two data formats

### Text file

Data is loaded from a `txt` file with following structure:

```
N DIM K
x1_{0} x2_{0} ... x{DIM}_{0}
.       .      .        .
.       .      .        .
.       .      .        .
x1_{N} x2_{N} ... x{DIM}_{N}
```

where:

- `N` - number of points
- `K` - number of clusters
- `DIM` - number of dimensions
- each line contains coordinates of i-th point separated by space

### Binary file

Data is loaded from a `bin` file with following structure:
`[3*int(4B)][N*DIM*float(4B)]`

where:

- first `12` bytes are `3` ints - `N`, `K` and `DIM`
- next `N * DIM * 4` bytes are coordinates of `N` points, each one with `DIM` coordinates, each coordinate being a float (4 bytes)

**Important Note (applies to both file formats)**
- `DIM` must always be in range [1, 20]
- `K` must always be in range [2, 20]


## Data layout

The current idea is to store all points' coordinates in a plain `float array`.
In 1-dimensional space, it's trivial — this array has a length equal to the number of points and `array[i]` will be coordinate of i-th point.
However, in more dimensional space we will do something different - we will store each dimension in part of an array, e.g. in 3-dim space we will do:

```
[x1, x2, x3, ..., x_n, y1, y2, y3, ..., y_n, z1, z2, z3, ..., z_n]
```

where `x_i` is `x` of i-th point.

## Algorithm

Pseudo code can be found at [http://www.eecs.northwestern.edu/~wkliao/Kmeans/index.html](http://www.eecs.northwestern.edu/~wkliao/Kmeans/index.html).

Main part of the algorithm can be split into two parts:

### Find new centroid for each point

In this part we create a thread for each point and calculate a new centroid for each one in parallel.. In each thread, we iterate over all centroids and find the one that is closest to given point (using euclidean distance). Centroids coordinates will be loaded into shared memory. 

### Find new centroids

After calculating centroid for each point we want to find new centroids. For this task we have two different methods:

#### First method

First, for each block in shared memory, we create two arrays — one for floats (to store the coordinates of centroids) and one of ints (it will contain count of all the points that are assigned to given centroid). Each element of the array starts as 0. We also create the same two arrays in global memory.
For each point we create thread and add points coordinates to a accumulator stored in shared memory, and we increase counter for given centroid.
When we finish given block, we take value for each centroid calculated in shared memory and add it to the main output array in global memory. At the end for every centroid we divide accumulated coordinates by count of elements assigned to it, getting new centroids.

#### Second method

In second method we will use Thrust API. First, we will use `thrust::sort_by_key` to group points with same membership to be next to each other. Next, we will use `thrust::reduce_by_key` to calculate mean for each cluster and this way get new centroids.
However, it's not so trivial due to the data layout. We will have to handle each dimension separately, meaning we will use `thrust::sort_by_key` and `thrust::reduce_by_key` separately for first dimension, second dimension, etc.

The main part will be run in loop until threshold condition is met. Each time new kernel will be launched, as we need block-level synchronization between loop steps.
