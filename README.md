# K-Means Clustering using Cuda

## Loading data

Data is loaded from `txt` file with following structure:

```
N K
x1_{0} x2_{0} ... x{DIM}_{0}
.       .      .        .
.       .      .        .
.       .      .        .
x1_{N} x2_{N} ... x{DIM}_{N}
```

where:

- `N` - number of points
- `K` - number of clusters
- each line contains coordinates of i-th point separated by space

Additionally:

- `DIM` - dimension of the space (number of each point's coordinates) - template parameter

## Data layout

Currently idea is to store all points values in plain `float array`.
In 1-dim space it's trivial - this array have length equal to number of points and `array[i]` will be coordinate of i-th point.
However, in more dimensional space we will do something different - we will store each dimension in part of an array, e.g. in 3-dim space we will do:

```
[x1, x2, x3, ..., x_n, y1, y2, y3, ..., y_n, z1, z2, z3, ..., z_n]
```

where `x_i` is `x` of i-th point.

## Algorithm

Pseudo code can be found at [http://www.eecs.northwestern.edu/~wkliao/Kmeans/index.html](http://www.eecs.northwestern.edu/~wkliao/Kmeans/index.html).

Main part of the algorithm can be split into two parts:

### Find new centroid for each point

In this part we create thread for each point and calculate new centroid for each one in parallel. In each thread, we iterate over all centroids and find the one that is closest to given point (using euclidean distance).

### Find new centroids

After calculating centroid for each point we want to find new centroids. For this task we have two different methods:

#### First method

Firstly for each block in shared memory we create two arrays - one of floats (it will contain coordinats of centroids) and one of size_t (it will contain count of all the points that are assigned to given centroid) with length equal to number of centroids. Each element of the array starts as 0. We also create same two arrays in normal memory (cudaMalloc).
For each point we create thread and add points coordinates to a accumulator stored in shared memory, and we increase counter for given centroid.
When we finish given block, we take value for each centroid calculated in shared memory and add it to the main output array in normal memory. At the end for every centroid we divide accumulated coordinates by count of elements assigned to it, getting new centroids.

#### Second method

In second method we will use Thrust API. Firsly, we will use `thrust::sort_by_key` to group points with same membership to be next to each other. Next, we will use `thrust::reduce_by_key` to calculate mean for each cluster and this way get new centroids.
However, it's not so trivial due to the data layout. We will have to handle each dimension separately, meanining we will use `thrust::sort_by_key` and `thrust::reduce_by_key` separately for first dimension, second dimension, etc.

The main part will be run in loop until threshold condition is met. Each time new kernel will be launched, as we need block synchronization between loop steps.
