#include "kernel.cuh"
#include <cuda_runtime.h>

#define BLOCK_X 64
#define BLOCK_Y 16

// Clamp helper: clamps and ensures a non-negative value
// For example: if the edge of the halo will be negative then this helper
// ensures that the loaded value is at coordinate 0
__device__ __forceinline__ int clamp(int v, int hi)
{
    return max(0, min(v, hi));
}

// Helper to index from 1D in[] to 2D shared memory tile
__device__ __forceinline__ unsigned char index2D(
    const unsigned char *in, int x, int y, int width)
{
    return in[y * width + x];
}

// Helper to take average of four numbers
__device__ __forceinline__ unsigned int avgFour(int a, int b, int c, int d)
{
    return (a + b + c + d) >> 2;
}

// Helper to take average of two numbers
__device__ __forceinline__ unsigned int avgTwo(int a, int b)
{
    return (a + b) >> 1;
}

__global__ void bayer_to_rgb(
    const unsigned char *__restrict__ in,
    unsigned char *__restrict__ out,
    int width,
    int height)
{
    // Global coordinates
    int x = blockIdx.x * BLOCK_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    // Shared memory tile with halo
    // Note: divisible by 3 for 66 x 18 tiles
    __shared__ unsigned char tile[BLOCK_Y + 2][BLOCK_X + 2];

    // Local coords in shared memory (offset by 1 for halo)
    // For example 0,0 is represented as 1,1 in shared memory
    int sx = threadIdx.x + 1;
    int sy = threadIdx.y + 1;

    // Load pixel in the center of the flower pattern (3x3 debayer pattern)
    int gx = clamp(x, width - 1);
    int gy = clamp(y, height - 1);
    // This should coallesce better because it will coallesce in parallel
    // This is defined as one pixel per thread
    // Each thread loads this singular pixel
    tile[sy][sx] = in[gy * width + gx];

    // Load halo (edges)
    if (threadIdx.x == 0)
        tile[sy][0] = index2D(in, clamp(x - 1, width - 1), gy, width);

    if (threadIdx.x == BLOCK_X - 1)
        tile[sy][BLOCK_X + 1] = index2D(in, clamp(x + 1, width - 1), gy, width);

    if (threadIdx.y == 0)
        tile[0][sx] = index2D(in, gx, clamp(y - 1, height - 1), width);

    if (threadIdx.y == BLOCK_Y - 1)
        tile[BLOCK_Y + 1][sx] = index2D(in, gx, clamp(y + 1, height - 1), width);

    // Corners - Probably optional but logically correct
    if (threadIdx.x == 0 && threadIdx.y == 0)
        tile[0][0] = index2D(in, clamp(x - 1, width - 1), clamp(y - 1, height - 1), width);

    if (threadIdx.x == BLOCK_X - 1 && threadIdx.y == 0)
        tile[0][BLOCK_X + 1] = index2D(in, clamp(x + 1, width - 1), clamp(y - 1, height - 1), width);

    if (threadIdx.x == 0 && threadIdx.y == BLOCK_Y - 1)
        tile[BLOCK_Y + 1][0] = index2D(in, clamp(x - 1, width - 1), clamp(y + 1, height - 1), width);

    if (threadIdx.x == BLOCK_X - 1 && threadIdx.y == BLOCK_Y - 1)
        tile[BLOCK_Y + 1][BLOCK_X + 1] = index2D(in, clamp(x + 1, width - 1), clamp(y + 1, height - 1), width);

    // Ensure that all threads are synced up after data transfer + halo creation
    __syncthreads();

    // Sanity check at edges
    if (x >= width || y >= height)
        return;

    // ----------------------------
    // Bayer pattern section (RGGB pattern
    // ----------------------------

    // Keep in mind the coordinates for x,y are not 1-indexed
    // Coordinates for sx and sy are 1-indexed

    bool y_even = !(y & 1);
    bool x_even = !(x & 1);

    // Red pattern is even
    // First red is at (0,0) of the array which is even
    bool is_red  = y_even && x_even;
    // Blue pattern is odd
    // First blue is at (1,1) of the array which are both odd
    bool is_blue = !y_even && !x_even;

    // Green is implicitly handled

    // Use intrinsics which might be better
    // Specific to CUDA
    unsigned int R, G, B;

    unsigned char center = tile[sy][sx];
    unsigned char up = tile[sy - 1][sx];
    unsigned char dn = tile[sy + 1][sx];
    unsigned char lf = tile[sy][sx - 1];
    unsigned char rt = tile[sy][sx + 1];

    unsigned char top_lf = tile[sy - 1][sx - 1];
    unsigned char top_rt = tile[sy - 1][sx + 1];
    unsigned char down_lf = tile[sy + 1][sx - 1];
    unsigned char down_rt = tile[sy + 1][sx + 1];

    if (is_red)
    {
        R = center;
        G = avgFour(up, dn, lf, rt);
        B = avgFour(top_lf, top_rt, down_lf, down_rt);
    }
    else if (is_blue)
    {
        B = center;
        G = avgFour(up, dn, lf, rt);
        R = avgFour(top_lf, top_rt, down_lf, down_rt);
    }
    else
    { // Otherwise green
        G = center;
        // Different pattern on different y level
        // If even then red is left/right and blue is the opposite
        // If odd then red is up/down and blue is the opposite
        if (y_even)
        {
            R = avgTwo(lf, rt);
            B = avgTwo(up, dn);
        }
        else
        {
            R = avgTwo(up, dn);
            B = avgTwo(lf, rt);
        }
    }

    // Multiple of 3 - make sure to assign each pixel with enough space
    int out_idx = (y * width + x) * 3;
    out[out_idx + 0] = (unsigned char)B;
    out[out_idx + 1] = (unsigned char)G;
    out[out_idx + 2] = (unsigned char)R;
}



void debayer_cuda(
    unsigned char* d_bayer,
    unsigned char* d_rgb,
    int width,
    int height
)
{
    // This thread allocation works!
    dim3 threads(BLOCK_X, BLOCK_Y);

    dim3 blocks(
        (width + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y
    );

    // Assumes RGGB as default
   bayer_to_rgb<<<blocks, threads>>>(
    d_bayer,
    d_rgb,
    width,
    height
   );
}
