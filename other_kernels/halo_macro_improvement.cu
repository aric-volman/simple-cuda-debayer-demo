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

__device__ __forceinline__ unsigned char load_pixel_halo(
    const uchar4* __restrict__ in4,
    int gx, int gy,
    int width,
    int height)
{
    // Clamp coordinates (handles halo safely)
    gx = max(0, min(gx, width  - 1));
    gy = max(0, min(gy, height - 1));

    // Align x to 4-byte boundary
    int gx4 = gx & ~3;

    // Ensure vector load stays in row bounds
    gx4 = min(gx4, width - 4);

    // Compute linear index (in pixels)
    int idx = gy * width + gx4;

    // Load 4 pixels at once
    uchar4 v = in4[idx >> 2];

    // Extract the correct byte
    return ((unsigned char*)&v)[gx - gx4];
}

__global__ void bayer_to_rgb(
    const unsigned char *__restrict__ in,
    unsigned char *__restrict__ out,
    int width,
    int height)
{
    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);

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

    // Clamp to multiple of 4
    int gx4 = (gx / 4) * 4;

    // Handle boundary case
    gx4 = min(gx4, width - 4);

    // Load a four pixel vector with index divided by 4 (>> 2)
    // Index aligned with multiple of 4
    uchar4 v = in4[(gy * width + gx4) >> 2];
    
    // In this thread retrieve pixel
    unsigned char pixel = ((unsigned char*)&v)[gx - gx4];
    
    // Load pixel
    tile[sy][sx] = pixel;

    // Load halo (edges)
    if (threadIdx.x == 0)
        tile[sy][0] = load_pixel_halo(in4, x - 1, y, width, height);

    if (threadIdx.x == BLOCK_X - 1)
        tile[sy][BLOCK_X + 1] = load_pixel_halo(in4, x + 1, y, width, height);

    if (threadIdx.y == 0)
        tile[0][sx] = load_pixel_halo(in4, x, y - 1, width, height);

    if (threadIdx.y == BLOCK_Y - 1)
        tile[BLOCK_Y + 1][sx] = load_pixel_halo(in4, x, y + 1, width, height);
        
    // Ensure that all threads are synced up after data transfer + halo creation
    __syncthreads();

    // ----------------------------
    // Bayer pattern section (RGGB pattern
    // ----------------------------

    // Keep in mind the coordinates for x,y are not 1-indexed
    // Coordinates for sx and sy are 1-indexed

    int x_odd = x & 1;

    int y_odd = y & 1;
    int y_even = y & 0;

    int is_red  = (!x_odd) & (!y_odd); // 1 or 0
    int is_blue = (x_odd) & (y_odd); // 1 or 0
    int is_green = 1 - is_red - is_blue; // Can be exclusively red or blue, so returns 0 if either red or blue

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

    unsigned int R_red = center;
    unsigned int G_red = avgFour(up, dn, lf, rt);
    unsigned int B_red = avgFour(top_lf, top_rt, down_lf, down_rt);
    
    unsigned int B_blue = center;
    unsigned int G_blue = avgFour(up, dn, lf, rt);
    unsigned int R_blue = avgFour(top_lf, top_rt, down_lf, down_rt);

    unsigned int G_green = center;

    // Different pattern on different y level
    // If even then red is left/right and blue is the opposite
    // If odd then red is up/down and blue is the opposite

    unsigned int R_green_even = avgTwo(lf, rt);
    unsigned int B_green_even = avgTwo(up, dn);

    unsigned int R_green_odd = avgTwo(up, dn);
    unsigned int B_green_odd = avgTwo(lf, rt);

    // Branchless logic for red and blue if green pixel
    unsigned int R_green = R_green_even * y_even + R_green_odd * y_odd;
    unsigned int B_green = B_green_even * y_even + B_green_odd * y_odd;

    R = R_red * is_red + R_blue * is_blue + R_green * is_green;
    G = G_red * is_red + G_blue * is_blue + G_green * is_green;
    B = B_red * is_red + B_blue * is_blue + B_green * is_green;
        

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
