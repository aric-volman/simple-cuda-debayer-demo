#include "kernel.cuh"
#include <cuda_runtime.h>

__global__ void bayer_to_rgb(
unsigned char* in,
unsigned char* out,
    int imgw,
    int imgh,
    int bpp,
    int2 r,
    int2 gr,
    int2 gb,
    int2 b)
{
    int x = 2 * ((blockDim.x * blockIdx.x) + threadIdx.x) + 1;
    int y = 2 * ((blockDim.y * blockIdx.y) + threadIdx.y) + 1;

    int elemCols = imgw * bpp;

    if ((x + 2) < imgw && (x - 1) >= 0 && (y + 2) < imgh && (y - 1) >= 0) {

        /* RED pixel */
        out[(y+r.y)*elemCols + (x+r.x)*bpp + 0] =
            in[(x+r.x) + (y+r.y)*imgw];

        out[(y+r.y)*elemCols + (x+r.x)*bpp + 1] =
            (in[(x+r.x-1)+(y+r.y)*imgw] +
             in[(x+r.x+1)+(y+r.y)*imgw] +
             in[(x+r.x)+(y+r.y-1)*imgw] +
             in[(x+r.x)+(y+r.y+1)*imgw]) / 4;

        out[(y+r.y)*elemCols + (x+r.x)*bpp + 2] =
            (in[(x+r.x-1)+(y+r.y-1)*imgw] +
             in[(x+r.x+1)+(y+r.y-1)*imgw] +
             in[(x+r.x-1)+(y+r.y+1)*imgw] +
             in[(x+r.x+1)+(y+r.y+1)*imgw]) / 4;

        /* GREEN on red row */
        out[(y+gr.y)*elemCols + (x+gr.x)*bpp + 0] =
            (in[(x+gr.x-1)+(y+gr.y)*imgw] +
             in[(x+gr.x+1)+(y+gr.y)*imgw]) / 2;

        out[(y+gr.y)*elemCols + (x+gr.x)*bpp + 1] =
            in[(x+gr.x)+(y+gr.y)*imgw];

        out[(y+gr.y)*elemCols + (x+gr.x)*bpp + 2] =
            (in[(x+gr.x)+(y+gr.y-1)*imgw] +
             in[(x+gr.x)+(y+gr.y+1)*imgw]) / 2;

        /* GREEN on blue row */
        out[(y+gb.y)*elemCols + (x+gb.x)*bpp + 0] =
            (in[(x+gb.x)+(y+gb.y-1)*imgw] +
             in[(x+gb.x)+(y+gb.y+1)*imgw]) / 2;

        out[(y+gb.y)*elemCols + (x+gb.x)*bpp + 1] =
            in[(x+gb.x)+(y+gb.y)*imgw];

        out[(y+gb.y)*elemCols + (x+gb.x)*bpp + 2] =
            (in[(x+gb.x-1)+(y+gb.y)*imgw] +
             in[(x+gb.x+1)+(y+gb.y)*imgw]) / 2;

        /* BLUE pixel */
        out[(y+b.y)*elemCols + (x+b.x)*bpp + 0] =
            (in[(x+b.x-1)+(y+b.y-1)*imgw] +
             in[(x+b.x+1)+(y+b.y-1)*imgw] +
             in[(x+b.x-1)+(y+b.y+1)*imgw] +
             in[(x+b.x+1)+(y+b.y+1)*imgw]) / 4;

        out[(y+b.y)*elemCols + (x+b.x)*bpp + 1] =
            (in[(x+b.x-1)+(y+b.y)*imgw] +
             in[(x+b.x+1)+(y+b.y)*imgw] +
             in[(x+b.x)+(y+b.y-1)*imgw] +
             in[(x+b.x)+(y+b.y+1)*imgw]) / 4;

        out[(y+b.y)*elemCols + (x+b.x)*bpp + 2] =
            in[(x+b.x)+(y+b.y)*imgw];
    }
}

void debayer_cuda(
    unsigned char* d_bayer,
    unsigned char* d_rgb,
    int width,
    int height
)
{
    // This thread allocation works!
    dim3 threads(32,32);
    dim3 blocks(width/2/32, height/2/32);
    
    // Assume RGGB as default - if not, can change if needed
    int2 r  = make_int2(0,0);
    int2 gr = make_int2(1,0);
    int2 gb = make_int2(0,1);
    int2 b  = make_int2(1,1);
    int bpp = 3; // RGB - three channels


   bayer_to_rgb<<<blocks, threads>>>(
    d_bayer,
    d_rgb,
    width,
    height,
    bpp,      // RGB
    r, gr, gb, b
   );
}
