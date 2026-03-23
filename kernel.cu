#include "kernel.cuh"
#include <cuda_runtime.h>

__global__ void bayer_to_rgb(
    unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
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

        // Read bayer pixel at offset (dx, dy) from thread's base position (x, y)
        auto bayer = [&](int dx, int dy) -> int {
            return in[(x + dx) + (y + dy) * imgw];
        };

        // Interpolation helpers: horizontal, vertical, 4-cardinal, 4-diagonal
        auto h_avg = [&](int dx, int dy) -> unsigned char {
            return (bayer(dx - 1, dy) + bayer(dx + 1, dy)) / 2;
        };
        auto v_avg = [&](int dx, int dy) -> unsigned char {
            return (bayer(dx, dy - 1) + bayer(dx, dy + 1)) / 2;
        };
        auto cross_avg = [&](int dx, int dy) -> unsigned char {
            // Horizontal neighbors (same row, likely L1-cached) before vertical
            return (bayer(dx - 1, dy) + bayer(dx + 1, dy) +
                    bayer(dx, dy - 1) + bayer(dx, dy + 1)) / 4;
        };
        auto diag_avg = [&](int dx, int dy) -> unsigned char {
            // Top row before bottom row to stride through memory sequentially
            return (bayer(dx - 1, dy - 1) + bayer(dx + 1, dy - 1) +
                    bayer(dx - 1, dy + 1) + bayer(dx + 1, dy + 1)) / 4;
        };

        // === Read phase ===
        // Compute all 12 channel values before issuing any stores. Grouping
        // all global loads together lets the GPU's memory scheduler coalesce
        // and pipeline them across the warp without interleaving with stores.

        /* RED pixel: R known, G cross-interpolated, B diag-interpolated */
        unsigned char r_R = bayer(r.x, r.y);
        unsigned char r_G = cross_avg(r.x, r.y);
        unsigned char r_B = diag_avg(r.x, r.y);

        /* GREEN on red row: R horiz-interpolated, G known, B vert-interpolated */
        unsigned char gr_R = h_avg(gr.x, gr.y);
        unsigned char gr_G = bayer(gr.x, gr.y);
        unsigned char gr_B = v_avg(gr.x, gr.y);

        /* GREEN on blue row: R vert-interpolated, G known, B horiz-interpolated */
        unsigned char gb_R = v_avg(gb.x, gb.y);
        unsigned char gb_G = bayer(gb.x, gb.y);
        unsigned char gb_B = h_avg(gb.x, gb.y);

        /* BLUE pixel: R diag-interpolated, G cross-interpolated, B known */
        unsigned char b_R = diag_avg(b.x, b.y);
        unsigned char b_G = cross_avg(b.x, b.y);
        unsigned char b_B = bayer(b.x, b.y);

        // === Write phase ===
        // Writing 3 channels consecutively per pixel keeps each thread's stores
        // in adjacent bytes. Pixel pairs on the same output row (r+gr, gb+b)
        // share cache lines, so the write order matches the output layout.
        auto write_pixel = [&](int px, int py,
                               unsigned char rv, unsigned char gv, unsigned char bv) {
            int base = (y + py) * elemCols + (x + px) * bpp;
            out[base + 0] = rv;
            out[base + 1] = gv;
            out[base + 2] = bv;
        };

        write_pixel(r.x,  r.y,  r_R,  r_G,  r_B);
        write_pixel(gr.x, gr.y, gr_R, gr_G, gr_B);
        write_pixel(gb.x, gb.y, gb_R, gb_G, gb_B);
        write_pixel(b.x,  b.y,  b_R,  b_G,  b_B);
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
