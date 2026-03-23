#pragma once

#include <cstddef>

void debayer_cuda(
    unsigned char* d_bayer,
    unsigned char* d_rgb,
    int width,
    int height
);
