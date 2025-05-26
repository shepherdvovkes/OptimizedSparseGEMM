#ifndef BCSR_SGEMM_PADDED_ADVANCED_H
#define BCSR_SGEMM_PADDED_ADVANCED_H

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "bcsr.h"
#include "../dense/dense.h"

/// Padded AVX2 SGEMM with packing, alignment, prefetch, register‐blocking and no scalar tails.
void bcsr_sgemm_padded_advanced(
    const dense_t X, const bcsr_t W, const dense_t B, dense_t Y,
    int M, int N, int K
) {
    assert(W.r > 0 && W.c > 0);
    const int r = W.r, c = W.c;
    const int c_pad = ((c + 7) & ~7);

    // allocate aligned W buffers once
    size_t block_bytes = (size_t)r * c_pad * sizeof(float);
    float* Wpad = aligned_alloc(32, W.nnz_blocks * block_bytes);

    // preprocess all blocks into Wpad (packing + zero‐fill)
    for (int bi = 0; bi < W.nnz_blocks; bi++) {
        float* dst = Wpad + bi * r * c_pad;
        const float* src = W.b_values + (size_t)bi * r * c;
        for (int i = 0; i < r; i++) {
            memcpy(dst + i*c_pad, src + i*c, c * sizeof(float));
            memset(dst + i*c_pad + c, 0, (c_pad - c)*sizeof(float));
        }
    }

    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        const float* Xrow = X + (size_t)m * K;
        for (int brow = 0; brow < W.br; brow++) {
            const float* Xblk = Xrow + brow * r;
            for (int bi = W.b_row_start[brow]; bi < W.b_row_start[brow+1]; bi++) {
                int bcol = W.b_col_idx[bi];
                float* Yblk = Y + (size_t)m * N + bcol * c;
                const float* Wblk_p = Wpad + bi * r * c_pad;

                // load bias
                for (int j = 0; j < c; j++) Yblk[j] = B[bcol * c + j];

                // register‐block r=4 (tune as needed)
                for (int i = 0; i < r; i += 4) {
                    __m256 x0 = _mm256_set1_ps(Xblk[i+0]);
                    __m256 x1 = _mm256_set1_ps(Xblk[i+1]);
                    __m256 x2 = _mm256_set1_ps(Xblk[i+2]);
                    __m256 x3 = _mm256_set1_ps(Xblk[i+3]);

                    for (int j = 0; j < c_pad; j += 8) {
                        _mm_prefetch((const char*)(Wblk_p + (i+0)*c_pad + j + 64), _MM_HINT_T0);

                        __m256 yv = _mm256_loadu_ps(Yblk + j);
                        __m256 w0 = _mm256_load_ps(Wblk_p + (i+0)*c_pad + j);
                        __m256 w1 = _mm256_load_ps(Wblk_p + (i+1)*c_pad + j);
                        __m256 w2 = _mm256_load_ps(Wblk_p + (i+2)*c_pad + j);
                        __m256 w3 = _mm256_load_ps(Wblk_p + (i+3)*c_pad + j);

                        yv = _mm256_fmadd_ps(x0, w0, yv);
                        yv = _mm256_fmadd_ps(x1, w1, yv);
                        yv = _mm256_fmadd_ps(x2, w2, yv);
                        yv = _mm256_fmadd_ps(x3, w3, yv);

                        _mm256_storeu_ps(Yblk + j, yv);
                    }
                }
            }
        }
    }

    free(Wpad);
}
#endif // BCSR_SGEMM_PADDED_ADVANCED_H
