#ifndef BCSR_SGEMM_MASKED_H
#define BCSR_SGEMM_MASKED_H

#include <immintrin.h>
#include <assert.h>
#include "bcsr.h"
#include "../dense/dense.h"

/**
 * Sparse GEMM using BCSR format, fully vectorized with AVX2 and
 * on-the-fly masked loads/stores to handle non-multiple-of-8 tails.
 */
void bcsr_sgemm_optimized(
    const dense_t X, const bcsr_t W, const dense_t B, dense_t Y,
    int M, int N, int K
) {
    // 1) Initialize Y with bias B
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            Y[m * N + n] = B[n];
        }
    }

    const int r = W.r;
    const int c = W.c;
    assert(r > 0 && c > 0);

#ifdef __AVX2__
    // Prepare mask for the tail (c % 8)
    const int tail = c & 7;
    int mask_arr[8] = {
        (tail > 0) ? -1 : 0,
        (tail > 1) ? -1 : 0,
        (tail > 2) ? -1 : 0,
        (tail > 3) ? -1 : 0,
        (tail > 4) ? -1 : 0,
        (tail > 5) ? -1 : 0,
        (tail > 6) ? -1 : 0,
        (tail > 7) ? -1 : 0
    };
    const __m256i mask = _mm256_loadu_si256((const __m256i*)mask_arr);

    // 2) Main computation loop
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        const float* Xrow = X + (size_t)m * K;
        for (int brow = 0; brow < W.br; brow++) {
            const float* Xblock = Xrow + brow * r;
            for (int bi = W.b_row_start[brow]; bi < W.b_row_start[brow + 1]; bi++) {
                int bcol = W.b_col_idx[bi];
                float* Yblock = Y + (size_t)m * N + bcol * c;
                const float* Wblock = W.b_values + (size_t)bi * r * c;

                for (int i = 0; i < r; i++) {
                    __m256 xv = _mm256_set1_ps(Xblock[i]);
                    int j = 0;
                    // Vectorized loop for full 8-wide chunks
                    for (; j + 8 <= c; j += 8) {
                        __m256 yv = _mm256_loadu_ps(Yblock + j);
                        __m256 wv = _mm256_loadu_ps(Wblock + (size_t)i * c + j);
                        __m256 zv = _mm256_fmadd_ps(xv, wv, yv);
                        _mm256_storeu_ps(Yblock + j, zv);
                    }
                    // Masked vector for the remaining tail
                    if (tail) {
                        __m256 yv = _mm256_maskload_ps(Yblock + j, mask);
                        __m256 wv = _mm256_maskload_ps(Wblock + (size_t)i * c + j, mask);
                        __m256 zv = _mm256_fmadd_ps(xv, wv, yv);
                        _mm256_maskstore_ps(Yblock + j, mask, zv);
                    }
                }
            }
        }
    }
#else
    // Fallback scalar implementation
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        const float* Xrow = X + (size_t)m * K;
        for (int brow = 0; brow < W.br; brow++) {
            const float* Xblock = Xrow + brow * r;
            for (int bi = W.b_row_start[brow]; bi < W.b_row_start[brow + 1]; bi++) {
                int bcol = W.b_col_idx[bi];
                float* Yblock = Y + (size_t)m * N + bcol * c;
                const float* Wblock = W.b_values + (size_t)bi * r * c;

                for (int i = 0; i < r; i++) {
                    float xv = Xblock[i];
                    for (int j = 0; j < c; j++) {
                        Yblock[j] += xv * Wblock[i * c + j];
                    }
                }
            }
        }
    }
#endif
}

#endif // BCSR_SGEMM_MASKED_H
