#ifndef BCSR_SGEMM_PADDED_H
#define BCSR_SGEMM_PADDED_H

#include <immintrin.h>
#include <assert.h>
#include <string.h>
#include "bcsr.h"
#include "../dense/dense.h"

/**
 * Sparse GEMM using BCSR format, fully vectorized with AVX2 by
 * padding each block to a multiple of 8 columns, avoiding scalar tails.
 */
void bcsr_sgemm_padded(
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
    // Compute padded column count (multiple of 8)
    const int c_padded = ((c + 7) / 8) * 8;

    // 2) Main loop over rows and blocks
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        const float* Xrow = X + (size_t)m * K;
        for (int brow = 0; brow < W.br; brow++) {
            const float* Xblock = Xrow + brow * r;
            for (int bi = W.b_row_start[brow]; bi < W.b_row_start[brow + 1]; bi++) {
                int bcol = W.b_col_idx[bi];
                float* Yblock = Y + (size_t)m * N + bcol * c;
                const float* Wblock = W.b_values + (size_t)bi * r * c;

                // Allocate temporary padded buffers on stack
                float Ypad[c_padded];
                float Wpad[r * c_padded];

                // 2a) Pad weight block: copy original and zero tail
                for (int i = 0; i < r; i++) {
                    // Copy actual columns
                    memcpy(Wpad + (size_t)i * c_padded,
                           Wblock + (size_t)i * c,
                           c * sizeof(float));
                    // Zero pad remainder
                    memset(Wpad + (size_t)i * c_padded + c,
                           0,
                           (c_padded - c) * sizeof(float));
                }

                // 2b) Load and pad output block
                memcpy(Ypad, Yblock, c * sizeof(float));
                // Zero the padded region beyond c
                memset(Ypad + c, 0, (c_padded - c) * sizeof(float));

                // 2c) Vectorized multiply-add over padded width
                for (int i = 0; i < r; i++) {
                    __m256 x_vec = _mm256_set1_ps(Xblock[i]);
                    for (int j = 0; j < c_padded; j += 8) {
                        __m256 y_vec = _mm256_loadu_ps(Ypad + j);
                        __m256 w_vec = _mm256_loadu_ps(Wpad + (size_t)i * c_padded + j);
                        y_vec = _mm256_fmadd_ps(x_vec, w_vec, y_vec);
                        _mm256_storeu_ps(Ypad + j, y_vec);
                    }
                }

                // 2d) Store back only the original columns
                memcpy(Yblock, Ypad, c * sizeof(float));
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

#endif // BCSR_SGEMM_PADDED_H
