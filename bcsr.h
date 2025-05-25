#include <immintrin.h>
#include "bcsr.h"
#include "../dense/dense.h"

// Tuned block sizes for x64 (256-bit AVX2, 8 floats per vector)
#define BLOCK_R 4  // block rows
#define BLOCK_C 8  // block columns

void bcsr_sgemm_optimized(
    const dense_t X, const bcsr_t W, const dense_t B, dense_t Y,
    int M, int N, int K
) {
    // 1) Initialize output Y with bias B
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            Y[m * N + n] = B[n];
        }
    }

    const int r = BLOCK_R;
    const int c = BLOCK_C;

    // Ensure the weight matrix uses the tuned block size
    assert(W.r == BLOCK_R && W.c == BLOCK_C);

    // 2) Main loop: for each row of X and each block-row of W
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        const float* Xrow_base = X + m * K;

        for (int brow = 0; brow < W.br; brow++) {
            const float* Xblock = Xrow_base + brow * r;

            for (int bi = W.b_row_start[brow]; bi < W.b_row_start[brow + 1]; bi++) {
                int bcol = W.b_col_idx[bi];
                float* Yblock = Y + m * N + bcol * c;
                const float* Wblock = W.b_values + bi * (r * c);

#ifdef __AVX2__
                int i = 0;
                // Unroll rows: process two block-rows at a time
                for (; i + 1 < r; i += 2) {
                    __m256 x0 = _mm256_set1_ps(Xblock[i]);
                    __m256 x1 = _mm256_set1_ps(Xblock[i + 1]);
                    int j = 0;
                    for (; j + c <= c; j += c) {
                        __m256 ycur = _mm256_loadu_ps(Yblock + j);
                        __m256 w0 = _mm256_loadu_ps(Wblock + i * c + j);
                        __m256 w1 = _mm256_loadu_ps(Wblock + (i + 1) * c + j);
                        ycur = _mm256_fmadd_ps(x0, w0, ycur);
                        ycur = _mm256_fmadd_ps(x1, w1, ycur);
                        _mm256_storeu_ps(Yblock + j, ycur);
                    }
                }
                // if one row remains
                for (; i < r; i++) {
                    __m256 xv = _mm256_set1_ps(Xblock[i]);
                    __m256 wv = _mm256_loadu_ps(Wblock + i * c);
                    __m256 ycur = _mm256_loadu_ps(Yblock);
                    ycur = _mm256_fmadd_ps(xv, wv, ycur);
                    _mm256_storeu_ps(Yblock, ycur);
                }
#else
                // Fallback: scalar implementation
                for (int i = 0; i < r; i++) {
                    float xv = Xblock[i];
                    for (int j = 0; j < c; j++) {
                        Yblock[j] += xv * Wblock[i * c + j];
                    }
                }
#endif
            }
        }
    }

    // Note: tuned for BLOCK_R=4, BLOCK_C=8 on x64 AVX2 platforms.
}
