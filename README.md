# OptimizedSparseGEMM

This repository contains several implementations of sparse matrix–matrix multiplication (SGEMM) in Block Compressed Sparse Row (BCSR) format, progressively optimized for AVX2 on x64.

## Files and Versions

1. **`bcsr_sgemm_basic`**  
   - A straightforward, scalar fallback implementation  
   - Used for correctness verification and baseline performance

2. **`bcsr_sgemm_optimized`**  
   - Adds AVX2 vectorization (8-wide FMA), loop-unrolling over both rows and columns  
   - Multi-threaded with OpenMP

3. **`bcsr_sgemm_masked.h`**  
   - **Latest Version 1**:  
     - Fully vectorized AVX2 core  
     - Handles non-multiple-of-8 “tail” columns via _masked_ loads and stores (`_mm256_maskload_ps` / `_mm256_maskstore_ps`), eliminating any scalar remainder loops  

4. **`bcsr_sgemm_padded.h`**  
   - **Latest Version 2**:  
     - Pads every block’s column dimension up to the next multiple of 8  
     - Uses stack-allocated buffers (`Wpad`, `Ypad`) filled with zeros for the padding region  
     - Executes only fixed-width (8-column) AVX2 loops, then copies back the original (unpadded) results

---

## Change Log

- **Mask-Vector Tail Handling**  
  - Introduced a per-block mask based on `c % 8`  
  - Replaced all scalar tail loops with masked AVX2 operations  
  - Removed all branching in the inner multiply-add

- **Zero-Padding Approach**  
  - Compute `c_padded = 8 * ceil(c/8)` at runtime  
  - Copy original weight/output data into padded buffers, zero-fill the remainder  
  - Perform pure 8-wide AVX2 multiply-adds across the padded width  
  - Strip off padding before writing results back to `Y`

These two versions demonstrate alternative strategies to ensure **100% vectorized** AVX2 execution without any scalar cleanup loops. Choose the one that best fits your memory and performance requirements.
