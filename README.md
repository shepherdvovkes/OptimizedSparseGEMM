# OptimizedSparseGEMM

This repo now contains *four* progressively optimized BCSR SGEMM kernels:

1. **Masked Core (previous)**  
   - On-the-fly AVX2 masked loads/stores for non-multiple-of-8 tails.

2. **Padded Core (previous)**  
   - Zero-padded blocks to the next multiple of 8, pure 8-wide AVX2.

---

## New Advanced Versions

### 3. `bcsr_sgemm_masked_advanced.h`
- **Register‐blocked** over 4 rows at a time.
- **Aligned loads/stores** (`_mm256_load_ps` / `_store_ps`).
- **Software prefetch** inserted for weight and output.
- **Single mask computed once** and reused.
- Minimizes branching in inner loops.

### 4. `bcsr_sgemm_padded_advanced.h`
- **Offline block packing**: all weight blocks pre-padded and aligned.
- **Aligned 32-byte buffers** for prefetch and load/store.
- **Register‐block** of 4 rows per iteration.
- Eliminates `memcpy/memset` in hot loop by moving to preload.
- Simplest inner loop: only `_mm256_fmadd_ps` on aligned data.

---

## Benchmarks & Tuning

- Both versions target x64 AVX2; tune `r` (4) and `c` (8) for your CPU.
- Measure with `std::chrono::high_resolution_clock`, capture GFLOPS.
- Compare memory bandwidth using tools like `perf` or Intel VTune.

Choose the variant that best fits your memory‐compute tradeoffs:
- **Masked** for minimal extra storage & dynamic tails.
- **Padded** for best pure‐vector throughput at slight memory overhead.
