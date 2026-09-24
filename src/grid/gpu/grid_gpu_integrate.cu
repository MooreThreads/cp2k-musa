/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2024 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../../offload/offload_runtime.h"
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_GRID)

#include <algorithm>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__OFFLOAD_MUSA)
#include <time.h>
#endif

#define GRID_DO_COLLOCATE 0
#include "../common/grid_common.h"
#include "grid_gpu_collint.h"
#include "grid_gpu_integrate.h"
#include <stdint.h>

// This has to be included after grid_gpu_collint.h
#include "../common/grid_process_vab.h"

#if defined(_OMP_H)
#error "OpenMP should not be used in .cu files to accommodate HIP."
#endif

#if defined(__OFFLOAD_MUSA)
static int grid_musa_profile_enabled_integrate() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *value = getenv("CP2K_MUSA_GRID_PROFILE");
    enabled = value != NULL && value[0] != '\0' && value[0] != '0';
  }
  return enabled;
}

static double grid_musa_profile_clock_integrate() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}
#endif

// Teen registers are sufficient to integrate lp <= 2 with a single grid sweep.
#define GRID_N_CXYZ_REGISTERS 10

/*******************************************************************************
 * \brief Add value to designated register without using dynamic indexing.
 *        Otherwise the array would be stored in local memory, which is slower.
 * https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda
 * \author Ole Schuett
 ******************************************************************************/
__device__ static inline void
add_to_register(const double value, const int index, cxyz_store *store) {
  switch (index) {
  case 0:
    store->regs[0] += value;
    break;
  case 1:
    store->regs[1] += value;
    break;
  case 2:
    store->regs[2] += value;
    break;
  case 3:
    store->regs[3] += value;
    break;
  case 4:
    store->regs[4] += value;
    break;
  case 5:
    store->regs[5] += value;
    break;
  case 6:
    store->regs[6] += value;
    break;
  case 7:
    store->regs[7] += value;
    break;
  case 8:
    store->regs[8] += value;
    break;
  case 9:
    store->regs[9] += value;
    break;
  }
}

/*******************************************************************************
 * \brief Integrate a single grid point with distance d{xyz} from center.
 * \author Ole Schuett
 ******************************************************************************/
__device__ static void gridpoint_to_cxyz(const double dx, const double dy,
                                         const double dz, const double zetp,
                                         const int lp, const double *gridpoint,
                                         cxyz_store *store) {

  // Squared distance of point from center.
  const double r2 = dx * dx + dy * dy + dz * dz;
  const double gaussian = exp(-zetp * r2);

  // Loading throught read-only cache reduces register usage for some reason.
  const double prefactor = __ldg(gridpoint) * gaussian;

  // Manually unrolled loops based on terms in coset_inv.
  if (store->offset == 0) {
    store->regs[0] += prefactor;
    if (lp >= 1) {
      store->regs[1] += prefactor * dx;
      store->regs[2] += prefactor * dy;
      store->regs[3] += prefactor * dz;
      if (lp >= 2) {
        store->regs[4] += prefactor * dx * dx;
        store->regs[5] += prefactor * dx * dy;
        store->regs[6] += prefactor * dx * dz;
        store->regs[7] += prefactor * dy * dy;
        store->regs[8] += prefactor * dy * dz;
        store->regs[9] += prefactor * dz * dz;
      }
    }

  } else if (store->offset == 10) {
    store->regs[0] += prefactor * dx * dx * dx;
    store->regs[1] += prefactor * dx * dx * dy;
    store->regs[2] += prefactor * dx * dx * dz;
    store->regs[3] += prefactor * dx * dy * dy;
    store->regs[4] += prefactor * dx * dy * dz;
    store->regs[5] += prefactor * dx * dz * dz;
    store->regs[6] += prefactor * dy * dy * dy;
    store->regs[7] += prefactor * dy * dy * dz;
    store->regs[8] += prefactor * dy * dz * dz;
    store->regs[9] += prefactor * dz * dz * dz;

  } else if (store->offset == 20) {
    store->regs[0] += prefactor * dx * dx * dx * dx;
    store->regs[1] += prefactor * dx * dx * dx * dy;
    store->regs[2] += prefactor * dx * dx * dx * dz;
    store->regs[3] += prefactor * dx * dx * dy * dy;
    store->regs[4] += prefactor * dx * dx * dy * dz;
    store->regs[5] += prefactor * dx * dx * dz * dz;
    store->regs[6] += prefactor * dx * dy * dy * dy;
    store->regs[7] += prefactor * dx * dy * dy * dz;
    store->regs[8] += prefactor * dx * dy * dz * dz;
    store->regs[9] += prefactor * dx * dz * dz * dz;

  } else if (store->offset == 30) {
    store->regs[0] += prefactor * dy * dy * dy * dy;
    store->regs[1] += prefactor * dy * dy * dy * dz;
    store->regs[2] += prefactor * dy * dy * dz * dz;
    store->regs[3] += prefactor * dy * dz * dz * dz;
    store->regs[4] += prefactor * dz * dz * dz * dz;
    if (lp >= 5) {
      store->regs[5] += prefactor * dx * dx * dx * dx * dx;
      store->regs[6] += prefactor * dx * dx * dx * dx * dy;
      store->regs[7] += prefactor * dx * dx * dx * dx * dz;
      store->regs[8] += prefactor * dx * dx * dx * dy * dy;
      store->regs[9] += prefactor * dx * dx * dx * dy * dz;
    }

  } else if (store->offset == 40) {
    store->regs[0] += prefactor * dx * dx * dx * dz * dz;
    store->regs[1] += prefactor * dx * dx * dy * dy * dy;
    store->regs[2] += prefactor * dx * dx * dy * dy * dz;
    store->regs[3] += prefactor * dx * dx * dy * dz * dz;
    store->regs[4] += prefactor * dx * dx * dz * dz * dz;
    store->regs[5] += prefactor * dx * dy * dy * dy * dy;
    store->regs[6] += prefactor * dx * dy * dy * dy * dz;
    store->regs[7] += prefactor * dx * dy * dy * dz * dz;
    store->regs[8] += prefactor * dx * dy * dz * dz * dz;
    store->regs[9] += prefactor * dx * dz * dz * dz * dz;

  } else if (store->offset == 50) {
    store->regs[0] += prefactor * dy * dy * dy * dy * dy;
    store->regs[1] += prefactor * dy * dy * dy * dy * dz;
    store->regs[2] += prefactor * dy * dy * dy * dz * dz;
    store->regs[3] += prefactor * dy * dy * dz * dz * dz;
    store->regs[4] += prefactor * dy * dz * dz * dz * dz;
    store->regs[5] += prefactor * dz * dz * dz * dz * dz;
    if (lp >= 6) {
      store->regs[6] += prefactor * dx * dx * dx * dx * dx * dx;
      store->regs[7] += prefactor * dx * dx * dx * dx * dx * dy;
      store->regs[8] += prefactor * dx * dx * dx * dx * dx * dz;
      store->regs[9] += prefactor * dx * dx * dx * dx * dy * dy;
    }

  } else if (store->offset == 60) {
    store->regs[0] += prefactor * dx * dx * dx * dx * dy * dz;
    store->regs[1] += prefactor * dx * dx * dx * dx * dz * dz;
    store->regs[2] += prefactor * dx * dx * dx * dy * dy * dy;
    store->regs[3] += prefactor * dx * dx * dx * dy * dy * dz;
    store->regs[4] += prefactor * dx * dx * dx * dy * dz * dz;
    store->regs[5] += prefactor * dx * dx * dx * dz * dz * dz;
    store->regs[6] += prefactor * dx * dx * dy * dy * dy * dy;
    store->regs[7] += prefactor * dx * dx * dy * dy * dy * dz;
    store->regs[8] += prefactor * dx * dx * dy * dy * dz * dz;
    store->regs[9] += prefactor * dx * dx * dy * dz * dz * dz;

  } else if (store->offset == 70) {
    store->regs[0] += prefactor * dx * dx * dz * dz * dz * dz;
    store->regs[1] += prefactor * dx * dy * dy * dy * dy * dy;
    store->regs[2] += prefactor * dx * dy * dy * dy * dy * dz;
    store->regs[3] += prefactor * dx * dy * dy * dy * dz * dz;
    store->regs[4] += prefactor * dx * dy * dy * dz * dz * dz;
    store->regs[5] += prefactor * dx * dy * dz * dz * dz * dz;
    store->regs[6] += prefactor * dx * dz * dz * dz * dz * dz;
    store->regs[7] += prefactor * dy * dy * dy * dy * dy * dy;
    store->regs[8] += prefactor * dy * dy * dy * dy * dy * dz;
    store->regs[9] += prefactor * dy * dy * dy * dy * dz * dz;

  } else if (store->offset == 80) {
    store->regs[0] += prefactor * dy * dy * dy * dz * dz * dz;
    store->regs[1] += prefactor * dy * dy * dz * dz * dz * dz;
    store->regs[2] += prefactor * dy * dz * dz * dz * dz * dz;
    store->regs[3] += prefactor * dz * dz * dz * dz * dz * dz;
    if (lp >= 7) {
      store->regs[4] += prefactor * dx * dx * dx * dx * dx * dx * dx;
      store->regs[5] += prefactor * dx * dx * dx * dx * dx * dx * dy;
      store->regs[6] += prefactor * dx * dx * dx * dx * dx * dx * dz;
      store->regs[7] += prefactor * dx * dx * dx * dx * dx * dy * dy;
      store->regs[8] += prefactor * dx * dx * dx * dx * dx * dy * dz;
      store->regs[9] += prefactor * dx * dx * dx * dx * dx * dz * dz;
    }

    // Handle higher offsets, ie. values of lp.
  } else {
    for (int i = 0; i < GRID_N_CXYZ_REGISTERS; i++) {
      double val = prefactor;
      const orbital a = coset_inv[i + store->offset];
      for (int j = 0; j < a.l[0]; j++) {
        val *= dx;
      }
      for (int j = 0; j < a.l[1]; j++) {
        val *= dy;
      }
      for (int j = 0; j < a.l[2]; j++) {
        val *= dz;
      }
      add_to_register(val, i, store);
    }
  }
}

/*******************************************************************************
 * \brief Integrates the grid into coefficients C_xyz.
 * \author Ole Schuett
 ******************************************************************************/
__device__ static void grid_to_cxyz(const kernel_params *params,
                                    const smem_task *task, const double *grid,
                                    double *cxyz, double *reduce_scratch) {

  // Atomics adds on shared memory are pretty slow. Hence, the coeffients are
  // accumulated in registers while looping over the grid points.
  // For larger values of lp we need to do multiple sweeps over the grid.
  // Due to the higher register usage and the multiple sweeps,
  // the integrate kernel runs about 70% slower than the collocate kernel.
  for (int offset = 0; offset < ncoset(task->lp);
       offset += GRID_N_CXYZ_REGISTERS) {

    double cxyz_regs[GRID_N_CXYZ_REGISTERS] = {0.0};
    cxyz_store store = {.regs = cxyz_regs, .offset = offset};

    if (task->use_orthorhombic_kernel) {
      ortho_cxyz_to_grid(params, task, &store, grid);
    } else {
      general_cxyz_to_grid(params, task, &store, grid);
    }

    // Reduce register values through shared scratch. This avoids one shared
    // double atomic per thread and coefficient on the MUSA path.
#if defined(__OFFLOAD_MUSA)
    const int tid = threadIdx.x + blockDim.x *
                    (threadIdx.y + blockDim.y * threadIdx.z);
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;
    // Transpose the scratch layout so all threads participate in a tree
    // reduction for every coefficient instead of ten serial accumulators.
#pragma unroll
    for (int i = 0; i < GRID_N_CXYZ_REGISTERS; i++) {
      reduce_scratch[i * nthreads + tid] = cxyz_regs[i];
    }
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
#pragma unroll
        for (int i = 0; i < GRID_N_CXYZ_REGISTERS; i++) {
          reduce_scratch[i * nthreads + tid] +=
              reduce_scratch[i * nthreads + tid + stride];
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < GRID_N_CXYZ_REGISTERS; i++) {
      if (tid == i && i + offset < ncoset(task->lp)) {
        cxyz[i + offset] += reduce_scratch[i * nthreads];
      }
    }
    __syncthreads();
#else
    // Add register values to coefficients stored in shared memory.
#pragma unroll // avoid dynamic indexing of registers
    for (int i = 0; i < GRID_N_CXYZ_REGISTERS; i++) {
      if (i + offset < ncoset(task->lp)) {
        atomicAddDouble(&cxyz[i + offset], cxyz_regs[i]);
      }
    }
#endif
  }
#if defined(__OFFLOAD_MUSA)
  if (params->profile_counters != NULL && threadIdx.x == 0 &&
      threadIdx.y == 0 && threadIdx.z == 0) {
    const int ncoeff = ncoset(task->lp);
    const int nsweeps =
        (ncoeff + GRID_N_CXYZ_REGISTERS - 1) / GRID_N_CXYZ_REGISTERS;
    const unsigned long long nthreads =
        (unsigned long long)blockDim.x * blockDim.y * blockDim.z;
    unsigned long long *counters =
        &params->profile_counters[5 * blockIdx.x];
    counters[1] += (unsigned long long)nsweeps;
    counters[2] += nthreads * (unsigned long long)ncoeff;
  }
#endif
  __syncthreads(); // because of concurrent writes to cxyz
}

/*******************************************************************************
 * \brief Contracts the subblock, going from cartesian harmonics to spherical.
 * \author Ole Schuett
 ******************************************************************************/
template <bool COMPUTE_TAU>
__device__ static void store_hab(const smem_task *task, const cab_store *cab) {

  // The spherical index runs over angular momentum and then over contractions.
  // The carthesian index runs over exponents and then over angular momentum.

#if defined(__OFFLOAD_MUSA)
  // For the common small shell pairs, reduce the four z-lane partial sums in
  // shared memory. This turns four contending global double atomics into one.
  // All threads execute the same two barriers, so both are convergent.
  const int npairs = task->nsgf_seta * task->nsgf_setb;
  if (task->nsgf_seta <= 4 && task->nsgf_setb <= 4 && npairs >= 8) {
    __shared__ double hab_reduce[4][4][4];
    const int i = threadIdx.x;
    const int j = threadIdx.y;
    const bool active = (i < task->nsgf_setb && j < task->nsgf_seta);
    double block_val = 0.0;
    if (active) {
      const int jco_start = ncoset(task->lb_min_basis - 1) + threadIdx.z;
      const int jco_end = ncoset(task->lb_max_basis);
      for (int jco = jco_start; jco < jco_end; jco += blockDim.z) {
        const orbital b = coset_inv[jco];
        const double sphib = task->sphib[i * task->maxcob + jco];
        const int ico_start = ncoset(task->la_min_basis - 1);
        const int ico_end = ncoset(task->la_max_basis);
        for (int ico = ico_start; ico < ico_end; ico++) {
          const orbital a = coset_inv[ico];
          const double hab =
              get_hab(a, b, task->zeta, task->zetb, cab, COMPUTE_TAU);
          const double sphia = task->sphia[j * task->maxcoa + ico];
          block_val += hab * sphia * sphib;
        }
      }
    }
    hab_reduce[threadIdx.x][threadIdx.y][threadIdx.z] = block_val;
    __syncthreads();
    if (active && threadIdx.z == 0) {
      const double sum = hab_reduce[i][j][0] + hab_reduce[i][j][1] +
                         hab_reduce[i][j][2] + hab_reduce[i][j][3];
      if (task->block_transposed) {
        atomicAddDouble(&task->hab_block[j * task->nsgfb + i], sum);
      } else {
        atomicAddDouble(&task->hab_block[i * task->nsgfa + j], sum);
      }
    }
    __syncthreads();
    return;
  }
#endif

  // This is a double matrix product. Since the block can be quite large the
  // two products are fused to conserve shared memory.
  for (int i = threadIdx.x; i < task->nsgf_setb; i += blockDim.x) {
    for (int j = threadIdx.y; j < task->nsgf_seta; j += blockDim.y) {
      double block_val = 0.0;
      const int jco_start = ncoset(task->lb_min_basis - 1) + threadIdx.z;
      const int jco_end = ncoset(task->lb_max_basis);
      for (int jco = jco_start; jco < jco_end; jco += blockDim.z) {
        const orbital b = coset_inv[jco];
        const double sphib = task->sphib[i * task->maxcob + jco];
        const int ico_start = ncoset(task->la_min_basis - 1);
        const int ico_end = ncoset(task->la_max_basis);
        for (int ico = ico_start; ico < ico_end; ico++) {
          const orbital a = coset_inv[ico];
          const double hab =
              get_hab(a, b, task->zeta, task->zetb, cab, COMPUTE_TAU);
          const double sphia = task->sphia[j * task->maxcoa + ico];
          block_val += hab * sphia * sphib;
        }
      }
      if (task->block_transposed) {
        atomicAddDouble(&task->hab_block[j * task->nsgfb + i], block_val);
      } else {
        atomicAddDouble(&task->hab_block[i * task->nsgfa + j], block_val);
      }
    }
  }
  __syncthreads(); // Not needed, but coalesced threads are nice.
}

/*******************************************************************************
 * \brief Adds contributions from cab to forces and virial.
 * \author Ole Schuett
 ******************************************************************************/
template <bool COMPUTE_TAU>
__device__ static void store_forces_and_virial(const kernel_params *params,
                                               const smem_task *task,
                                               const cab_store *cab) {

  for (int i = threadIdx.x; i < task->nsgf_setb; i += blockDim.x) {
    for (int j = threadIdx.y; j < task->nsgf_seta; j += blockDim.y) {
      double block_val;
      if (task->block_transposed) {
        block_val = task->pab_block[j * task->nsgfb + i] * task->off_diag_twice;
      } else {
        block_val = task->pab_block[i * task->nsgfa + j] * task->off_diag_twice;
      }
      const int jco_start = ncoset(task->lb_min_basis - 1) + threadIdx.z;
      const int jco_end = ncoset(task->lb_max_basis);
      for (int jco = jco_start; jco < jco_end; jco += blockDim.z) {
        const double sphib = task->sphib[i * task->maxcob + jco];
        const int ico_start = ncoset(task->la_min_basis - 1);
        const int ico_end = ncoset(task->la_max_basis);
        for (int ico = ico_start; ico < ico_end; ico++) {
          const double sphia = task->sphia[j * task->maxcoa + ico];
          const double pabval = block_val * sphia * sphib;
          const orbital b = coset_inv[jco];
          const orbital a = coset_inv[ico];
          for (int k = 0; k < 3; k++) {
            const double force_a =
                get_force_a(a, b, k, task->zeta, task->zetb, cab, COMPUTE_TAU);
            atomicAddDouble(&task->forces_a[k], force_a * pabval);
            const double force_b = get_force_b(a, b, k, task->zeta, task->zetb,
                                               task->rab, cab, COMPUTE_TAU);
            atomicAddDouble(&task->forces_b[k], force_b * pabval);
          }
          if (params->virial != NULL) {
            for (int k = 0; k < 3; k++) {
              for (int l = 0; l < 3; l++) {
                const double virial_a = get_virial_a(
                    a, b, k, l, task->zeta, task->zetb, cab, COMPUTE_TAU);
                const double virial_b =
                    get_virial_b(a, b, k, l, task->zeta, task->zetb, task->rab,
                                 cab, COMPUTE_TAU);
                const double virial = pabval * (virial_a + virial_b);
                atomicAddDouble(&params->virial[k * 3 + l], virial);
              }
            }
          }
        }
      }
    }
  }
  __syncthreads(); // Not needed, but coalesced threads are nice.
}

/*******************************************************************************
 * \brief Initializes the cxyz matrix with zeros.
 * \author Ole Schuett
 ******************************************************************************/
__device__ static void zero_cxyz(const smem_task *task, double *cxyz) {
  if (threadIdx.z == 0 && threadIdx.y == 0) {
    for (int i = threadIdx.x; i < ncoset(task->lp); i += blockDim.x) {
      cxyz[i] = 0.0;
    }
  }
  __syncthreads(); // because of concurrent writes to cxyz
}

/*******************************************************************************
 * \brief Cuda kernel for integrating all tasks of one grid level.
 * \author Ole Schuett
 ******************************************************************************/
template <bool COMPUTE_TAU, bool CALCULATE_FORCES>
__device__ static void integrate_kernel(const kernel_params *params) {

  // Copy task from global to shared memory and precompute some stuff.
  __shared__ smem_task task;
  load_task(params, &task);

#if defined(__OFFLOAD_MUSA)
  if (params->profile_counters != NULL && threadIdx.x == 0 &&
      threadIdx.y == 0 && threadIdx.z == 0) {
    params->profile_counters[5 * blockIdx.x] = 1;
  }
#endif

  // Check if radius is below the resolution of the grid.
  if (2.0 * task.radius < task.dh_max) {
    return; // nothing to do
  }

  // Allot dynamic shared memory.
  extern __shared__ double shared_memory[];
  double *smem_cab = &shared_memory[params->smem_cab_offset];
  double *smem_alpha = &shared_memory[params->smem_alpha_offset];
  double *smem_cxyz = &shared_memory[params->smem_cxyz_offset];
  double *smem_reduce = &shared_memory[params->smem_reduce_offset];

  // Allocate Cab from global memory if it does not fit into shared memory.
  cab_store cab = {.data = NULL, .n1 = task.n1};
  if (params->smem_cab_length < task.n1 * task.n2) {
#if defined(__OFFLOAD_MUSA)
    cab.data = params->cab_pool + task.cab_offset;
#else
    cab.data = malloc_cab(&task);
#endif
  } else {
    cab.data = smem_cab;
  }

  zero_cab(&cab, task.n1 * task.n2);
  compute_alpha(&task, smem_alpha);

  zero_cxyz(&task, smem_cxyz);
  grid_to_cxyz(params, &task, params->grid, smem_cxyz, smem_reduce);
  cab_to_cxyz(&task, smem_alpha, &cab, smem_cxyz);

  store_hab<COMPUTE_TAU>(&task, &cab);
#if defined(__OFFLOAD_MUSA)
  if (params->profile_counters != NULL && threadIdx.x == 0 &&
      threadIdx.y == 0 && threadIdx.z == 0) {
    params->profile_counters[5 * blockIdx.x + 3] = 1;
  }
#endif
  if (CALCULATE_FORCES) {
    store_forces_and_virial<COMPUTE_TAU>(params, &task, &cab);
#if defined(__OFFLOAD_MUSA)
    if (params->profile_counters != NULL && threadIdx.x == 0 &&
        threadIdx.y == 0 && threadIdx.z == 0) {
      params->profile_counters[5 * blockIdx.x + 4] = 1;
    }
#endif
  }

  if (params->smem_cab_length < task.n1 * task.n2) {
#if !defined(__OFFLOAD_MUSA)
    free_cab(cab.data);
#else
    // CAB pool storage is released by the host after stream completion.
#endif
  }
}

/*******************************************************************************
 * \brief Specialized Cuda kernel for compute_tau=false & calculate_forces=false
 * \author Ole Schuett
 ******************************************************************************/
__global__ static void grid_integrate_density(const kernel_params params) {
  integrate_kernel<false, false>(&params);
}

/*******************************************************************************
 * \brief Specialized Cuda kernel for compute_tau=true & calculate_forces=false.
 * \author Ole Schuett
 ******************************************************************************/
__global__ static void grid_integrate_tau(const kernel_params params) {
  integrate_kernel<true, false>(&params);
}

/*******************************************************************************
 * \brief Specialized Cuda kernel for compute_tau=false & calculate_forces=true.
 * \author Ole Schuett
 ******************************************************************************/
__global__ static void
grid_integrate_density_forces(const kernel_params params) {
  integrate_kernel<false, true>(&params);
}

/*******************************************************************************
 * \brief Specialized Cuda kernel for compute_tau=true & calculate_forces=true.
 * \author Ole Schuett
 ******************************************************************************/
__global__ static void grid_integrate_tau_forces(const kernel_params params) {
  integrate_kernel<true, true>(&params);
}

/*******************************************************************************
 * \brief Launches the Cuda kernel that integrates all tasks of one grid level.
 * \author Ole Schuett
 ******************************************************************************/
void grid_gpu_integrate_one_grid_level(
    const grid_gpu_task_list *task_list, const int first_task,
    const int last_task, const bool compute_tau, const grid_gpu_layout *layout,
    const offloadStream_t stream, const double *pab_blocks_dev,
    const double *grid_dev, double *hab_blocks_dev, double *forces_dev,
    double *virial_dev, int *lp_diff) {

  // Compute max angular momentum.
  const bool calculate_forces = (forces_dev != NULL);
  const bool calculate_virial = (virial_dev != NULL);
  assert(!calculate_virial || calculate_forces);
  const process_ldiffs ldiffs =
      process_get_ldiffs(calculate_forces, calculate_virial, compute_tau);
  *lp_diff = ldiffs.la_max_diff + ldiffs.lb_max_diff; // for reporting stats
  const int la_max = task_list->lmax + ldiffs.la_max_diff;
  const int lb_max = task_list->lmax + ldiffs.lb_max_diff;
  const int lp_max = la_max + lb_max;

  const int ntasks = last_task - first_task + 1;
  if (ntasks == 0) {
    return; // Nothing to do and lp_diff already set.
  }

#if defined(__OFFLOAD_MUSA)
  const int profile = grid_musa_profile_enabled_integrate();
  const double profile_begin =
      profile ? grid_musa_profile_clock_integrate() : 0.0;
#endif

  init_constant_memory();

  // Small Cab blocks are stored in shared mem, larger ones in global memory.
  const int CAB_SMEM_LIMIT = ncoset(5) * ncoset(5); // = 56 * 56 = 3136

  // Compute required shared memory.
  const int alpha_len = 3 * (lb_max + 1) * (la_max + 1) * (lp_max + 1);
  const int cxyz_len = ncoset(lp_max);
  const int cab_len = imin(CAB_SMEM_LIMIT, ncoset(lb_max) * ncoset(la_max));
  const size_t smem_per_block =
      (alpha_len + cxyz_len + cab_len +
       GRID_N_CXYZ_REGISTERS * 4 * 4 * 4) * sizeof(double);

#if defined(__OFFLOAD_MUSA)
  // Build a host-side pool for CAB blocks that do not fit in shared memory.
  const int ntasks_level = last_task - first_task + 1;
  size_t *cab_offsets_host =
      (size_t *)calloc((size_t)ntasks_level, sizeof(size_t));
  if (cab_offsets_host == NULL) {
    fprintf(stderr, "CP2K: failed to allocate MUSA CAB offset table (%d tasks)\n",
            ntasks_level);
    abort();
  }
  size_t cab_pool_len = 0;
  size_t cab_max_len = 0;
  int cab_oversized_tasks = 0;
  for (int ilocal = 0; ilocal < ntasks_level; ilocal++) {
    const grid_gpu_task *host_task =
        &task_list->tasks_host[first_task + ilocal];
    const int n1 = ncoset(host_task->la_max_basis + ldiffs.la_max_diff);
    const int n2 = ncoset(host_task->lb_max_basis + ldiffs.lb_max_diff);
    const size_t required = (size_t)n1 * (size_t)n2;
    if (required > (size_t)CAB_SMEM_LIMIT) {
      if (cab_pool_len > SIZE_MAX - required) {
        fprintf(stderr, "CP2K: MUSA CAB pool size overflow\n");
        abort();
      }
      cab_offsets_host[ilocal] = cab_pool_len;
      cab_pool_len += required;
      cab_oversized_tasks++;
      if (required > cab_max_len) cab_max_len = required;
    }
  }
  size_t *cab_offsets_dev = NULL;
  double *cab_pool = NULL;
  if (cab_oversized_tasks > 0) {
    if (cab_pool_len > SIZE_MAX / sizeof(double)) {
      fprintf(stderr, "CP2K: MUSA CAB pool byte size overflow\n");
      abort();
    }
    offloadMalloc((void **)&cab_offsets_dev,
                  (size_t)ntasks_level * sizeof(size_t));
    offloadMemcpyHtoD(cab_offsets_dev, cab_offsets_host,
                      (size_t)ntasks_level * sizeof(size_t));
    offloadMalloc((void **)&cab_pool, cab_pool_len * sizeof(double));
    const char *cab_debug = getenv("CP2K_MUSA_CAB_POOL_DEBUG");
    if (cab_debug != NULL && cab_debug[0] != '\0' && cab_debug[0] != '0') {
      fprintf(stderr,
              "CP2K MUSA CAB integrate: tasks=%d oversized=%d pool_doubles=%zu max_task_doubles=%zu\n",
              ntasks_level, cab_oversized_tasks, cab_pool_len, cab_max_len);
    }
  }
#else
  size_t *cab_offsets_dev = NULL;
  double *cab_pool = NULL;
#endif

#if defined(__OFFLOAD_MUSA)
  unsigned long long *profile_counters_dev = NULL;
  unsigned long long *profile_counters_host = NULL;
  unsigned long long profile_active_blocks = 0;
  unsigned long long profile_sweeps = 0;
  unsigned long long profile_cxyz_atomic_est = 0;
  unsigned long long profile_hab_blocks = 0;
  unsigned long long profile_force_blocks = 0;
  if (profile) {
    const size_t profile_count = (size_t)ntasks * 5;
    profile_counters_host =
        (unsigned long long *)calloc(profile_count, sizeof(*profile_counters_host));
    if (profile_counters_host == NULL) {
      fprintf(stderr, "CP2K: failed to allocate MUSA GRID profile counters\n");
      abort();
    }
    offloadMalloc((void **)&profile_counters_dev,
                  profile_count * sizeof(*profile_counters_dev));
    offloadMemset(profile_counters_dev, 0,
                  profile_count * sizeof(*profile_counters_dev));
  }
#endif

  // kernel parameters
  kernel_params params;
  params.smem_cab_length = cab_len;
  params.smem_cab_offset = 0;
  params.smem_alpha_offset = params.smem_cab_offset + cab_len;
  params.smem_cxyz_offset = params.smem_alpha_offset + alpha_len;
  params.smem_reduce_offset = params.smem_cxyz_offset + cxyz_len;
#if defined(__OFFLOAD_MUSA)
  params.cab_offsets = cab_offsets_dev;
  params.cab_pool = cab_pool;
  params.profile_counters = profile_counters_dev;
#endif
  params.first_task = first_task;
  params.grid = grid_dev;
  params.tasks = task_list->tasks_dev;
  params.pab_blocks = pab_blocks_dev;
  params.hab_blocks = hab_blocks_dev;
  params.forces = forces_dev;
  params.virial = virial_dev;
  params.la_min_diff = ldiffs.la_min_diff;
  params.lb_min_diff = ldiffs.lb_min_diff;
  params.la_max_diff = ldiffs.la_max_diff;
  params.lb_max_diff = ldiffs.lb_max_diff;
  memcpy(params.dh, layout->dh, 9 * sizeof(double));
  memcpy(params.dh_inv, layout->dh_inv, 9 * sizeof(double));
  memcpy(params.npts_global, layout->npts_global, 3 * sizeof(int));
  memcpy(params.npts_local, layout->npts_local, 3 * sizeof(int));
  memcpy(params.shift_local, layout->shift_local, 3 * sizeof(int));

  // Launch !
  const int nblocks = ntasks;
  const dim3 threads_per_block(4, 4, 4);

#if defined(__OFFLOAD_MUSA)
  const double profile_launch_begin =
      profile ? grid_musa_profile_clock_integrate() : 0.0;
#endif

  if (!compute_tau && !calculate_forces) {
    grid_integrate_density<<<nblocks, threads_per_block, smem_per_block,
                             stream>>>(params);
  } else if (compute_tau && !calculate_forces) {
    grid_integrate_tau<<<nblocks, threads_per_block, smem_per_block, stream>>>(
        params);
  } else if (!compute_tau && calculate_forces) {
    grid_integrate_density_forces<<<nblocks, threads_per_block, smem_per_block,
                                    stream>>>(params);
  } else if (compute_tau && calculate_forces) {
    grid_integrate_tau_forces<<<nblocks, threads_per_block, smem_per_block,
                                stream>>>(params);
  }
  OFFLOAD_CHECK(offloadGetLastError());

#if defined(__OFFLOAD_MUSA)
  const double profile_launch_end =
      profile ? grid_musa_profile_clock_integrate() : 0.0;
#endif

#if defined(__OFFLOAD_MUSA)
  // Keep the normal level launch asynchronous so independent level streams
  // can overlap. A device-side CAB pool and profiling counters are released
  // or copied below, so those two paths still require a local wait.
  if (cab_pool != NULL || profile) {
    offloadStreamSynchronize(stream);
  }
  if (profile) {
    const size_t profile_count = (size_t)ntasks * 5;
    offloadMemcpyDtoH(profile_counters_host, profile_counters_dev,
                      profile_count * sizeof(*profile_counters_host));
    for (int iblock = 0; iblock < ntasks; iblock++) {
      const unsigned long long *counters =
          &profile_counters_host[5 * iblock];
      profile_active_blocks += counters[0];
      profile_sweeps += counters[1];
      profile_cxyz_atomic_est += counters[2];
      profile_hab_blocks += counters[3];
      profile_force_blocks += counters[4];
    }
    offloadFree(profile_counters_dev);
    free(profile_counters_host);
  }
  if (cab_pool != NULL) {
    offloadFree(cab_pool);
  }
  if (cab_oversized_tasks > 0) {
    offloadFree(cab_offsets_dev);
  }
  free(cab_offsets_host);
  if (profile) {
    const double profile_sync_end = grid_musa_profile_clock_integrate();
    fprintf(stderr,
            "CP2K MUSA GRID PROFILE integrate tasks=%d nblocks=%d smem=%zu prep_ms=%.3f launch_ms=%.3f sync_ms=%.3f total_ms=%.3f active_blocks=%llu sweeps=%llu cxyz_atomic_est=%llu hab_blocks=%llu force_blocks=%llu\n",
            ntasks, nblocks, smem_per_block,
            1.0e3 * (profile_launch_begin - profile_begin),
            1.0e3 * (profile_launch_end - profile_launch_begin),
            1.0e3 * (profile_sync_end - profile_launch_end),
            1.0e3 * (profile_sync_end - profile_begin),
            profile_active_blocks, profile_sweeps, profile_cxyz_atomic_est,
            profile_hab_blocks, profile_force_blocks);
  }
#endif
}

#endif // defined(__OFFLOAD) && !defined(__NO_OFFLOAD_GRID)
// EOF
