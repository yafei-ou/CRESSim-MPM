// BSD 3-Clause License
//
// Copyright (c) 2025, Yafei Ou and Mahdi Tavakoli
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cuda_runtime.h>

#include "gpu_pb_mpm_solver.cuh"
#include "pb_mpm_algorithm.h"
#include "debug_logger.h"
#include "constants.h"
#include "check_cuda.cuh"

namespace crmpm
{
    GpuPbMpmSolver::GpuPbMpmSolver(int numParticles, float cellSize, Bounds3 gridBound)
    {
        // Pre-computed values
        mNumMaxParticles = numParticles;

        mGridBound = gridBound;
        mCellSize = cellSize;
        mInvCellSize = 1 / cellSize;
        Vec3f gridSize = mGridBound.maximum - mGridBound.minimum;
        mNumNodesPerDim = (gridSize / cellSize + 1.0f).cast<int>();
        mNumNodes = mNumNodesPerDim.x() * mNumNodesPerDim.y() * mNumNodesPerDim.z();
        mCellVolume = cellSize * cellSize * cellSize;
    }

    void GpuPbMpmSolver::initialize()
    {
        dmParticlePositionMass = mParticleData->positionMass;
        dmParticleVelocity = mParticleData->velocity;
        dmParticleMaterialProperties0 = mParticleMaterialData->params0;
        dmParticleMaterialTypes = mParticleMaterialData->type;

        // GPU particle data
        CR_CHECK_CUDA(cudaMallocAsync<float>(&dmParticleInitialVolume, mNumMaxParticles * sizeof(float), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleGradientDeformationTensorColumn0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleGradientDeformationTensorColumn1, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleGradientDeformationTensorColumn2, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleAffineMomentumColumn0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleAffineMomentumColumn1, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleAffineMomentumColumn2, mNumMaxParticles * sizeof(float4), mCudaStream));

        // GPU node data
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmNodeMomentumVelocityMass, mNumNodes * sizeof(float4), mCudaStream));

        // Reset all GPU data to zero
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleInitialVolume, 0, mNumMaxParticles * sizeof(float), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleGradientDeformationTensorColumn0, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleGradientDeformationTensorColumn1, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleGradientDeformationTensorColumn2, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleAffineMomentumColumn0, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleAffineMomentumColumn1, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleAffineMomentumColumn2, 0, mNumMaxParticles * sizeof(float4), mCudaStream));

        CR_CHECK_CUDA(cudaMemsetAsync(dmNodeMomentumVelocityMass, 0, mNumNodes * sizeof(float4)));
    }

    void GpuPbMpmSolver::computeInitialData(unsigned int numParticlesToCompute,
                                                   const unsigned int *CR_RESTRICT indices)
    {
        // Initial P2G and G2P for particle volume
        dim3 block(CR_CUDA_THREADS_PER_BLOCK);
        dim3 grid((numParticlesToCompute + block.x - 1) / block.x);
        pbMpmComputeInitialGridMassKernel<<<grid, block, 0, mCudaStream>>>(numParticlesToCompute, indices, mGridBound.minimum,
                                                           mInvCellSize, mNumNodesPerDim, mCellVolume, dmParticlePositionMass,
                                                           dmParticleGradientDeformationTensorColumn0, dmParticleGradientDeformationTensorColumn1,
                                                           dmParticleGradientDeformationTensorColumn2, dmNodeMomentumVelocityMass);
        pbMpmComputeInitialVolumeKernel<<<grid, block, 0, mCudaStream>>>(numParticlesToCompute, indices, mGridBound.minimum,
                                                         mInvCellSize, mNumNodesPerDim, mCellVolume, dmParticlePositionMass,
                                                         dmNodeMomentumVelocityMass,
                                                         dmParticleGradientDeformationTensorColumn0, dmParticleGradientDeformationTensorColumn1,
                                                         dmParticleGradientDeformationTensorColumn2, dmParticleInitialVolume);
    }

    float GpuPbMpmSolver::step()
    {
        for (int i = 0; i < mNumSovlerIterations; i++)
        {
            resetGrid();
            particleToGrid();
            updateGrid();
            gridToParticle();
        }
        integrateParticle();
        return mIntegrationStepSize;
    }

    void GpuPbMpmSolver::_release()
    {
        // Device
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleInitialVolume, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleGradientDeformationTensorColumn0, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleGradientDeformationTensorColumn1, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleGradientDeformationTensorColumn2, mCudaStream));

        CR_CHECK_CUDA(cudaFreeAsync(dmParticleAffineMomentumColumn0, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleAffineMomentumColumn1, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleAffineMomentumColumn2, mCudaStream));

        CR_CHECK_CUDA(cudaFreeAsync(dmNodeMomentumVelocityMass, mCudaStream));
    }

    void GpuPbMpmSolver::resetGrid()
    {
        CR_CHECK_CUDA(cudaMemsetAsync(dmNodeMomentumVelocityMass, 0, mNumNodes * sizeof(Vec3f), mCudaStream));
    }

    void GpuPbMpmSolver::particleToGrid()
    {
        dim3 block(CR_CUDA_THREADS_PER_BLOCK);
        dim3 grid((mNumActiveParticles + block.x - 1) / block.x);
        pbMpmParticleToGridKernel<<<grid, block, 0, mCudaStream>>>(
            mIntegrationStepSize, mGridBound.minimum, mInvCellSize, mCellSize, mNumNodesPerDim,
            mNumActiveParticles, mActiveMask,
            dmParticlePositionMass,
            dmParticleGradientDeformationTensorColumn0, dmParticleGradientDeformationTensorColumn1,
            dmParticleGradientDeformationTensorColumn2,
            dmParticleAffineMomentumColumn0,
            dmParticleAffineMomentumColumn1,
            dmParticleAffineMomentumColumn2,
            dmParticleMaterialProperties0, dmParticleMaterialTypes,
            mNumShapes,
            mShapeIds,
            *mShapeData,
            *mGeometryData,
            *mGeometrySdfData,
            dmParticleVelocity, dmNodeMomentumVelocityMass);
    }

    void GpuPbMpmSolver::updateGrid()
    {
        dim3 block(CR_CUDA_THREADS_PER_BLOCK);
        dim3 grid((mNumNodes + block.x - 1) / block.x);
        if (mShapeContactModel == ShapeContactModel::eKinematic)
        {
            pbMpmUpdateGridKernel<false><<<grid, block, 0, mCudaStream>>>(
                mIntegrationStepSize,
                mNumNodes,
                mGridBound.minimum,
                mCellSize,
                mNumNodesPerDim,
                mNumShapes,
                mShapeIds, // The data is on GPU
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                dmNodeMomentumVelocityMass);
        }
        else
        {
            pbMpmUpdateGridKernel<true><<<grid, block, 0, mCudaStream>>>(
                mIntegrationStepSize,
                mNumNodes,
                mGridBound.minimum,
                mCellSize,
                mNumNodesPerDim,
                mNumShapes,
                mShapeIds, // The data is on GPU
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                dmNodeMomentumVelocityMass);
        }
    }

    void GpuPbMpmSolver::gridToParticle()
    {
        dim3 block(CR_CUDA_THREADS_PER_BLOCK);
        dim3 grid((mNumActiveParticles + block.x - 1) / block.x);
        pbMpmGridToParticleKernel<<<grid, block, 0, mCudaStream>>>(
            mGridBound.minimum,
            mGridBound.maximum,
            mInvCellSize,
            mCellSize,
            mCellVolume,
            mNumNodesPerDim,
            mIntegrationStepSize,
            mNumActiveParticles,
            mActiveMask,
            dmNodeMomentumVelocityMass,
            mNumShapes,
            mShapeIds,
            *mShapeData,
            *mGeometryData,
            *mGeometrySdfData,
            dmParticlePositionMass,
            dmParticleVelocity,
            dmParticleAffineMomentumColumn0,
            dmParticleAffineMomentumColumn1,
            dmParticleAffineMomentumColumn2);
    }

    void GpuPbMpmSolver::integrateParticle()
    {
        dim3 block(CR_CUDA_THREADS_PER_BLOCK);
        dim3 grid((mNumActiveParticles + block.x - 1) / block.x);
        pbMpmIntegrateParticleKernel<<<grid, block>>>(
            mGridBound.minimum,
            mGridBound.maximum,
            mIntegrationStepSize,
            mGravity,
            mNumActiveParticles,
            mActiveMask,
            dmParticleAffineMomentumColumn0,
            dmParticleAffineMomentumColumn1,
            dmParticleAffineMomentumColumn2,
            dmParticleMaterialTypes,
            mNumShapes,
            mShapeIds, // The data is on GPU
            *mShapeData,
            *mGeometryData,
            *mGeometrySdfData,
            dmParticlePositionMass,
            dmParticleVelocity,
            dmParticleGradientDeformationTensorColumn0,
            dmParticleGradientDeformationTensorColumn1,
            dmParticleGradientDeformationTensorColumn2);
    }

    CR_CUDA_GLOBAL void pbMpmComputeInitialGridMassKernel(
        const unsigned int numParticlesToCompute,
        const unsigned int *CR_RESTRICT indices,
        const Vec3f gridBoundMin,
        const float invCellSize,
        const Vec3i numNodesPerDim,
        const float gridVolume,
        const float4 *CR_RESTRICT particlePositionMass,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn0,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn1,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn2,
        float4 *CR_RESTRICT nodeMomentumVelocityMass)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticlesToCompute)
            return;

        unsigned int idx = indices[i];

        // Set deformation gradient tensor to identity
        particleGradientDeformationTensorColumn0[idx] = make_float4(1, 0, 0, 0);
        particleGradientDeformationTensorColumn1[idx] = make_float4(0, 1, 0, 0);
        particleGradientDeformationTensorColumn2[idx] = make_float4(0, 0, 1, 0);

        pbMpmComputeInitialGridMass<true>(
            gridBoundMin,
            invCellSize,
            numNodesPerDim,
            gridVolume,
            particlePositionMass[idx],
            nodeMomentumVelocityMass);
    }

    CR_CUDA_GLOBAL void pbMpmComputeInitialVolumeKernel(
        const unsigned int numParticlesToCompute,
        const unsigned int *CR_RESTRICT indices,
        const Vec3f gridBoundMin,
        const float invCellSize,
        const Vec3i numNodesPerDim,
        const float gridVolume,
        const float4 *CR_RESTRICT particlePositionMass,
        const float4 *CR_RESTRICT nodeMomentumVelocityMass,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn0,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn1,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn2,
        float *CR_RESTRICT particleInitialVolume)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticlesToCompute)
            return;

        unsigned int idx = indices[i];

        // Set deformation gradient tensor to identity
        particleGradientDeformationTensorColumn0[idx] = make_float4(1, 0, 0, 0);
        particleGradientDeformationTensorColumn1[idx] = make_float4(0, 1, 0, 0);
        particleGradientDeformationTensorColumn2[idx] = make_float4(0, 0, 1, 0);

        pbMpmComputeInitialVolume(
            gridBoundMin,
            invCellSize,
            numNodesPerDim,
            gridVolume,
            particlePositionMass[idx],
            nodeMomentumVelocityMass,
            particleInitialVolume[idx]);
    }

    CR_CUDA_GLOBAL void pbMpmParticleToGridKernel(
        const float integrationStepSize,
        const Vec3f gridBoundMin,
        const float invCellSize,
        const float cellSize,
        const Vec3i numNodesPerDim,
        const unsigned int numActiveParticles,
        const unsigned char *CR_RESTRICT activeMask,
        const float4 *CR_RESTRICT particlePositionMass,
        const float4 *CR_RESTRICT particleGradientDeformationTensorColumn0,
        const float4 *CR_RESTRICT particleGradientDeformationTensorColumn1,
        const float4 *CR_RESTRICT particleGradientDeformationTensorColumn2,
        const float4 *CR_RESTRICT particleAffineMomentumColumn0,
        const float4 *CR_RESTRICT particleAffineMomentumColumn1,
        const float4 *CR_RESTRICT particleAffineMomentumColumn2,
        const float4 *CR_RESTRICT particleMaterialProperties0,
        const ParticleMaterialType *CR_RESTRICT particleMaterialTypes,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData shapeData,
        const GeometryData geometryData,
        const GeometrySdfData geometrySdfData,
        Vec3f *CR_RESTRICT particleVelocity,
        float4 *CR_RESTRICT nodeMomentumVelocityMass)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numActiveParticles || !activeMask[idx])
            return;

        // Reconstruct tensors
        Mat3x3f particleDeformationTensor(particleGradientDeformationTensorColumn0[idx],
                                          particleGradientDeformationTensorColumn1[idx],
                                          particleGradientDeformationTensorColumn2[idx]);

        Mat3x3f particleAffineMomentum(particleAffineMomentumColumn0[idx],
                                       particleAffineMomentumColumn1[idx],
                                       particleAffineMomentumColumn2[idx]);

        pbMpmParticleToGrid<true>(
            integrationStepSize,
            gridBoundMin,
            invCellSize,
            cellSize,
            numNodesPerDim,
            particlePositionMass[idx],
            particleVelocity[idx],
            particleDeformationTensor,
            particleMaterialProperties0[idx],
            particleMaterialTypes[idx],
            numShapes,
            shapeIds,
            shapeData,
            geometryData,
            geometrySdfData,
            particleAffineMomentum,
            nodeMomentumVelocityMass);
    }

    template <bool useEffectiveMass>
    CR_CUDA_GLOBAL void pbMpmUpdateGridKernel(
        const float integrationStepSize,
        const int numNodes,
        const Vec3f gridBoundMin,
        const float cellSize,
        const Vec3i numNodesPerDim,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData shapeData,
        const GeometryData geometryData,
        const GeometrySdfData geometrySdfData,
        float4 *CR_RESTRICT nodeMomentumVelocityMass)
    {
        int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (nodeIdx >= numNodes)
            return;

        pbMpmUpdateGrid<true, useEffectiveMass>(
            integrationStepSize,
            nodeIdx,
            gridBoundMin,
            cellSize,
            numNodesPerDim,
            numShapes,
            shapeIds,
            shapeData,
            geometryData,
            geometrySdfData,
            nodeMomentumVelocityMass);
    }

    CR_CUDA_GLOBAL void pbMpmGridToParticleKernel(
        const Vec3f gridBoundMin,
        const Vec3f gridBoundMax,
        const float invCellSize,
        const float cellSize,
        const float mCellVolume,
        const Vec3i numNodesPerDim,
        const float integrationStepSize,
        const unsigned int numActiveParticles,
        const unsigned char *CR_RESTRICT activeMask,
        const float4 *CR_RESTRICT nodeMomentumVelocityMass,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData shapeData,
        const GeometryData geometryData,
        const GeometrySdfData geometrySdfData,
        float4 *CR_RESTRICT particlePositionMass,
        Vec3f *CR_RESTRICT particleVelocity,
        float4 *CR_RESTRICT particleAffineMomentumColumn0,
        float4 *CR_RESTRICT particleAffineMomentumColumn1,
        float4 *CR_RESTRICT particleAffineMomentumColumn2)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numActiveParticles || !activeMask[idx])
            return;

        Mat3x3f particleAffineMomentum(particleAffineMomentumColumn0[idx],
                                       particleAffineMomentumColumn1[idx],
                                       particleAffineMomentumColumn2[idx]);

        pbMpmGridToParticle(
            gridBoundMin,
            gridBoundMax,
            invCellSize,
            cellSize,
            mCellVolume,
            numNodesPerDim,
            integrationStepSize,
            nodeMomentumVelocityMass,
            numShapes,
            shapeIds,
            shapeData,
            geometryData,
            geometrySdfData,
            particlePositionMass[idx],
            particleVelocity[idx],
            particleAffineMomentum);

        particleAffineMomentumColumn0[idx] = particleAffineMomentum.col0;
        particleAffineMomentumColumn1[idx] = particleAffineMomentum.col1;
        particleAffineMomentumColumn2[idx] = particleAffineMomentum.col2;
    }

    CR_CUDA_GLOBAL void pbMpmIntegrateParticleKernel(
        const Vec3f gridBoundMin,
        const Vec3f gridBoundMax,
        const float integrationStepSize,
        const Vec3f gravity,
        const unsigned int numActiveParticles,
        const unsigned char *CR_RESTRICT activeMask,
        const float4 *CR_RESTRICT particleAffineMomentumColumn0,
        const float4 *CR_RESTRICT particleAffineMomentumColumn1,
        const float4 *CR_RESTRICT particleAffineMomentumColumn2,
        const ParticleMaterialType *CR_RESTRICT particleMaterialTypes,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData shapeData,
        const GeometryData geometryData,
        const GeometrySdfData geometrySdfData,
        float4 *CR_RESTRICT particlePositionMass,
        Vec3f *CR_RESTRICT particleVelocity,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn0,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn1,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn2)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numActiveParticles || !activeMask[idx])
            return;

        Mat3x3f particleGradientDeformationTensor(particleGradientDeformationTensorColumn0[idx],
                                                  particleGradientDeformationTensorColumn1[idx],
                                                  particleGradientDeformationTensorColumn2[idx]);

        Mat3x3f particleAffineMomentum(particleAffineMomentumColumn0[idx],
                                       particleAffineMomentumColumn1[idx],
                                       particleAffineMomentumColumn2[idx]);

        pbMpmIntegrateParticle<true>(
            gridBoundMin,
            gridBoundMax,
            integrationStepSize,
            gravity,
            particleAffineMomentum,
            particleMaterialTypes[idx],
            numShapes,
            shapeIds,
            shapeData,
            geometryData,
            geometrySdfData,
            particlePositionMass[idx],
            particleVelocity[idx],
            particleGradientDeformationTensor);

        particleGradientDeformationTensorColumn0[idx] = particleGradientDeformationTensor.col0;
        particleGradientDeformationTensorColumn1[idx] = particleGradientDeformationTensor.col1;
        particleGradientDeformationTensorColumn2[idx] = particleGradientDeformationTensor.col2;
    }
} // namespace crmpm