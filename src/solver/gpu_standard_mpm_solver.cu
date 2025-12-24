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

#include "gpu_standard_mpm_solver.cuh"
#include "standard_mpm_algorithm.h"
#include "constants.h"
#include "check_cuda.cuh"

namespace crmpm
{
    GpuStandardMpmSolver::GpuStandardMpmSolver(int numParticles, float cellSize, Bounds3 gridBound)
    {
        // Pre-computed values for simulation
        mNumMaxParticles = numParticles;

        mGridBound = gridBound;
        mCellSize = cellSize;
        mInvCellSize = 1 / cellSize;
        Vec3f gridSize = mGridBound.maximum - mGridBound.minimum;
        mNumNodesPerDim = (gridSize / cellSize + 1.0f).cast<int>();
        mNumNodes = mNumNodesPerDim.x() * mNumNodesPerDim.y() * mNumNodesPerDim.z();
        mGridVolume = cellSize * cellSize * cellSize;
    }

    void GpuStandardMpmSolver::initialize()
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

        // GPU node data
        CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmNodeMomentumVelocityMass, mNumNodes * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMallocAsync<Vec3f>(&dmNodeForce, mNumNodes * sizeof(Vec3f), mCudaStream));

        // Reset all GPU data to zero
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleInitialVolume, 0, mNumMaxParticles * sizeof(float), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleGradientDeformationTensorColumn0, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleGradientDeformationTensorColumn1, 0, mNumMaxParticles * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmParticleGradientDeformationTensorColumn2, 0, mNumMaxParticles * sizeof(float4), mCudaStream));

        CR_CHECK_CUDA(cudaMemsetAsync(dmNodeMomentumVelocityMass, 0, mNumNodes * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmNodeForce, 0, mNumNodes * sizeof(Vec3f), mCudaStream));
    }

    void GpuStandardMpmSolver::computeInitialData(unsigned int numParticlesToCompute,
                                                         const unsigned int *CR_RESTRICT indices)
    {
        // Initial P2G and G2P for particle volume
        int blockSize = 128;
        dim3 block(blockSize);
        dim3 grid((numParticlesToCompute + block.x - 1) / block.x);
        standardMpmComputeInitialGridMassKernel<<<grid, block, 0, mCudaStream>>>(numParticlesToCompute, mGridBound.minimum, mInvCellSize, mNumNodesPerDim, mGridVolume, dmParticlePositionMass,
                                                                 dmParticleGradientDeformationTensorColumn0, dmParticleGradientDeformationTensorColumn1,
                                                                 dmParticleGradientDeformationTensorColumn2, dmNodeMomentumVelocityMass);
        standardMpmComputeInitialVolumeKernel<<<grid, block, 0, mCudaStream>>>(numParticlesToCompute, mGridBound.minimum, mInvCellSize, mNumNodesPerDim, mGridVolume, dmParticlePositionMass,
                                                               dmNodeMomentumVelocityMass,
                                                               dmParticleGradientDeformationTensorColumn0, dmParticleGradientDeformationTensorColumn1,
                                                               dmParticleGradientDeformationTensorColumn2, dmParticleInitialVolume);
    }

    void GpuStandardMpmSolver::resetGrid()
    {
        CR_CHECK_CUDA(cudaMemsetAsync(dmNodeMomentumVelocityMass, 0, mNumNodes * sizeof(float4), mCudaStream));
        CR_CHECK_CUDA(cudaMemsetAsync(dmNodeForce, 0, mNumNodes * sizeof(Vec3f), mCudaStream));
    }

    void GpuStandardMpmSolver::particleToGrid()
    {
        int blockSize = 128;
        dim3 block(blockSize);
        dim3 grid((mNumActiveParticles + block.x - 1) / block.x);
        standardMpmParticleToGridKernel<<<grid, block, 0, mCudaStream>>>(mNumActiveParticles, mGravity, mGridBound.minimum, mInvCellSize, mNumNodesPerDim,
                                                         dmParticlePositionMass, dmParticleVelocity, dmParticleInitialVolume,
                                                         dmParticleGradientDeformationTensorColumn0, dmParticleGradientDeformationTensorColumn1,
                                                         dmParticleGradientDeformationTensorColumn2,
                                                         dmParticleMaterialProperties0, dmParticleMaterialTypes,
                                                         dmNodeMomentumVelocityMass, dmNodeForce);
    }

    void GpuStandardMpmSolver::updateGrid()
    {
        int blockSize = 128;
        dim3 block(blockSize);
        dim3 grid((mNumNodes + block.x - 1) / block.x);
        if (mShapeContactModel == ShapeContactModel::eKinematic)
        {
            standardMpmUpdateGridKernel<false><<<grid, block, 0, mCudaStream>>>(
                mNumNodes,
                mGridBound.minimum,
                mCellSize,
                mNumNodesPerDim,
                mIntegrationStepSize,
                dmNodeForce,
                mNumShapes,
                mShapeIds,
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                dmNodeMomentumVelocityMass);
        }
        else
        {
            standardMpmUpdateGridKernel<true><<<grid, block, 0, mCudaStream>>>(
                mNumNodes,
                mGridBound.minimum,
                mCellSize,
                mNumNodesPerDim,
                mIntegrationStepSize,
                dmNodeForce,
                mNumShapes,
                mShapeIds,
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                dmNodeMomentumVelocityMass);
        }
    }

    void GpuStandardMpmSolver::gridToParticle()
    {
        int blockSize = 128;
        dim3 block(blockSize);
        dim3 grid((mNumActiveParticles + block.x - 1) / block.x);
        standardMpmGridToParticleKernel<<<grid, block, 0, mCudaStream>>>(
            mNumActiveParticles,
            mGridBound.minimum,
            mGridBound.maximum,
            mInvCellSize,
            mNumNodesPerDim,
            mIntegrationStepSize,
            dmNodeMomentumVelocityMass,
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

    float GpuStandardMpmSolver::step()
    {
        resetGrid();
        particleToGrid();
        updateGrid();
        gridToParticle();
        return mIntegrationStepSize;
    }

    void GpuStandardMpmSolver::_release()
    {
        // Device
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleInitialVolume, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleGradientDeformationTensorColumn0, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleGradientDeformationTensorColumn1, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmParticleGradientDeformationTensorColumn2, mCudaStream));

        CR_CHECK_CUDA(cudaFreeAsync(dmNodeMomentumVelocityMass, mCudaStream));
        CR_CHECK_CUDA(cudaFreeAsync(dmNodeForce, mCudaStream));
    }

    CR_CUDA_GLOBAL void standardMpmComputeInitialGridMassKernel(
        const int numParticles,
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
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numParticles)
            return;

        // Set deformation gradient tensor to identity
        particleGradientDeformationTensorColumn0[idx] = make_float4(1, 0, 0, 0);
        particleGradientDeformationTensorColumn1[idx] = make_float4(0, 1, 0, 0);
        particleGradientDeformationTensorColumn2[idx] = make_float4(0, 0, 1, 0);

        standardMpmComputeInitialGridMass<true>(
            gridBoundMin,
            invCellSize,
            numNodesPerDim,
            gridVolume,
            particlePositionMass[idx],
            nodeMomentumVelocityMass);
    }

    CR_CUDA_GLOBAL void standardMpmComputeInitialVolumeKernel(
        const int numParticles,
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
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numParticles)
            return;

        // Set deformation gradient tensor to identity
        particleGradientDeformationTensorColumn0[idx] = make_float4(1, 0, 0, 0);
        particleGradientDeformationTensorColumn1[idx] = make_float4(0, 1, 0, 0);
        particleGradientDeformationTensorColumn2[idx] = make_float4(0, 0, 1, 0);

        standardMpmComputeInitialVolume(
            gridBoundMin,
            invCellSize,
            numNodesPerDim,
            gridVolume,
            particlePositionMass[idx],
            nodeMomentumVelocityMass,
            particleInitialVolume[idx]);
    }

    /**
     * @brief Standard MPM P2G kernel
     */
    CR_CUDA_GLOBAL void standardMpmParticleToGridKernel(
        const int numParticles,
        const Vec3f gravity,
        const Vec3f gridBoundMin,
        const float invCellSize,
        const Vec3i numNodesPerDim,
        const float4 *CR_RESTRICT particlePositionMass,
        const Vec3f *CR_RESTRICT particleVelocity,
        const float *CR_RESTRICT particleInitialVolume,
        const float4 *CR_RESTRICT particleGradientDeformationTensorColumn0,
        const float4 *CR_RESTRICT particleGradientDeformationTensorColumn1,
        const float4 *CR_RESTRICT particleGradientDeformationTensorColumn2,
        const float4 *CR_RESTRICT particleMaterialProperties0,
        const ParticleMaterialType *CR_RESTRICT particleMaterialTypes,
        float4 *CR_RESTRICT nodeMomentumVelocityMass,
        Vec3f *CR_RESTRICT nodeForce)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numParticles)
            return;

        // Reconstruct tensors
        Mat3x3f particleDeformationTensor(particleGradientDeformationTensorColumn0[idx],
                                          particleGradientDeformationTensorColumn1[idx],
                                          particleGradientDeformationTensorColumn2[idx]);

        standardMpmParticleToGrid<true>(
            gravity,
            gridBoundMin,
            invCellSize,
            numNodesPerDim,
            particlePositionMass[idx],
            particleVelocity[idx],
            particleInitialVolume[idx],
            particleDeformationTensor,
            particleMaterialProperties0[idx],
            particleMaterialTypes[idx],
            nodeMomentumVelocityMass,
            nodeForce);
    }

    template <bool useEffectiveMass>
    CR_CUDA_GLOBAL void standardMpmUpdateGridKernel(
        const int numNodes,
        const Vec3f gridBoundMin,
        const float cellSize,
        const Vec3i numNodesPerDim,
        const float integrationStepSize,
        const Vec3f *CR_RESTRICT nodeForce,
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

        standardMpmUpdateGrid<true, useEffectiveMass>(
            nodeIdx,
            gridBoundMin,
            cellSize,
            numNodesPerDim,
            integrationStepSize,
            nodeForce,
            numShapes,
            shapeIds,
            shapeData,
            geometryData,
            geometrySdfData,
            nodeMomentumVelocityMass);
    }

    CR_CUDA_GLOBAL void standardMpmGridToParticleKernel(
        const int numParticles,
        const Vec3f gridBoundMin,
        const Vec3f gridBoundMax,
        const float invCellSize,
        const Vec3i numNodesPerDim,
        const float integrationStepSize,
        const float4 *CR_RESTRICT nodeMomentumVelocityMass,
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
        if (idx >= numParticles)
            return;

        Mat3x3f particleGradientDeformationTensor(particleGradientDeformationTensorColumn0[idx],
                                                  particleGradientDeformationTensorColumn1[idx],
                                                  particleGradientDeformationTensorColumn2[idx]);

        standardMpmGridToParticle<true>(
            gridBoundMin,
            gridBoundMax,
            invCellSize,
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
            particleGradientDeformationTensor);

        particleGradientDeformationTensorColumn0[idx] = particleGradientDeformationTensor.col0;
        particleGradientDeformationTensorColumn1[idx] = particleGradientDeformationTensor.col1;
        particleGradientDeformationTensorColumn2[idx] = particleGradientDeformationTensor.col2;
    }
} // namespace crmpm
