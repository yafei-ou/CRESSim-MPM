/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2025, Yafei Ou and Mahdi Tavakoli
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CR_GPU_PB_MPM_SOLVER_CUH
#define CR_GPU_PB_MPM_SOLVER_CUH

#include "mpm_solver_gpu.h"
#include "particle_data.h"
#include "preprocessor.h"

namespace crmpm
{
    class GpuPbMpmSolver : public MpmSolverGpu
    {
    public:
        GpuPbMpmSolver(int numParticles, float cellSize, Bounds3 gridBound);
        ~GpuPbMpmSolver() = default;

        void initialize() override;
        float step() override;
        void computeInitialData(unsigned int numParticlesToCompute, const unsigned int *CR_RESTRICT indices) override;
        void setSolverIterations(int n) override { mNumSovlerIterations = n; }

    private:
        int mNumSovlerIterations = 10;

        // Host data
        int mNumNodes;
        Vec3i mNumNodesPerDim;

        // Preset values
        float mInvCellSize;
        float mCellSize;
        float mCellVolume;

        // Linked device data
        float4 *dmParticlePositionMass;
        Vec3f *dmParticleVelocity;
        float4 *dmParticleMaterialProperties0;
        ParticleMaterialType *dmParticleMaterialTypes;

        // Local device data
        float *dmParticleInitialVolume;
        float4 *dmParticleGradientDeformationTensorColumn0;
        float4 *dmParticleGradientDeformationTensorColumn1;
        float4 *dmParticleGradientDeformationTensorColumn2;
        float4 *dmParticleAffineMomentumColumn0;
        float4 *dmParticleAffineMomentumColumn1;
        float4 *dmParticleAffineMomentumColumn2;

        float4 *dmNodeMomentumVelocityMass;

        void resetGrid() override;
        void particleToGrid() override;
        void updateGrid() override;
        void gridToParticle() override;
        void integrateParticle();
        void _release() override;
    };

    CR_CUDA_GLOBAL static void pbMpmComputeInitialGridMassKernel(
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
        float4 *CR_RESTRICT nodeMomentumVelocityMass);

    CR_CUDA_GLOBAL static void pbMpmComputeInitialVolumeKernel(
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
        float *CR_RESTRICT particleInitialVolume);

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
        float4 *CR_RESTRICT nodeMomentumVelocityMass);

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
        float4 *CR_RESTRICT nodeMomentumVelocityMass);

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
        float4 *CR_RESTRICT particleAffineMomentumColumn2);

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
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn2);
}

#endif // ！CR_GPU_PB_MPM_SOLVER_CUH