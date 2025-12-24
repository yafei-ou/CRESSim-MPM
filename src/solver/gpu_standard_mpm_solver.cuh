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

#ifndef CR_GPU_STANDARD_MPM_SOLVER_CUH
#define CR_GPU_STANDARD_MPM_SOLVER_CUH

#include "mpm_solver_gpu.h"
#include "particle_data.h"
#include "preprocessor.h"

namespace crmpm
{
    class GpuStandardMpmSolver : public MpmSolverGpu
    {
    public:
        GpuStandardMpmSolver(int numParticles, float cellSize, Bounds3 gridBound);
        ~GpuStandardMpmSolver() = default;

        void initialize() override;
        float step() override;
        void computeInitialData(unsigned int numParticlesToCompute, const unsigned int *CR_RESTRICT indices) override;

    private:
        // Host data
        int mNumNodes;
        Vec3i mNumNodesPerDim;

        // Preset values
        float mInvCellSize;
        float mCellSize;
        float mGridVolume;

        // Device data (particle)
        float4 *dmParticlePositionMass;
        Vec3f *dmParticleVelocity;
        float *dmParticleInitialVolume;
        float4 *dmParticleGradientDeformationTensorColumn0;
        float4 *dmParticleGradientDeformationTensorColumn1;
        float4 *dmParticleGradientDeformationTensorColumn2;
        float4 *dmParticleMaterialProperties0;
        ParticleMaterialType *dmParticleMaterialTypes;

        // Device data (node)
        float4 *dmNodeMomentumVelocityMass;
        Vec3f *dmNodeForce;

        void resetGrid() override;
        void particleToGrid() override;
        void updateGrid() override;
        void gridToParticle() override;
        void _release() override;
    };

    CR_CUDA_GLOBAL static void standardMpmComputeInitialGridMassKernel(
        const int numParticles,
        const Vec3f gridBoundMin,
        const float invCellSize,
        const Vec3i numNodesPerDim,
        const float gridVolume,
        const float4 *CR_RESTRICT particlePositionMass,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn0,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn1,
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn2,
        float4 *CR_RESTRICT nodeMomentumVelocityMass);

    CR_CUDA_GLOBAL static void standardMpmComputeInitialVolumeKernel(
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
        float *CR_RESTRICT particleInitialVolume);

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
        Vec3f *CR_RESTRICT nodeForce);

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
        float4 *CR_RESTRICT nodeMomentumVelocityMass);

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
        float4 *CR_RESTRICT particleGradientDeformationTensorColumn2);
}

#endif // !CR_GPU_STANDARD_MPM_SOLVER_CUH