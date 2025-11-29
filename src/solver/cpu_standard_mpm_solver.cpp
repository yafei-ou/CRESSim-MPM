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

#include "cpu_standard_mpm_solver.h"
#include "debug_logger.h"
#include "shape_functions.h"
#include "constitutive_models.h"
#include "standard_mpm_algorithm.h"

namespace crmpm
{
    CpuStandardMpmSolver::CpuStandardMpmSolver(float numParticles, float cellSize, Bounds3 gridBound)
    {
        // Pre-computed values for simulation
        mNumMaxParticles = numParticles;

        mGridBound = gridBound;
        mCellSize = cellSize;
        mInvCellSize = 1 / cellSize;
        Vec3f gridSize = mGridBound.maximum - mGridBound.minimum;
        mNumNodesPerDim = (gridSize / cellSize + 1.0f).cast<int>();
        mNumNodes = mNumNodesPerDim.x() * mNumNodesPerDim.y() * mNumNodesPerDim.z();
        mCellVolume = cellSize * cellSize * cellSize;
    }

    void CpuStandardMpmSolver::initialize()
    {
        // Initialize data
        mParticles = new CpuStandardMpmParticleData(mNumMaxParticles);
        mGrid = new CpuStandardMpmNode(mNumNodes);

        mParticles->positionMass = mParticleData->positionMass;
        mParticles->velocity = mParticleData->velocity;

        resetGrid();
    }

    void CpuStandardMpmSolver::computeInitialData(unsigned int numParticlesToCompute, const unsigned int *CR_RESTRICT indices)
    {
        for (int i = 0; i < numParticlesToCompute; ++i)
        {
            unsigned int idx = indices[i];
            mParticles->gradientDeformationTensor[idx].setIdentity();
            standardMpmComputeInitialGridMass<false>(
                mGridBound.minimum,
                mInvCellSize,
                mNumNodesPerDim,
                mCellVolume,
                mParticles->positionMass[idx],
                mGrid->mMomentumVelocityMass);
        }
        for (int i = 0; i < numParticlesToCompute; ++i)
        {
            unsigned int idx = indices[i];
            standardMpmComputeInitialVolume(
                mGridBound.minimum,
                mInvCellSize,
                mNumNodesPerDim,
                mCellVolume,
                mParticles->positionMass[idx],
                mGrid->mMomentumVelocityMass,
                mParticles->initialVolume[idx]);
        }
    }

    void CpuStandardMpmSolver::resetGrid()
    {
        for (int idx = 0; idx < mGrid->size; ++idx)
        {
            mGrid->mForce[idx] = Vec3f(0, 0, 0);
            mGrid->mMomentumVelocityMass[idx] = make_float4(0, 0, 0, 0);
        }
    }

    void CpuStandardMpmSolver::particleToGrid()
    {
        // P2G
        for (int idx = 0; idx < mNumActiveParticles; ++idx)
        {
            standardMpmParticleToGrid<false>(
                mGravity,
                mGridBound.minimum,
                mInvCellSize,
                mNumNodesPerDim,
                mParticles->positionMass[idx],
                mParticles->velocity[idx],
                mParticles->initialVolume[idx],
                mParticles->gradientDeformationTensor[idx],
                mParticleMaterialData->params0[idx],
                mParticleMaterialData->type[idx],
                mGrid->mMomentumVelocityMass,
                mGrid->mForce);
        }
    }

    void CpuStandardMpmSolver::updateGrid()
    {
        for (int nodeIdx = 0; nodeIdx < mNumNodes; ++nodeIdx)
        {
            standardMpmUpdateGrid<false>(
                nodeIdx,
                mGridBound.minimum,
                mCellSize,
                mNumNodesPerDim,
                mIntegrationStepSize,
                mGrid->mForce,
                mNumShapes,
                mShapeIds,
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                mGrid->mMomentumVelocityMass);
        }
    }

    void CpuStandardMpmSolver::gridToParticle()
    {
        for (int idx = 0; idx < mNumActiveParticles; ++idx)
        {
            if (!mActiveMask[idx])
                continue;

            standardMpmGridToParticle<false>(
                mGridBound.minimum,
                mGridBound.maximum,
                mInvCellSize,
                mNumNodesPerDim,
                mIntegrationStepSize,
                mGrid->mMomentumVelocityMass,
                mNumShapes,
                mShapeIds, // The data is on GPU
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                mParticles->positionMass[idx],
                mParticles->velocity[idx],
                mParticles->gradientDeformationTensor[idx]);
        }
    }

    float CpuStandardMpmSolver::step()
    {
        resetGrid();
        particleToGrid();
        updateGrid();
        gridToParticle();
        return mIntegrationStepSize;
    }
} // namespace crmpm
