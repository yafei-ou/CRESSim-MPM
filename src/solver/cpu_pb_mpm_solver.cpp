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

#include "cpu_pb_mpm_solver.h"
#include "pb_mpm_algorithm.h"
#include "debug_logger.h"

namespace crmpm
{
    CpuPbMpmSolver::CpuPbMpmSolver(float numParticles, float cellSize, Bounds3 gridBound)
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

    void CpuPbMpmSolver::initialize()
    {
        // Initialize data
        mParticles = new CpuPbMpmParticleData(mNumMaxParticles);
        mGrid = new CpuPbMpmNode(mNumNodes);

        mParticles->positionMass = mParticleData->positionMass;
        mParticles->velocity = mParticleData->velocity;

        resetGrid();
    }

    void CpuPbMpmSolver::computeInitialData(unsigned int numParticlesToCompute,
                                                   const unsigned int *CR_RESTRICT indices)
    {
        for (int i = 0; i < numParticlesToCompute; ++i)
        {
            unsigned int idx = indices[i];
            mParticles->gradientDeformationTensor[idx].setIdentity();

            pbMpmComputeInitialGridMass<false>(
                mGridBound.minimum,
                mInvCellSize,
                mNumNodesPerDim,
                mCellVolume,
                mParticles->positionMass[idx],
                mGrid->mMomentumVelocityMass);
        }

        // G2P for particle volume
        for (int i = 0; i < numParticlesToCompute; ++i)
        {
            unsigned int idx = indices[i];
            pbMpmComputeInitialVolume(
                mGridBound.minimum,
                mInvCellSize,
                mNumNodesPerDim,
                mCellVolume,
                mParticles->positionMass[idx],
                mGrid->mMomentumVelocityMass,
                mParticles->initialVolume[idx]);
        }
    }

    float CpuPbMpmSolver::step()
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

    void CpuPbMpmSolver::resetGrid()
    {
        for (int idx = 0; idx < mGrid->size; ++idx)
        {
            mGrid->mMomentumVelocityMass[idx] = make_float4(0, 0, 0, 0);
        }
    }

    void CpuPbMpmSolver::particleToGrid()
    {
        // P2G
        for (int idx = 0; idx < mNumActiveParticles; ++idx)
        {
            if (!mActiveMask[idx])
                continue;

            pbMpmParticleToGrid<false>(
                mIntegrationStepSize,
                mGridBound.minimum,
                mInvCellSize,
                mCellSize,
                mNumNodesPerDim,
                mParticles->positionMass[idx],
                mParticles->velocity[idx],
                mParticles->gradientDeformationTensor[idx],
                mParticleMaterialData->params0[idx],
                mParticleMaterialData->type[idx],
                mNumShapes,
                mShapeIds,
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                mParticles->affineMomentum[idx],
                mGrid->mMomentumVelocityMass);
        }
    }

    void CpuPbMpmSolver::updateGrid()
    {
        for (int nodeIdx = 0; nodeIdx < mNumNodes; ++nodeIdx)
        {
            pbMpmUpdateGrid<false>(
                mIntegrationStepSize,
                nodeIdx,
                mGridBound.minimum,
                mCellSize,
                mNumNodesPerDim,
                mNumShapes,
                mShapeIds,
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                mGrid->mMomentumVelocityMass);
        }
    }

    void CpuPbMpmSolver::gridToParticle()
    {
        for (int idx = 0; idx < mNumActiveParticles; ++idx)
        {
            if (!mActiveMask[idx])
                continue;

            pbMpmGridToParticle(
                mGridBound.minimum,
                mGridBound.maximum,
                mInvCellSize,
                mCellSize,
                mCellVolume,
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
                mParticles->affineMomentum[idx]);
        }
    }

    void CpuPbMpmSolver::integrateParticle()
    {
        for (int idx = 0; idx < mNumActiveParticles; ++idx)
        {
            if (!mActiveMask[idx])
                continue;

            pbMpmIntegrateParticle<false>(
                mGridBound.minimum,
                mGridBound.maximum,
                mIntegrationStepSize,
                mGravity,
                mParticles->affineMomentum[idx],
                mParticleMaterialData->type[idx],
                mNumShapes,
                mShapeIds,
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                mParticles->positionMass[idx],
                mParticles->velocity[idx],
                mParticles->gradientDeformationTensor[idx]);
        }
    }

} // namespace crmpm
