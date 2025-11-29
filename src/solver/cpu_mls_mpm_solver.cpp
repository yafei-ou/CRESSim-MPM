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

#include "cpu_mls_mpm_solver.h"
#include "debug_logger.h"
#include "shape_functions.h"
#include "mat33.h"
#include "constitutive_models.h"
#include "mls_mpm_algorithm.h"

namespace crmpm
{
    CpuMlsMpmSolver::CpuMlsMpmSolver(float numParticles, float cellSize, Bounds3 gridBound)
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

    void CpuMlsMpmSolver::initialize()
    {
        // Initialize data
        mParticles = new CpuMlsMpmParticleData(mNumMaxParticles);
        mGrid = new CpuMlsMpmNode(mNumNodes);

        mParticles->positionMass = mParticleData->positionMass;
        mParticles->velocity = mParticleData->velocity;

        resetGrid();
    }

    void CpuMlsMpmSolver::computeInitialData(unsigned int numParticlesToCompute,
                                                    const unsigned int *CR_RESTRICT indices)
    {
        for (int i = 0; i < numParticlesToCompute; ++i)
        {
            unsigned int idx = indices[i];
            mParticles->gradientDeformationTensor[idx].setIdentity();

            mlsMpmComputeInitialGridMass<false>(
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
            mlsMpmComputeInitialVolume(
                mGridBound.minimum,
                mInvCellSize,
                mNumNodesPerDim,
                mCellVolume,
                mParticles->positionMass[idx],
                mGrid->mMomentumVelocityMass,
                mParticles->initialVolume[idx]);
        }
    }

    void CpuMlsMpmSolver::resetGrid()
    {
        for (int idx = 0; idx < mGrid->size; ++idx)
        {
            mGrid->mMomentumVelocityMass[idx] = make_float4(0, 0, 0, 0);
        }
    }

    void CpuMlsMpmSolver::particleToGrid()
    {
        // P2G
        for (int idx = 0; idx < mNumActiveParticles; ++idx)
        {
            if (!mActiveMask[idx])
                continue;

            mlsMpmParticleToGrid<false>(
                mIntegrationStepSize,
                mGravity,
                mGridBound.minimum,
                mInvCellSize,
                mCellSize,
                mNumNodesPerDim,
                mParticles->positionMass[idx],
                mParticles->initialVolume[idx],
                mParticles->gradientDeformationTensor[idx],
                mParticles->affineMomentum[idx],
                mParticleMaterialData->params0[idx],
                mParticleMaterialData->type[idx],
                mNumShapes,
                mShapeIds,
                *mShapeData,
                *mGeometryData,
                *mGeometrySdfData,
                mParticles->velocity[idx],
                mGrid->mMomentumVelocityMass);
        }
    }

    void CpuMlsMpmSolver::updateGrid()
    {
        for (int nodeIdx = 0; nodeIdx < mNumNodes; ++nodeIdx)
        {
            mlsMpmUpdateGrid<false>(
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

    void CpuMlsMpmSolver::gridToParticle()
    {
        for (int idx = 0; idx < mNumActiveParticles; ++idx)
        {
            if (!mActiveMask[idx])
                continue;

            mlsMpmGridToParticle<false>(
                mGridBound.minimum,
                mGridBound.maximum,
                mInvCellSize,
                mCellSize,
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
                mParticles->gradientDeformationTensor[idx],
                mParticles->affineMomentum[idx]);
        }
    }

    float CpuMlsMpmSolver::step()
    {
        resetGrid();
        particleToGrid();
        updateGrid();
        gridToParticle();
        return mIntegrationStepSize;
    }

} // namespace crmpm