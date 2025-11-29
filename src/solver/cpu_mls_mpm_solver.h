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

#ifndef CR_CPU_MLS_MPM_SOLVER_H
#define CR_CPU_MLS_MPM_SOLVER_H

#include "mpm_solver_base.h"
#include "cpu_mpm_node_base.h"
#include "mat33.h"

namespace crmpm
{
    struct CpuMlsMpmParticleData : ParticleData
    {
        float *initialVolume;
        Mat3x3f *gradientDeformationTensor;
        Mat3x3f *affineMomentum; // B * D^(1) from APIC

        CpuMlsMpmParticleData(int size) : ParticleData(size)
        { 
            initialVolume = new float[size];
            gradientDeformationTensor = new Mat3x3f[size];
            affineMomentum = new Mat3x3f[size];
        }

        ~CpuMlsMpmParticleData()
        {
            delete[] initialVolume;
            delete[] gradientDeformationTensor;
            delete[] affineMomentum;
        }
    };

    struct CpuMlsMpmNode : CpuMpmNodeBase
    {
        float4 *mMomentumVelocityMass;

        CpuMlsMpmNode(int size) : CpuMpmNodeBase(size)
        {
            mMomentumVelocityMass = new float4[size];
        }

        virtual ~CpuMlsMpmNode() override
        {
            delete[] mMomentumVelocityMass;
        }
    };

    class CpuMlsMpmSolver : public MpmSolverBase
    {
    public:
        CpuMlsMpmSolver(float numParticles, float cellSize, Bounds3 gridBound);
        ~CpuMlsMpmSolver() = default;

        void initialize() override;
        float step() override;
        void computeInitialData(unsigned int numParticlesToCompute, const unsigned int *CR_RESTRICT indices) override;

    private:
        // MPM data
        CpuMlsMpmParticleData *mParticles;
        CpuMlsMpmNode *mGrid;
        int mNumNodes;
        Vec3i mNumNodesPerDim;

        // Preset values
        float mInvCellSize;
        float mCellSize;
        float mCellVolume;

        void resetGrid() override;
        void particleToGrid() override;
        void updateGrid() override;
        void gridToParticle() override;
    };
}

#endif //!CR_CPU_MLS_MPM_SOLVER_H