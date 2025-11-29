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

#ifndef CR_SCENE_IMPL_H
#define CR_SCENE_IMPL_H

#include "scene.h"
#include "simulation_status.h"
#include "preprocessor.h"
#include "ref_counted.h"
#include "particle_data.h"
#include "mpm_solver_base.h"
#include "array.h"
#include "single_thread_manager.h"
#include "gpu_data_dirty_flags.h"


// Forward declare CUstream_st and cudaStream_t
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace crmpm
{
    class ShapeImpl;
    class SimulationFactoryImpl;

    class SceneImpl : public Scene, public RefCounted
    {
    public:

        /**
         * Advance simulation by dt
         */
        void advance(float dt) override; // simulation_factory.h must be included.

        void fetchResults() override;

        CR_FORCE_INLINE unsigned int getMaxNumParticles() const override
        {
            return mMaxNumParticles;
        }

        CR_FORCE_INLINE unsigned int getNumAllocatedParticles() const override
        {
            return mNumAllocatedParticles;
        }

        bool allocateParticles(unsigned int n, unsigned int &offset) override
        {
            const int newNum = mNumAllocatedParticles + n;
            if (newNum > mMaxNumParticles)
                return false;
            offset = mNumAllocatedParticles;
            mNumAllocatedParticles = newNum;
            return true;
        }

        CR_FORCE_INLINE ParticleData &getParticleData() override
        {
            return mCpuParticleData;
        }

        CR_FORCE_INLINE int getParticleDataGlobalOffset() override
        {
            return mParticleDataOffset;
        }

        /**
         * @brief This may return ParticleData with invalid pointers, if 
         * the Scene is not a GPU scene.
         */
        CR_FORCE_INLINE ParticleData &getParticleDataGpu() override
        {
            return mGpuParticleData;
        }

        CR_FORCE_INLINE ParticleMaterialData &getParticleMaterialData() override
        {
            return mCpuParticleMaterialData;
        }

        /**
         * @brief Active particle mask. No modification should be made to this directly.
         * Use setActiveMaskRange instead.
         */
        CR_FORCE_INLINE const unsigned char *getActiveMask() const override
        {
            return mActiveParticleMask;
        }

        /**
         * @brief Set the active mask with value from start to start + length.
         */
        void setActiveMaskRange(unsigned int start, unsigned int length, char value) override;

        /**
         * @brief Indices for computing initial particle data. Modification to this should be followed by
         * markDirty(SceneDataDirtyFlags::eComputeInitialData)
         */
        CR_FORCE_INLINE Array<unsigned int> &getComputeInitialDataIndices() override
        {
            return mComputeInitialDataIndices;
        }

        CR_FORCE_INLINE void markDirty(SceneDataDirtyFlags flags) override
        {
            mDataDirtyFlags |= flags;
        }

        void addShape(Shape *shape) override;

        void syncDataIfNeeded() override;

    protected:
        SceneImpl(unsigned int maxNumParticles,
                  unsigned int shapeCapacity,
                  float4 *particlePositionMass,
                  Vec3f *particleVelocity,
                  int particleDataOffset)
            : mMaxNumParticles(maxNumParticles),
              mParticlePositionMass(particlePositionMass + particleDataOffset),
              mParticleVelocity(particleVelocity + particleDataOffset),
              mParticleDataOffset(particleDataOffset),
              mNumAllocatedParticles(0),
              mCpuParticleData(maxNumParticles),
              mCpuParticleMaterialData(maxNumParticles),
              mGpuParticleData(maxNumParticles),
              mGpuParticleMaterialData(maxNumParticles),
              mStatus(SimulationStatus::eIdle),
              mShapeCapacity(shapeCapacity),
              mShapeIds(shapeCapacity),
              mAdvanceTaskFn(std::bind(&SceneImpl::_advanceTaskFn, this, std::placeholders::_1)) {}

        ~SceneImpl() = default;

        void initialize();

        CR_FORCE_INLINE void setFactory(SimulationFactoryImpl &factory) { mFactory = &factory; }

        void setSolver(MpmSolverBase &solver, bool isGpu);

        void beforeAdvance();

        void afterAdvance();

        SimulationStatus getSimulationStatus();

        void setSimulationStatus(SimulationStatus status);

    private:
        SimulationFactoryImpl *mFactory;

        // Simulation status
        SimulationStatus mStatus;

        // Solver type
        bool mIsGpu;

        // Gpu data dirty flags
        SceneDataDirtyFlags mDataDirtyFlags;

        // Particle data

        unsigned int mMaxNumParticles;
        unsigned int mNumAllocatedParticles;
        unsigned char *mActiveParticleMask;
        unsigned char *dmActiveParticleMask;

        Array<unsigned int> mComputeInitialDataIndices;
        unsigned int *dmComputeInitialDataIndices;

        // Particle data SoAs
        ParticleData mCpuParticleData;
        ParticleMaterialData mCpuParticleMaterialData;

        int mParticleDataOffset;
        float4 *mParticlePositionMass; // after applying offset
        Vec3f *mParticleVelocity;      // after applying offset
        float4 *mParticleMaterialPropertiesGroup1;
        ParticleMaterialType *mParticleMaterialTypes;

        // Scene manages GPU common data, which can be modifed by user.
        ParticleData mGpuParticleData;
        ParticleMaterialData mGpuParticleMaterialData;
        float4 *dmParticlePositionMass;
        Vec3f *dmParticleVelocity;
        float4 *dmParticleMaterialPropertiesGroup1;
        ParticleMaterialType *dmParticleMaterialTypes;

        // Shape indices
        int mShapeCapacity;
        Array<int> mShapeIds;
        int* dmShapeIds;
        Array<ShapeImpl *> mShapes;

        // Solver
        MpmSolverBase *mSolver;

        // CUDA stream
        cudaStream_t mCudaStream;

        // A separate thread for simulation solving
        SingleThreadManager mThreadManager;
        Task mAdvanceTaskFn;

        void _release() override;

        void _advanceTaskFn(void *data);

        friend class SimulationFactoryImpl;
    };
} // namespace crmpm

#endif // !CR_SCENE_IMPL_H
