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

#ifndef CR_SCENE_H
#define CR_SCENE_H

#include "preprocessor.h"
#include "releasable.h"
#include "particle_data.h"
#include "material_data.h"
#include "array.h"
#include "gpu_data_dirty_flags.h"
#include "engine_export.h"

namespace crmpm
{
    class Shape;

    class CRMPM_ENGINE_API Scene : virtual public Releasable
    {
    public:
        /**
         * @brief Advances the simulation.
         *
         * @param dt Time step to advance the simulation.
         */
        virtual void advance(float dt) = 0;

        /**
         * @brief Blocks the thread until simulation results are available
         * from the last call to `Scene::advance()`.
         * 
         * If `SimulationFactory.advanceAll()` was called, this automatically
         * calls `SimulationFactory.fetchResultsAll()` and waits until all
         * simulations have been completed.
         */
        virtual void fetchResults() = 0;

        /**
         * @brief Get the allowed maximum number of particles.
         */
        virtual unsigned int getMaxNumParticles() const = 0;

        /**
         * @brief Get the allocated number of particles.
         */
        virtual unsigned int getNumAllocatedParticles() const = 0;

        /**
         * @brief Allocate a group of particles.
         *
         * @param[in] n Number of particles
         * @param[out] offset The index offset of the allocated particle group
         */
        virtual bool allocateParticles(unsigned int n, unsigned int &offset) = 0;

        /**
         * @brief Get the current particle data SOA (in host memory).
         */
        virtual ParticleData &getParticleData() = 0;

        /**
         * @brief Get the particle data offset index in the global shared
         * particle data block managed by SimulationFactory.
         * 
         * This is only useful if the complete particle data block from 
         * SimulationFactory is obtained by `SimulationFactory.getParticleDataAll()`.
         */
        virtual int getParticleDataGlobalOffset() = 0;

        /**
         * @brief Get the current particle data SOA in GPU device memory.
         * 
         * @brief It gives the CUDA device pointer if applicable.
         * 
         * This may return `ParticleData` with invalid pointers, if 
         * the Scene is not a GPU scene.
         */
        virtual ParticleData &getParticleDataGpu() = 0;

        /**
         * @brief Get the particle material SOA for all particles.
         */
        virtual ParticleMaterialData &getParticleMaterialData() = 0;

        /**
         * @brief Get the active particle mask. No modification should be made
         * to this directly. Use `Scene::setActiveMaskRange()` for modification.
         */
        virtual const unsigned char *getActiveMask() const = 0;

        /**
         * @brief Set the active mask with value from start to start + length.
         */
        virtual void setActiveMaskRange(unsigned int start, unsigned int length, char value) = 0;

        /**
         * @brief Indices for computing initial particle data. Modification to this should
         * be followed by `markDirty(SceneDataDirtyFlags::eComputeInitialData)`
         */
        virtual Array<unsigned int> &getComputeInitialDataIndices() = 0;

        /**
         * @brief Mark GPU data dirty. Required after any data on the CPU is modified.
         * 
         * See `SceneDataDirtyFlags`.
         */
        virtual void markDirty(SceneDataDirtyFlags flags) = 0;

        /**
         * @brief Add a Shape to the Scene.
         */
        virtual void addShape(Shape *shape) = 0;

        /**
         * @brief Manually trigger a CPU to GPU data copy if SceneDataDirtyFlags is set
         * 
         * SceneDataDirtyFlags will be rest after calling.
         */
        virtual void syncDataIfNeeded() = 0;

    protected:
        Scene() {}

        virtual ~Scene() {}
    };
} // namespace crmpm

#endif // !CR_SCENE_H
