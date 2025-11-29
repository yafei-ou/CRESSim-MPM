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

#ifndef CR_MPM_SOLVER_BASE_H
#define CR_MPM_SOLVER_BASE_H

#include "mpm_solver.h"
#include "ref_counted.h"
#include "material_data.h"

namespace crmpm
{
    class MpmSolverBase : public MpmSolver, public RefCounted
    {
    public:
        MpmSolverBase() {}
        ~MpmSolverBase() {}

        virtual void initialize() override = 0;

        virtual float step() override = 0;

        virtual void fetchResults() {};

        void setGravity(const Vec3f &gravity) override
        {
            mGravity = gravity;
        }

        /**
         * @brief Set solver integration step size. For PB-MPM, a larger number can be used.
         */
        virtual void setIntegrationStepSize(float dt) override
        {
            mIntegrationStepSize = dt;
        }

        /**
         * @brief Set the solver iterations for PB-MPM. No effect on other solvers.
         * This will affect the stiffness and collision performance.
         * Small numbers causes nonrecoverable deformation after collision due to particle correction.
         */
        virtual void setSolverIterations(int n) override {}

        /**
         * Compute the initial particle data, such as volume.
         * 
         * @param[in] indices Host or device data for particle indices to compute initial data
         */
        virtual void computeInitialData(unsigned int numParticlesToCompute,
                                        const unsigned int *CR_RESTRICT indices) override = 0;

        ParticleData &getParticleData() override final
        {
            return *mParticleData;
        }

        void bindParticleData(ParticleData &particleData) override
        {
            mParticleData = &particleData;
        }

        void bindParticleMaterials(ParticleMaterialData &particleMaterialData) override
        {
            mParticleMaterialData = &particleMaterialData;
        }

        void setNumActiveParticles(unsigned int numActiveParticles) override
        {
            mNumActiveParticles = numActiveParticles;
        }

        void bindActiveMask(unsigned char *activeMask) override
        {
            mActiveMask = activeMask;
        }

        /**
         * Update shape number
         */
        void setNumShapes(int numShapes) override
        {
            mNumShapes = numShapes;
        }

        /**
         * Bind a shape ID list
         */
        void bindShapeIds(int *shapeIds) override
        {
            mShapeIds = shapeIds;
        }

        void bindShapeData(ShapeData &shapeData) override
        {
            mShapeData = &shapeData;
        };

        void bindGeometryData(GeometryData &geometryData) override
        {
            mGeometryData = &geometryData;
        };

        void bindGeometrySdfData(GeometrySdfData &geometrySdfData) override
        {
            mGeometrySdfData = &geometrySdfData;
        };

    protected:
        // Integration step
        float mIntegrationStepSize = 0.002; // 0.001 if we need E = 1e8 (density ~6000)

        // Gravity
        Vec3f mGravity;

        Bounds3 mGridBound;

        int mNumMaxParticles;
        unsigned int mNumActiveParticles = 0;
        unsigned char *mActiveMask;

        ParticleData *mParticleData;
        ParticleMaterialData *mParticleMaterialData;

        // Rigid body data
        int mNumShapes;
        int *mShapeIds;
        ShapeData *mShapeData;
        GeometryData *mGeometryData;
        GeometrySdfData *mGeometrySdfData;

        virtual void resetGrid() = 0;
        virtual void particleToGrid() = 0;
        virtual void updateGrid() = 0;
        virtual void gridToParticle() = 0;
    };
}

#endif // !CR_MPM_SOLVER_BASE_H