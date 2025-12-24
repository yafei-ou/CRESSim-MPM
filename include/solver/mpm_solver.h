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

#ifndef CR_MPM_SOLVER_H
#define CR_MPM_SOLVER_H

#include "releasable.h"
#include "particle_data.h"
#include "material_data.h"
#include "bounds3.h"
#include "geometry.h"
#include "shape.h"
#include "engine_export.h"

namespace crmpm
{
    enum class ShapeContactModel
    {
        eKinematic,
        eEffectiveMass,
    };

    enum class MpmSolverType
    {
        eCpuMlsMpmSolver,
        eGpuMlsMpmSolver,

        eCpuPbMpmSolver,
        eGpuPbMpmSolver,

        eCpuStandardMpmSolver,
        eGpuStandardMpmSolver,
    };

    class CRMPM_ENGINE_API MpmSolver : virtual public Releasable
    {
    public:
        /**
         * @brief Initialize the solver.
         */
        virtual void initialize() = 0;

        /**
         * @brief Step the simulation based on the set integration time step.
         */
        virtual float step() = 0;

        /**
         * @brief Wait until all results are available (no effect for CPU solvers)
         */
        virtual void fetchResults() = 0;

        /**
         * @brief Set gravity.
         */
        virtual void setGravity(const Vec3f &gravity) = 0;

        /**
         * @brief Set solver integration step size. For PB-MPM, a larger number can be used.
         */
        virtual void setIntegrationStepSize(float dt) = 0;

        /**
         * @brief Set the solver iterations for PB-MPM. No effect on other solvers.
         * This will affect the stiffness and collision performance.
         * Small numbers causes nonrecoverable deformation after collision due to particle correction.
         */
        virtual void setSolverIterations(int n) = 0;

        /**
         * @breif Compute the initial particle data, such as volume.
         * 
         * @param[in] indices Host or device data for particle indices to compute initial data
         */
        virtual void computeInitialData(unsigned int numParticlesToCompute,
                                        const unsigned int *CR_RESTRICT indices) = 0;

        /**
         * @brief Get the particle data used in the solver.
         * 
         * If the solver is a GPU one, this should return the particle data in the device (with 
         * a CUDA pointer).
         */
        virtual ParticleData &getParticleData() = 0;

        /**
         * @brief Set the particle data SOA for the solver.
         * 
         * If the solver is a GPU one, this should be device data (with a CUDA pointer)
         */
        virtual void bindParticleData(ParticleData &particleData) = 0;

        /**
         * @brief Bind the particle materials SOA for the solver.
         * 
         * If the solver is a GPU one, this should be device data (with a CUDA pointer)
         */
        virtual void bindParticleMaterials(ParticleMaterialData &particleMaterialData) = 0;

        /**
         * @brief Set the number of active particles.
         * 
         * The solver will check the first `numActiveParticles` in the SOAs and see if the
         * indices are active based on the mask set by `MpmSolver::bindActiveMask()`.
         */
        virtual void setNumActiveParticles(unsigned int numActiveParticles) = 0;

        /**
         * @brief Bind the active mask data.
         * 
         * The active mask. This is in addition to `setNumActiveParticles`. If a value in
         * the first `numActiveParticles` elements of the mask is zero, the corresponding
         * particle is inactive.
         */
        virtual void bindActiveMask(unsigned char *activeMask) = 0;

        /**
         * @brief Update the number of shapes
         */
        virtual void setNumShapes(int numShapes) = 0;

        /**
         * @brief Bind a shape ID list. This will be used to retrieve Shape data set by
         * `MpmSolver::bindShapeData()`.
         */
        virtual void bindShapeIds(int *shapeIds) = 0;

        /**
         * @brief Bind ShapeData SOA. The actual shapes used in this solver is set by
         * `MpmSolver::bindShapeIds()`.
         */
        virtual void bindShapeData(ShapeData &shapeData) = 0;

        /**
         * @brief Bind GeometryData SOA. Shapes used in this solver will point to this
         * data for their underlining geometry data.
         */
        virtual void bindGeometryData(GeometryData &geometryData) = 0;

        /**
         * @brief Bind GeometrySdfData SOA. Shapes used in this solver will point to this
         * data for their underlining SDF geometry data.
         */
        virtual void bindGeometrySdfData(GeometrySdfData &geometrySdfData) = 0;

        /**
         * Set `ShapeContactModel` for this solver.
         * 
         * `ShapeContactModel::eKinematic` is the used by default. This method views dynamic
         * rigid bodies as kinematic when resolving contact response.
         * 
         * `ShapeContactModel::eEffectiveMass` is experimental and can give incorrect results.
         * This method calculates the effective mass at the grid node when it is in contact
         * with a dynamic rigid body.
         * 
         * This setting does not affect static and kinematic shapes.
         */
        virtual void setShapeContactModel(ShapeContactModel shapeContactModel) = 0;

    protected:
        MpmSolver() {}
        virtual ~MpmSolver() {}
    };
} // namespace crmpm


#endif // !CR_MPM_SOLVER_H