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

#ifndef CR_SIMULATION_FACTORY_H
#define CR_SIMULATION_FACTORY_H

#include "preprocessor.h"
#include "geometry.h"
#include "mpm_solver.h"
#include "shape.h"
#include "scene.h"
#include "particle_object.h"
#include "array.h"
#include "triangle_mesh.h"
#include "sdf_builder.h"
#include "gpu_data_dirty_flags.h"
#include "engine_export.h"

namespace crmpm
{
    struct SceneDesc
    {
        MpmSolverType solverType;                                       // MPM solver type
        unsigned int numMaxParticles;                                   // Maximum number of particles
        Vec3f gravity = Vec3f(0, -9.81f, 0);                            // Gravity
        Bounds3 gridBounds;                                             // MPM grid bounds
        float gridCellSize;                                             // MPM grid cell size
        float solverIntegrationStepSize = 0.002f;                       // Integration step size
        unsigned int solverIterations = 10;                             // Only for PB-MPM: solver iterations
        ShapeContactModel contactModel = ShapeContactModel::eKinematic; // Contact model between MPM and dynamic rigid shapes
    };

    struct ShapeDesc
    {
        ShapeType type;               // Shape type. See `ShapeType`
        Transform transform;          // Initial pose
        Vec3f invScale = Vec3f(1.0f); // Inverse scale

        float sdfSmoothDistance = 0.05f; // SDF smooth distance (a fattened area for contact)
        float sdfFatten = 0.0f;          // SDF fatten distance (fatten the actual distance)
        float drag = 1.0f;               // Drag force, a damping force during contact
        float friction = 0.0f;           // Dynamic friction coefficient

        float4 comInvMass;  // Center of mass + inverse mass
        float4 inertiaInv0; // Inertia tensor inverse (diagonal component)
        float4 inertiaInv1; // Inertia tensor inverse (off-diagonal component)
    };

    struct ParticleObjectDesc
    {
        float particleSpacing = 0.01f; // Particle spacing
        float particleMass;            // Particle mass

        Geometry *geometry; // Geometry for constructing a particle group
        Vec3f position;     // World coordinate position of the geometry
        Quat rotation;      // World coordinate rotation of the geometry
        Vec3f invScale;     // World coordinate inverse scale of the geometry

        ParticleMaterialType materialType; // Particle material type
        float4 materialParams;             // Particle material parameters
    };

    class CRMPM_ENGINE_API SimulationFactory
    {
    public:
        /**
         * @brief Check if the SimulationFactory is using GPU (CUDA)
         */
        virtual bool isGpu() const = 0;

        /**
         * @brief Advance all scenes.
         */
        virtual void advanceAll(float dt) = 0;

        /**
         * @brief Fetch all results after advanceAll().
         */
        virtual void fetchResultsAll() = 0;

        /**
         * @brief Create a `Scene`
         * 
         * @param[in] desc Scene description. See `SceneDesc`.
         */
        virtual Scene *createScene(const SceneDesc &desc) = 0;

        /**
         * @brief Create a `Shape`
         *
         * @param[in] geom A `Geometry` object
         * @param[in] shapeDesc Shape description. See `ShapeDesc`.
         */
        virtual Shape *createShape(Geometry &geom, const ShapeDesc &shapeDesc) = 0;

        // Overloads for creating geometries

        /**
         * @brief Create an analytical Geometry
         * 
         * @param[in] type Geometry type.
         * @param[in] params0 Geometry parameters. See `GeometryData::params0`
         */
        virtual Geometry *createGeometry(GeometryType type, float4 params0) = 0;

        /**
         * @brief Create a triangle mesh Geometry
         * 
         * @param[in] type Geometry type: can be `GeometryType::eTriangleMesh` or
         * `GeometryType::eTriangleMeshSlicer`.
         * @param[in] mesh A TriangleMesh
         * @param[in] sdfDesc SDF description. See SdfDesc
         */
        virtual Geometry *createGeometry(GeometryType type, TriangleMesh &mesh, SdfDesc &sdfDesc) = 0;

        /**
         * @brief Create a connected line Geometry
         * 
         * @param[in] type Geometry type: can only be `GeometryType::eConnectedLineSegments`
         * @param[in] numPoints The number of points on the line segment
         * @param[in] points The points on the line segment
         * @param[in] fattenBounds A distance for fatten the Geometry bounds.
         */
        virtual Geometry *createGeometry(GeometryType type, int numPoints, Vec3f *points, float fattenBounds = 0.5f) = 0;

        /**
         * @brief Create a ParticleObject
         * 
         * @param[in] desc ParticleObject description. See `ParticleObjectDesc`.
         */
        virtual ParticleObject *createParticleObject(const ParticleObjectDesc &desc) = 0;

        /**
         * Mark GPU data dirty. Required after any data on the CPU is modified.
         * 
         * See `SimulationFactoryGpuDataDirtyFlags`.
         */
        virtual void markDirty(SimulationFactoryGpuDataDirtyFlags flags) = 0;

        /**
         * @brief Get the shared CPU particle data block for all scenes
         */
        virtual ParticleData &getParticleDataAll() = 0;

    protected:
        SimulationFactory() {};

        virtual ~SimulationFactory() {};
    };

    /**
     * @brief Create a `SimulationFactory`.
     * 
     * A `SimulationFactory` object is needed for creating all simulation objects.
     * 
     * @param[in] shapeCapacity Maximum number of shapes.
     * @param[in] geometryCapacity Maximum number of geometries.
     * @param[in] sdfDataCapacity Maximum number of SDF data (in terms of float4 structs).
     * @param[in] buildGpuData If the factory uses GPU.
     */
    CRMPM_ENGINE_API SimulationFactory *createFactory(int particleCapacity,
                                                      int shapeCapacity,
                                                      int geometryCapacity,
                                                      int sdfDataCapacity = 10000,
                                                      bool buildGpuData = true,
                                                      int numCudaStreams = 1);

    /**
     * @brief Release the `SimulationFactory`
     */
    CRMPM_ENGINE_API void releaseFactory(SimulationFactory *simulationFactory);

} // namespace crmpm

#endif // !CR_SIMULATION_FACTORY_H