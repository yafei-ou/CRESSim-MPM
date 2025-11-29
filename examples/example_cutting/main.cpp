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

#include <iostream>
#include <vector>
#include "debug_logger.h"
#include "particle_object.h"
#include "simulation_factory.h"
#include "transform.h"
#include "obj_loader.h"
#include "visualizer.h"

int main()
{
    // Visualizer
    Visualizer vis;

    // Create factory
    crmpm::SimulationFactory *simFactory = crmpm::createFactory(10000, 4, 4, 15000, true);

    // Create Scene
    CR_DEBUG_LOG_INFO("%s", "Create Scene");

    crmpm::SceneDesc sceneDesc;
    sceneDesc.solverType = crmpm::MpmSolverType::eGpuMlsMpmSolver;
    sceneDesc.numMaxParticles = 10000;
    sceneDesc.gravity = crmpm::Vec3f(0, -9.81f, 0);
    sceneDesc.gridBounds = crmpm::Bounds3(crmpm::Vec3f(-5.0f, -0.5f, -5.0f), crmpm::Vec3f(5.0f, 5.0f, 5.0f));
    sceneDesc.gridCellSize = 0.2f;
    sceneDesc.solverIntegrationStepSize = 0.002f;
    sceneDesc.solverIterations = 20;

    crmpm::Scene *scene = simFactory->createScene(sceneDesc);

    CR_DEBUG_LOG_INFO("%s", "Create Scene ok");

    // Except for PB-MPM, compute Lame parameters
    constexpr float E = 1e5f;  // Young's modulus
    constexpr float nu = 0.4f; // Poisson's ratio
    constexpr float lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    constexpr float mu = E / (2 * (1 + nu));

    CR_DEBUG_LOG_INFO("%s", "Create particle object");

    // Geometry for initializing particles
    crmpm::Geometry *geom1 = simFactory->createGeometry(crmpm::GeometryType::eBox, make_float4(1, 1, 1, 0));

    crmpm::ParticleObjectDesc poDesc;
    poDesc.particleSpacing = 0.1f;
    poDesc.particleMass = 1.0f;
    poDesc.geometry = geom1;
    poDesc.position = crmpm::Vec3f(0.0, 2.0, 0.0);
    poDesc.rotation = crmpm::Quat();
    poDesc.invScale = crmpm::Vec3f(2, 2, 2);
    poDesc.materialType = crmpm::ParticleMaterialType::eCoRotational;
    poDesc.materialParams = make_float4(lambda, mu, 0, 0);
    crmpm::ParticleObject *po1 = simFactory->createParticleObject(poDesc);
    geom1->release(); // This reduces the ref to geom1 by 1
    if (po1->addToScene(*scene))
    {
        CR_DEBUG_LOG_INFO("%s", "Create particle object ok");
    }
    else
    {
        CR_DEBUG_LOG_INFO("%s", "Create particle object failed");
    }

    // Slicer geometry and shape
    CR_DEBUG_LOG_INFO("%s", "Create slicer Geometry");
    crmpm::Geometry *geom2 = simFactory->createGeometry(crmpm::GeometryType::eQuadSlicer, make_float4(1, 1, 0, 0));
    CR_DEBUG_LOG_INFO("%s", "Create slicer Geometry ok");

    CR_DEBUG_LOG_INFO("%s", "Create slicer Shape");
    crmpm::Transform shapeTransform = crmpm::Transform();
    shapeTransform.position = crmpm::Vec3f(0.0f, 2.0f, 0.0f);
    shapeTransform.rotation = crmpm::Quat(0, 0, 0.7071068, 0.7071068);
    crmpm::ShapeDesc shapeDesc;
    shapeDesc.type = crmpm::ShapeType::eKinematic;
    shapeDesc.transform = shapeTransform;
    shapeDesc.sdfSmoothDistance = 0.0f;
    shapeDesc.sdfFatten = 0.1f;
    shapeDesc.drag = 0.5;
    shapeDesc.friction = 0;
    shapeDesc.invScale = crmpm::Vec3f(1, 1, 1);
    crmpm::Shape *shape1 = simFactory->createShape(*geom2, shapeDesc);
    geom2->release(); // This reduces ref to geom2 by 1
    CR_DEBUG_LOG_INFO("%s", "Create slicer Shape ok");

    scene->advance(0.02); // Initial advance

    float timeElapsed = 0;
    bool shouldAddShape = true;

    // Main loop
    while (vis.running())
    {
        scene->fetchResults();

        timeElapsed += 0.02f;

        if (timeElapsed > 2)
        {
            if (shouldAddShape)
            {
                scene->addShape(shape1);
                shouldAddShape = false;
            }
            shapeTransform.position -= crmpm::Vec3f(0, 0.01f, 0);
            shape1->setKinematicTarget(shapeTransform, 0.02f);
        }

        // Get particle data
        const crmpm::ParticleData &particles = scene->getParticleData();
        std::vector<float> positions;
        positions.reserve(particles.size * 3);

        for (int i = 0; i < particles.size; i++)
        {
            const auto position = particles.positionMass[i];
            positions.push_back(position.x);
            positions.push_back(position.y);
            positions.push_back(position.z);
        }

        vis.beginRender();
        vis.drawParticles(positions);
        vis.endRender();

        scene->advance(0.02);
    }

    shape1->release();
    po1->release();
    scene->release();
    crmpm::releaseFactory(simFactory);

    return 0;
}
