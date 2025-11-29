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


void testConnectedLine()
{
    // Visualizer.
    Visualizer vis;

    // Initialize factory
    crmpm::SimulationFactory *simFactory = crmpm::createFactory(4000, 4, 4, 15000, true);

    // Initialize scene
    CR_DEBUG_LOG_INFO("%s", "Create Scene");

    crmpm::SceneDesc sceneDesc;
    sceneDesc.solverType = crmpm::MpmSolverType::eCpuMlsMpmSolver;
    sceneDesc.numMaxParticles = 4000;
    sceneDesc.gravity = crmpm::Vec3f(0, -9.81f, 0);
    sceneDesc.gridBounds = crmpm::Bounds3(crmpm::Vec3f(0.0f, -0.2f, 0.0f), crmpm::Vec3f(10.5f, 10.5f, 10.5f));
    sceneDesc.gridCellSize = 0.2f;
    sceneDesc.solverIntegrationStepSize = 0.002f;
    sceneDesc.solverIterations = 20;

    crmpm::Scene *scene = simFactory->createScene(sceneDesc);

    // Except for PB-MPM, compute Lame parameters
    constexpr float E = 10e5f;  // Young's modulus
    constexpr float nu = 0.4f; // Poisson's ratio
    constexpr float lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    constexpr float mu = E / (2 * (1 + nu));

    CR_DEBUG_LOG_INFO("%s", "Create Box Geometry");
    crmpm::Geometry *geom1 = simFactory->createGeometry(crmpm::GeometryType::eBox, make_float4(0.5, 0.5, 0.5, 0));

    CR_DEBUG_LOG_INFO("%s", "Create Line Geometry");
    crmpm::Vec3f *points = new crmpm::Vec3f[2];
    points[0] = crmpm::Vec3f(-0.3, -0.8, 0);
    points[1] = crmpm::Vec3f(0, 0, 0);
    // points[2] = crmpm::Vec3f(1, 1, 0);
    crmpm::Geometry *geom2 = simFactory->createGeometry(crmpm::GeometryType::eConnectedLineSegments, 2, points);

    CR_DEBUG_LOG_INFO("%s", "Create particle object");
    crmpm::ParticleObjectDesc poDesc;
    poDesc.particleSpacing = 0.1f;
    poDesc.particleMass = 1.0f;
    poDesc.geometry = geom1;
    poDesc.position = crmpm::Vec3f(0.8, 2.0, 0.8);
    poDesc.rotation = crmpm::Quat();
    poDesc.invScale = crmpm::Vec3f(1, 1, 1);
    poDesc.materialType = crmpm::ParticleMaterialType::eCoRotational;
    poDesc.materialParams = make_float4(lambda, mu, 0, 0);
    // poDesc.materialParams = make_float4(1000.0f, 0.8f, 0, 0);
    crmpm::ParticleObject *po1 = simFactory->createParticleObject(poDesc);
    if(po1->addToScene(*scene))
    {
        CR_DEBUG_LOG_INFO("%s", "Create particle object ok");
    }

    CR_DEBUG_LOG_INFO("%s", "Create Shape");
    crmpm::Transform shapeTransform = crmpm::Transform();
    shapeTransform.position = crmpm::Vec3f(0.8, 1, 0.8);
    // shapeTransform.rotation = crmpm::Quat();
    // shapeTransform.rotation = crmpm::Quat(0, 0, 0.258819, 0.9659258);
    // shapeTransform.rotation = crmpm::Quat(0, 0, 0.7071068, 0.7071068);
    // shapeTransform.rotation = crmpm::Quat(0.5, -0.5, 0.5, 0.5);
    crmpm::ShapeDesc shapeDesc;
    shapeDesc.type = crmpm::ShapeType::eKinematic;
    shapeDesc.transform = shapeTransform;
    shapeDesc.sdfSmoothDistance = 0.0f;
    shapeDesc.sdfFatten = 0.2f;
    shapeDesc.drag = 1;
    shapeDesc.friction = 0.5;
    shapeDesc.invScale = crmpm::Vec3f(1, 1, 1);

    crmpm::Shape *shape1 = simFactory->createShape(*geom2, shapeDesc);

    CR_DEBUG_LOG_INFO("%s", "Initial Advance");
    scene->advance(0.002);

    float timeElapsed = 0;
    bool shouldChange = true;

    // Main loop.
    while (vis.running())
    {
        scene->fetchResults();

        timeElapsed += 0.02f;

        if (timeElapsed > 2 && shouldChange)
        {
            scene->addShape(shape1);
            shouldChange = false;
        }

        if (timeElapsed > 3)
        {
            shapeTransform.position += crmpm::Vec3f(0, -0.01, 0);
            // shapeTransform.rotation *= crmpm::Quat(0, 0, -0.0008727, 0.9999996);
            // shape1->setKinematicTarget(shapeTransform, 0.02f);
            // shapeTransform.position += crmpm::Vec3f(0.001f, 0, 0);
            // shapeTransform.rotation *= crmpm::Quat(0, 0, -0.0008727, 0.9999996);
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

    geom1->release();
    po1->release();
    shape1->release();
    scene->release();
    crmpm::releaseFactory(simFactory);
}

int main()
{
    testConnectedLine();
    return 0;
}
