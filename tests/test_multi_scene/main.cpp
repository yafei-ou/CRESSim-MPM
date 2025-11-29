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

#include "array.h"
#include "debug_logger.h"
#include "particle_object.h"
#include "simulation_factory.h"
#include "transform.h"
#include "obj_loader.h"
#include "visualizer.h"


crmpm::Scene *createTestScene(crmpm::SimulationFactory &simFactory)
{
    // Initialize scene
    CR_DEBUG_LOG_INFO("%s", "Create Scene");

    crmpm::SceneDesc sceneDesc;
    sceneDesc.solverType = crmpm::MpmSolverType::eGpuMlsMpmSolver;
    sceneDesc.numMaxParticles = 4000;
    sceneDesc.gravity = crmpm::Vec3f(0, -9.81f, 0);
    sceneDesc.gridBounds = crmpm::Bounds3(crmpm::Vec3f(0.0f, 0.0f, 0.0f), crmpm::Vec3f(10.5f, 100.5f, 10.5f));
    sceneDesc.gridCellSize = 0.2f;
    sceneDesc.solverIntegrationStepSize = 0.002f;
    sceneDesc.solverIterations = 20;

    crmpm::Scene *scene = simFactory.createScene(sceneDesc);

    // Except for PB-MPM, compute Lame parameters
    constexpr float E = 5e5f;  // Young's modulus
    constexpr float nu = 0.4f; // Poisson's ratio
    constexpr float lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    constexpr float mu = E / (2 * (1 + nu));

    CR_DEBUG_LOG_INFO("%s", "Create Geometry");
    crmpm::Geometry *geom1 = simFactory.createGeometry(crmpm::GeometryType::eBox, make_float4(0.5, 0.5, 0.5, 0));

    CR_DEBUG_LOG_INFO("%s", "Create particle object");
    crmpm::ParticleObjectDesc poDesc;
    poDesc.particleSpacing = 0.1f;
    poDesc.particleMass = 1.0f;
    poDesc.geometry = geom1;
    poDesc.position = crmpm::Vec3f(0.8, 1.6, 0.8);
    poDesc.rotation = crmpm::Quat();
    poDesc.invScale = crmpm::Vec3f(1, 1, 1);
    poDesc.materialType = crmpm::ParticleMaterialType::eCoRotational;
    poDesc.materialParams = make_float4(lambda, mu, 0, 0);
    // poDesc.materialParams = make_float4(1000.0f, 0.8f, 0, 0);
    crmpm::ParticleObject *po1 = simFactory.createParticleObject(poDesc);
    if(po1->addToScene(*scene))
    {
        CR_DEBUG_LOG_INFO("%s", "Create particle object ok");
    }
    return scene;
}

void testMultiScene()
{
    // Initialize factory
    crmpm::SimulationFactory *simFactory = crmpm::createFactory(1000000, 4, 4, 15000, true);
    crmpm::Array<crmpm::Scene*> scenes(5);
    for (int i = 0; i < 10; ++i)
    {
        scenes.pushBack(createTestScene(*simFactory));
    }

    float timeElapsed = 0;
    bool shouldChange = true;

    // Main loop.
    for (int i = 0; i < 10; ++i)
    {
        CR_DEBUG_LOG_EXECUTION_TIME(
            for (auto scene : scenes) {
                scene->advance(0.02);
            }
            for (auto scene : scenes) {
                scene->fetchResults();
            }
            );
    }
    crmpm::releaseFactory(simFactory);
}

void testMultiSceneAdvanceAll()
{
    // Initialize factory
    crmpm::SimulationFactory *simFactory = crmpm::createFactory(1000000, 4, 4, 15000, true, 2);
    crmpm::Array<crmpm::Scene*> scenes(5);
    for (int i = 0; i < 20; ++i)
    {
        scenes.pushBack(createTestScene(*simFactory));
    }

    bool shouldChange = true;

    for (int i = 0; i < 10; ++i)
    {
        CR_DEBUG_LOG_EXECUTION_TIME(
            simFactory->advanceAll(0.02);
            simFactory->fetchResultsAll(););
    }

    crmpm::releaseFactory(simFactory);
}

void testMultiSceneAdvanceAllMultiStream()
{
    // Initialize factory
    crmpm::SimulationFactory *simFactory = crmpm::createFactory(1000000, 4, 4, 15000, true, 2);
    crmpm::Array<crmpm::Scene*> scenes(5);
    for (int i = 0; i < 20; ++i)
    {
        scenes.pushBack(createTestScene(*simFactory));
    }

    bool shouldChange = true;

    for (int i = 0; i < 10; ++i)
    {
        CR_DEBUG_LOG_EXECUTION_TIME(
            simFactory->advanceAll(0.02);
            simFactory->fetchResultsAll(););
    }

    crmpm::releaseFactory(simFactory);
}


int main()
{
    // testMultiScene();
    testMultiSceneAdvanceAll();
    testMultiSceneAdvanceAllMultiStream();
    return 0;
}
