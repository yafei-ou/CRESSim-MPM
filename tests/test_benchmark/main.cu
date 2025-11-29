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
#include "visualizer.h"

#ifdef _MSC_VER
#include <device_launch_parameters.h>
#endif // _MSC_VER

__global__ void warmupKernel() {
    // A dummy kernel that does nothing meaningful
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        // Simple statement to ensure some work happens
    }
}

void warmupCUDA() {
    // Launch the warmup kernel with a minimal configuration
    warmupKernel << <1, 1 >> > ();

    // Synchronize to ensure the kernel completes
    cudaDeviceSynchronize();

    // Check for any errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Warmup kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    else {
        std::cout << "CUDA warmup kernel completed successfully." << std::endl;
    }
}


// #define BENCHMARK_VISUAL
#define BENCHMARK_RUN

// #define BENCHMARK_SOLVER crmpm::MpmSolverType::eGpuMlsMpmSolver
// #define BENCHMARK_SOLVER_INTEGRATION_STEP 0.002f
// #define BENCHMARK_PARTICLE_PROPERTIES make_float4(lambda, mu, 0, 0)

// #define BENCHMARK_SOLVER crmpm::MpmSolverType::eGpuPbMpmSolver
// #define BENCHMARK_SOLVER_INTEGRATION_STEP 0.02f
// #define BENCHMARK_SOLVER_ITERATIONS 20
// #define BENCHMARK_PARTICLE_PROPERTIES make_float4(100, 0.9f, 0, 0)

#define BENCHMARK_SOLVER crmpm::MpmSolverType::eGpuStandardMpmSolver
#define BENCHMARK_SOLVER_INTEGRATION_STEP 0.002f
#define BENCHMARK_SOLVER_ITERATIONS 20
#define BENCHMARK_PARTICLE_PROPERTIES make_float4(lambda, mu, 0, 0)

void testCubeParticleObject(float invSize)
{
    // Initialize factory
    crmpm::SimulationFactory *simFactory = crmpm::createFactory(1000000, 4, 4, 15000, true);

    // Initialize scene
    crmpm::SceneDesc sceneDesc;
    sceneDesc.solverType = BENCHMARK_SOLVER;
    sceneDesc.numMaxParticles = 1000000;
    sceneDesc.gravity = crmpm::Vec3f(0, -1, 0);
    sceneDesc.gridBounds = crmpm::Bounds3(crmpm::Vec3f(-5.5, -5.5, -5.5), crmpm::Vec3f(5.5, 5.5, 5.5));
    sceneDesc.gridCellSize = 0.2f;
    sceneDesc.solverIntegrationStepSize = BENCHMARK_SOLVER_INTEGRATION_STEP;
    sceneDesc.solverIterations = BENCHMARK_SOLVER_ITERATIONS;

    // Grid number
    crmpm::Bounds3 gridBound = sceneDesc.gridBounds;
    float cellSize = sceneDesc.gridCellSize;
    crmpm::Vec3f gridSize = gridBound.maximum - gridBound.minimum;
    crmpm::Vec3i numNodesPerDim = (gridSize / cellSize + 1.0f).cast<int>();
    int numNodes = numNodesPerDim.x() * numNodesPerDim.y() * numNodesPerDim.z();
    CR_DEBUG_LOG_INFO("Num grids: %d", numNodes);

    crmpm::Scene *scene = simFactory->createScene(sceneDesc);

    // Except for PB-MPM, compute Lame parameters
    constexpr float E = 5e5f;  // Young's modulus
    constexpr float nu = 0.4f; // Poisson's ratio
    constexpr float lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    constexpr float mu = E / (2 * (1 + nu));

    crmpm::Geometry *geom1 = simFactory->createGeometry(crmpm::GeometryType::eBox, make_float4(0.5, 0.5, 0.5, 0));

    crmpm::ParticleObjectDesc poDesc;
    poDesc.particleSpacing = 0.1f;
    poDesc.particleMass = 10;
    poDesc.geometry = geom1;
    poDesc.position = crmpm::Vec3f();
    poDesc.rotation = crmpm::Quat();
    poDesc.invScale = crmpm::Vec3f(invSize, invSize, invSize);
    poDesc.materialType = crmpm::ParticleMaterialType::eCoRotational;
    poDesc.materialParams = BENCHMARK_PARTICLE_PROPERTIES;
    crmpm::ParticleObject *po1 = simFactory->createParticleObject(poDesc);
    po1->addToScene(*scene);

    int numParticles = scene->getNumAllocatedParticles();
    CR_DEBUG_LOG_INFO("Num particles: %d", numParticles);


#ifdef BENCHMARK_VISUAL
    // Visualizer.
    Visualizer vis;

    // Main loop.
    while (vis.running())
    {
        scene->fetchResults();

        // Get particle data
        const crmpm::ParticleData &particles = scene->getParticleData();
        std::vector<float> positions;
        positions.reserve(particles.mSize * 3);

        for (int i = 0; i < particles.mSize; i++)
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

#endif

#ifdef BENCHMARK_RUN

    CR_DEBUG_LOG_EXECUTION_TIME(
        for (int i = 0; i < 100; ++i) {
            scene->advance(0.02);
            scene->fetchResults();
        });
#endif

    geom1->release();
    po1->release();
    scene->release();
    crmpm::releaseFactory(simFactory);
}

int main()
{
    warmupCUDA();
    // testCubeParticleObject(1);
    testCubeParticleObject(0.1);
    testCubeParticleObject(0.11);
    testCubeParticleObject(0.12);
    testCubeParticleObject(0.13);
    testCubeParticleObject(0.15);
    testCubeParticleObject(0.2);
    testCubeParticleObject(1);
    return 0;
}
