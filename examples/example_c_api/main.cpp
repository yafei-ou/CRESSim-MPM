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
#include "visualizer.h"
#include "crmpm_c_api.h"

#if defined(__CUDACC__) // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define ALIGN(n) __declspec(align(n))
#else
#error "No align for compiler!"
#endif

struct ALIGN(16) float4
{
    float x, y, z, w;
};

namespace crmpm
{
    enum class MpmSolverType
    {
        eCpuMlsMpmSolver,
        eGpuMlsMpmSolver,

        eCpuPbMpmSolver,
        eGpuPbMpmSolver
    };

    enum class GeometryType : int
    {
        // Regular geometries

        eBox,
        eSphere,
        ePlane,
        eCapsule,
        eTriangleMesh,

        // Open boundary slicer geometries

        eQuadSlicer,
        eTriangleMeshSlicer
    };

    enum class ParticleMaterialType : unsigned int
    {
        eCoRotational,
        eNeoHookean,
        eLinearElastic,
    };

} // namespace crmpm

int main()
{
    // Visualizer.
    Visualizer vis;

    CrInitializeEngine(10000, 4, 4, 0, true, 1);

    // Initialize scene
    auto solverType = static_cast<int>(crmpm::MpmSolverType::eCpuMlsMpmSolver);
    auto numMaxParticles = 4000;
    auto gravity = CrVec3f3();
    gravity.x = 0;
    gravity.y = -9.81f;
    gravity.z = 0;
    auto gridBounds = CrBounds3();
    gridBounds.center.x = 0;
    gridBounds.center.y = 5;
    gridBounds.center.z = 0;
    gridBounds.extents.x = 5.5f;
    gridBounds.extents.y = 5.5f;
    gridBounds.extents.z = 5.5f;

    auto gridCellSize = 0.2f;
    auto solverIntegrationStepSize = 0.002f;
    auto solverIterations = 20;

    CrSceneHandle scene = CrCreateScene(solverType, 4000, gravity, gridBounds, gridCellSize, solverIntegrationStepSize, solverIterations, CrShapeContactModel::CR_SHAPE_CONTACT_MODEL_KINEMATIC);

    // Except for PB-MPM, compute Lame parameters
    constexpr float E = 5e5f;  // Young's modulus
    constexpr float nu = 0.4f; // Poisson's ratio
    constexpr float lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    constexpr float mu = E / (2 * (1 + nu));

    CrVec4f geomParams = {0.5, 0.5, 0.5, 0};
    CrGeometryHandle geom1 = CrCreateGeometry(static_cast<int>(crmpm::GeometryType::eBox), geomParams);

    auto particleSpacing = 0.05f;
    auto particleMass = 1.0f;
    auto geometry = geom1;
    auto position = CrVec3f3();
    position.x = 0.0;
    position.y = 1.6;
    position.z = 0.0;
    auto rotation = CrQuat();
    rotation.x = 0;
    rotation.y = 0;
    rotation.z = 0;
    rotation.w = 1;
    auto invScale = CrVec3f3();
    invScale.x = 2;
    invScale.y = 2;
    invScale.z = 2;
    auto materialType = static_cast<int>(crmpm::ParticleMaterialType::eCoRotational);
    auto materialParams = CrVec4f();
    materialParams = {lambda, mu, 0, 0};
    unsigned int offset;
    unsigned int numParticles;
    CrParticleObjectHandle po1 = CrCreateParticleObject(particleSpacing, particleMass, geom1, &position, &rotation, &invScale, materialType, &materialParams, &numParticles);
    CrAddParticleObjectToScene(po1, scene, &offset);

    CrAdvance(scene, 0.02f);

    // Main loop.
    while (vis.running())
    {
        CrFetchResults(scene);

        // Get particle data
        const CrParticleData &particles = CrGetParticleData(scene);
        std::vector<float> positions;
        positions.reserve(particles.numParticles * 3);

        for (int i = 0; i < particles.numParticles; i++)
        {
            const auto position = static_cast<float4 *>(particles.positionMass)[i];
            positions.push_back(position.x);
            positions.push_back(position.y);
            positions.push_back(position.z);
        }

        vis.beginRender();
        vis.drawParticles(positions);
        vis.endRender();

        CrAdvance(scene, 0.02f);
    }

    CrFetchResults(scene);

    CrReleaseGeometry(geom1);
    CrReleaseScene(scene);
    CrReleaseParticleObject(po1);
    CrReleaseEngine();
    return 0;
}
