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

int main(int argc, char* argv[])
{
    std::string objFile;
    if (argc > 1)
    {
        objFile = argv[1];
    }
    else
    {
        std::cerr << "Usage: example_object_creation <obj_file>\n";
        return 1;
    }

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
    constexpr float E = 5e5f;  // Young's modulus
    constexpr float nu = 0.4f; // Poisson's ratio
    constexpr float lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    constexpr float mu = E / (2 * (1 + nu));

    CR_DEBUG_LOG_INFO("%s", "Create particle object");

    // Geometry for initializing particles
    crmpm::Geometry *geom1 = simFactory->createGeometry(crmpm::GeometryType::eSphere, make_float4(1, 0.0, 0, 0));

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
    if (po1->addToScene(*scene))
    {
        CR_DEBUG_LOG_INFO("%s", "Create particle object ok");
    }
    else
    {
        CR_DEBUG_LOG_INFO("%s", "Create particle object failed");
    }

    // Load triangle mesh
    ObjLoader loader;
    try
    {
        loader.load(objFile);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
    
    CR_DEBUG_LOG_INFO("%s", "load model ok");

    int numPoints = loader.getVertices().size();
    int numTriangles = loader.getTriangles().size() / 3;
    crmpm::Vec3f3 *verts = loader.getVertices().data();
    unsigned int *triangles = loader.getTriangles().data();

    // Flat triangle mesh data for rendering

    std::vector<float> flatVertices;
    std::vector<unsigned int> flatIndices;

    flatIndices.resize(numTriangles * 3);
    std::memcpy(flatIndices.data(), triangles, numTriangles * 3 * sizeof(unsigned int));

    flatVertices.resize(numPoints * 3);
    std::memcpy(flatVertices.data(), verts, numPoints * 3 * sizeof(float));

    // Create TriangleMesh geometry and shape

    CR_DEBUG_LOG_INFO("%s", "create triangle mesh geometry");
    crmpm::TriangleMesh mesh;
    mesh.points = verts;
    mesh.numPoints = numPoints;
    mesh.triangles = triangles;
    mesh.numTriangles = numTriangles;
    crmpm::SdfDesc sdfDesc;
    sdfDesc.cellSize = 0.05;
    sdfDesc.fatten = 0.2;
    crmpm::Geometry *geom2 = simFactory->createGeometry(crmpm::GeometryType::eTriangleMesh, mesh, sdfDesc);
    CR_DEBUG_LOG_INFO("%s", "create triangle mesh geometry ok");

    CR_DEBUG_LOG_INFO("%s", "Create Shape");
    crmpm::Transform shapeTransform = crmpm::Transform();
    shapeTransform.position = crmpm::Vec3f(0.0f, 0.5f, 0.0f);

    crmpm::ShapeDesc shapeDesc;
    shapeDesc.type = crmpm::ShapeType::eKinematic;
    shapeDesc.transform = shapeTransform;
    shapeDesc.sdfSmoothDistance = 0.0f;
    shapeDesc.sdfFatten = 0.1f;
    shapeDesc.drag = 1;
    shapeDesc.friction = 0;
    shapeDesc.invScale = crmpm::Vec3f(2, 2, 2);

    crmpm::Shape *shape1 = simFactory->createShape(*geom2, shapeDesc);
    scene->addShape(shape1);

    scene->advance(0.02); // Initial advance

    float timeElapsed = 0;
    bool shouldChange = true;

    // Main loop
    while (vis.running())
    {
        scene->fetchResults();

        timeElapsed += 0.02f;

        if (timeElapsed > 2 && shouldChange)
        {
            po1->reset(); // Reset particle object
            shouldChange = false;
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
        Eigen::Quaternionf q(shapeTransform.rotation.data.w, shapeTransform.rotation.data.x, shapeTransform.rotation.data.y, shapeTransform.rotation.data.z);
        vis.drawMeshWireframe(flatVertices,
                              flatIndices,
                              Eigen::Vector3f(shapeTransform.position.x(), shapeTransform.position.y(), shapeTransform.position.z()),
                              q.toRotationMatrix(),
                              Eigen::Vector3f(1 / shapeDesc.invScale.x(), 1 / shapeDesc.invScale.y(), 1 / shapeDesc.invScale.z()));
        vis.endRender();

        scene->advance(0.02);
    }

    geom1->release();
    geom2->release();
    po1->release();
    shape1->release();
    scene->release();
    crmpm::releaseFactory(simFactory);

    return 0;
}
