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

#include "vec3.h"
#include "sdf_builder.h"
#include "debug_logger.h"

#include "aabb_tree.h"
#include "obj_loader.h"

#include "simulation_factory.h"

void testSdfGeometry()
{
    ObjLoader loader;
    loader.load("plane.obj"); // Replace with your file path
    CR_DEBUG_LOG_INFO("%s", "load model ok");

    int numPoints = loader.getVertices().size();
    int numTriangles = loader.getTriangles().size() / 3;
    crmpm::Vec3f3 *verts = loader.getVertices().data();
    unsigned int *triangles = loader.getTriangles().data();

    crmpm::TriangleMesh mesh;
    mesh.points = verts;
    mesh.numPoints = numPoints;
    mesh.triangles = triangles;
    mesh.numTriangles = numTriangles;

    // Initialize factory
    crmpm::SimulationFactory *simFactory = crmpm::createFactory(1000, 4, 4, 20000, true);

    CR_DEBUG_LOG_INFO("%s", "geom1");
    crmpm::SdfDesc sdfDesc;
    sdfDesc.cellSize = 0.1;
    sdfDesc.fatten = 0.2;
    sdfDesc.slicerSpineArea = make_float4(-1, 0, 0, 0);
    crmpm::Geometry *geom1 = simFactory->createGeometry(crmpm::GeometryType::eTriangleMeshSlicer, mesh, sdfDesc);
    CR_DEBUG_LOG_INFO("%s", "create triangle mesh geometry ok");

    CR_DEBUG_LOG_INFO("%s", "Query SDF");
    crmpm::Bounds3 bounds = geom1->getBounds();
    CR_DEBUG_LOG_INFO("%f", bounds.minimum.x());
    CR_DEBUG_LOG_INFO("%f", bounds.minimum.y());
    CR_DEBUG_LOG_INFO("%f", bounds.minimum.z());
    CR_DEBUG_LOG_INFO("%f", bounds.maximum.x());
    CR_DEBUG_LOG_INFO("%f", bounds.maximum.y());
    CR_DEBUG_LOG_INFO("%f", bounds.maximum.z());

    CR_DEBUG_LOG_INFO("%s", "Query SDF");
    float4 querySdfResults;
    geom1->queryPointSdf(crmpm::Vec3f(-1.1, 0.0, 0), querySdfResults);
    CR_DEBUG_LOG_INFO("%f", querySdfResults.x);
    CR_DEBUG_LOG_INFO("%f", querySdfResults.y);
    CR_DEBUG_LOG_INFO("%f", querySdfResults.z);
    CR_DEBUG_LOG_INFO("%f", querySdfResults.w);


    CR_DEBUG_LOG_INFO("%s", "geom2");
    crmpm::Geometry *geom2 = simFactory->createGeometry(crmpm::GeometryType::eCapsule, make_float4(1, 2, 3, 4));
    CR_DEBUG_LOG_INFO("%s", "create geometry 2 ok");

    CR_DEBUG_LOG_INFO("%s", "Query SDF");
    bounds = geom2->getBounds();
    CR_DEBUG_LOG_INFO("%f", bounds.minimum.x());
    CR_DEBUG_LOG_INFO("%f", bounds.minimum.y());
    CR_DEBUG_LOG_INFO("%f", bounds.minimum.z());
    CR_DEBUG_LOG_INFO("%f", bounds.maximum.x());
    CR_DEBUG_LOG_INFO("%f", bounds.maximum.y());
    CR_DEBUG_LOG_INFO("%f", bounds.maximum.z());

    CR_DEBUG_LOG_INFO("%s", "Create Shape");
    crmpm::Transform shapeTransform = crmpm::Transform();
    shapeTransform.position = crmpm::Vec3f(0.5f, 0.0f, 0.5f);
    shapeTransform.rotation = crmpm::Quat(0, 0, 0.258819, 0.9659258);
    crmpm::ShapeDesc shapeDesc;
    shapeDesc.type = crmpm::ShapeType::eKinematic;
    shapeDesc.transform = shapeTransform;
    shapeDesc.sdfSmoothDistance = 0.0f;
    shapeDesc.sdfFatten = 0.01f;
    shapeDesc.drag = 1;
    shapeDesc.friction = 0;
    shapeDesc.invScale = crmpm::Vec3f(1, 1, 1);
    crmpm::Shape *shape1 = simFactory->createShape(*geom1, shapeDesc);

    CR_DEBUG_LOG_INFO("%s", "Create Scene");
    crmpm::SceneDesc sceneDesc;
    sceneDesc.solverType = crmpm::MpmSolverType::eGpuPbMpmSolver;
    sceneDesc.numMaxParticles = 1000;
    sceneDesc.gravity = crmpm::Vec3f(0, -9.81f, 0);
    sceneDesc.gridBounds = crmpm::Bounds3(crmpm::Vec3f(0.0f, 0.0f, 0.0f), crmpm::Vec3f(1.5f, 10.5f, 1.5f));
    sceneDesc.gridCellSize = 0.02f;
    sceneDesc.solverIterations = 10;
    crmpm::Scene *scene = simFactory->createScene(sceneDesc);

    scene->addShape(shape1);

    scene->advance(0.002);
    scene->fetchResults();
    scene->advance(0.002);
    scene->fetchResults();
    crmpm::releaseFactory(simFactory);
}

int main()
{
    testSdfGeometry();
    return 0;
}