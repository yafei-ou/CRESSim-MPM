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

#include "crmpm_c_api.h"
#include "simulation_factory.h"

/* Global variables */

bool gIsEngineInitialized = false;
crmpm::SimulationFactory *gSimulationFactory;

/* Data conversion */

/**
 * Convert 3-element Vec3f3 to 4-element crmpm::Vec3f
 */
static crmpm::Vec3f toEngine(const CrVec3f3 &v)
{
    return crmpm::Vec3f(v.x, v.y, v.z);
}

static float4 toEngine(const CrVec4f &v)
{
    return make_float4(v.x, v.y, v.z, v.w);
}

static crmpm::Quat toEngine(const CrQuat &q)
{
    return crmpm::Quat(q.x, q.y, q.z, q.w);
}

static crmpm::Bounds3 toEngine(const CrBounds3 &b)
{
    crmpm::Bounds3 bounds;
    bounds.minimum = toEngine(b.center) - toEngine(b.extents);
    bounds.maximum = toEngine(b.center) + toEngine(b.extents);
    return bounds;
}

/* Engine management */

void CrInitializeEngine(
    unsigned int shapeCapacity,
    unsigned int geometryCapacity,
    unsigned int sdfDataCapacity,
    int buildGpuData)
{
    if (!gIsEngineInitialized)
    {
        gSimulationFactory = crmpm::createFactory(shapeCapacity, geometryCapacity, sdfDataCapacity, buildGpuData);
        gIsEngineInitialized = true;
    }
}

int CrGetInitializationStatus()
{
    return gIsEngineInitialized;
}

void CrReleaseEngine()
{
    crmpm::releaseFactory(gSimulationFactory);
    gIsEngineInitialized = false;
}

/* Physics step */

void CrAdvance(CrSceneHandle scene, float dt)
{
    crmpm::Scene *cppScene = static_cast<crmpm::Scene *>(scene);
    cppScene->advance(dt);
}

void CrFetchResults(CrSceneHandle scene)
{
    crmpm::Scene *cppScene = static_cast<crmpm::Scene *>(scene);
    cppScene->fetchResults();
}

/* Simulation data */

CrParticleData CrGetParticleData(CrSceneHandle scene)
{
    crmpm::Scene *cppScene = static_cast<crmpm::Scene *>(scene);
    crmpm::ParticleData &particleData = cppScene->getParticleData();
    CrParticleData data;
    data.numParticles = cppScene->getNumAllocatedParticles();
    data.positionMass = particleData.positionMass;
    data.velocity = particleData.velocity;
    return data;
}

CrParticleData CrGetParticleDataGpu(CrSceneHandle scene)
{
    crmpm::Scene *cppScene = static_cast<crmpm::Scene *>(scene);
    crmpm::ParticleData &particleData = cppScene->getParticleDataGpu();
    CrParticleData data;
    data.numParticles = cppScene->getNumAllocatedParticles();
    data.positionMass = particleData.positionMass;
    data.velocity = particleData.velocity;
    return data;
}

void CrGetShapeCouplingForceTorque(CrShapeHandle shape, CrVec4f *outForce, CrVec4f *outTorque)
{
    crmpm::Shape *cppShape = static_cast<crmpm::Shape *>(shape);
    memcpy(outForce, &cppShape->getShapeForce(), sizeof(float4));
    memcpy(outTorque, &cppShape->getShapeTorque(), sizeof(float4));
}

void CrSceneMarkDirty(CrSceneHandle scene, CrSceneDataDirtyFlags flags)
{
    crmpm::Scene *cppScene = static_cast<crmpm::Scene *>(scene);
    cppScene->markDirty(static_cast<crmpm::SceneDataDirtyFlags>(flags));
    return ;
}

void CrSyncSceneDataIfNeeded(CrSceneHandle scene)
{
    crmpm::Scene *cppScene = static_cast<crmpm::Scene *>(scene);
    cppScene->syncDataIfNeeded();
}

/* Object creation */

CrSceneHandle CrCreateScene(
    CrMpmSolverType solverType,
    int numMaxParticles,
    CrVec3f3 gravity,
    CrBounds3 gridBounds,
    float gridCellSize,
    float solverIntegrationStepSize,
    int solverIterations)
{
    CR_DEBUG_LOG_INFO("%s", "Create Scene");
    crmpm::SceneDesc sceneDesc;
    sceneDesc.solverType = static_cast<crmpm::MpmSolverType>(solverType);
    sceneDesc.numMaxParticles = numMaxParticles;
    sceneDesc.gravity = toEngine(gravity);
    sceneDesc.gridBounds = toEngine(gridBounds);
    sceneDesc.gridCellSize = gridCellSize;
    sceneDesc.solverIntegrationStepSize = solverIntegrationStepSize;
    sceneDesc.solverIterations = solverIterations;

    return gSimulationFactory->createScene(sceneDesc);
}

CrShapeHandle CrCreateShape(
    CrGeometryHandle geometry,
    CrVec3f3 position,
    CrQuat rotation,
    CrShapeType shapeType,
    float sdfSmoothDistance,
    float sdfFatten,
    float drag,
    float friction,
    CrVec3f3 invScale)
{
    CR_DEBUG_LOG_INFO("%s", "Create Shape");
    crmpm::Transform shapeTransform;
    shapeTransform.position = toEngine(position);
    shapeTransform.rotation = toEngine(rotation);

    crmpm::ShapeDesc shapeDesc;
    shapeDesc.type = static_cast<crmpm::ShapeType>(shapeType);
    shapeDesc.transform = shapeTransform;
    shapeDesc.sdfSmoothDistance = sdfSmoothDistance;
    shapeDesc.sdfFatten = sdfFatten;
    shapeDesc.drag = drag;
    shapeDesc.friction = friction;
    shapeDesc.invScale = toEngine(invScale);

    crmpm::Geometry *cppGeometry = static_cast<crmpm::Geometry *>(geometry);

    return gSimulationFactory->createShape(*cppGeometry, shapeDesc);
}

CrGeometryHandle CrCreateGeometry(
    CrGeometryType geometryType,
    CrVec4f geometryParams)
{
    CR_DEBUG_LOG_INFO("%s", "Create Geometry");
    return gSimulationFactory->createGeometry(static_cast<crmpm::GeometryType>(geometryType), toEngine(geometryParams));
}

CrGeometryHandle CrCreateTrimeshGeometry(
    CrGeometryType geometryType,
    void *points,
    int numPoints,
    unsigned int *triangles,
    int numTriangles,
    float sdfCellSize,
    float sdfFatten,
    CrVec4f sdfSlicerSpineArea)
{
    CR_DEBUG_LOG_INFO("%s", "Create Geometry with triangle mesh");
    crmpm::TriangleMesh mesh;
    mesh.points = static_cast<crmpm::Vec3f3 *>(points);
    mesh.numPoints = numPoints;
    mesh.triangles = triangles;
    mesh.numTriangles = numTriangles;
    crmpm::SdfDesc sdfDesc;
    sdfDesc.cellSize = sdfCellSize;
    sdfDesc.fatten = sdfFatten;
    sdfDesc.slicerSpineArea = toEngine(sdfSlicerSpineArea);
    crmpm::Geometry *geom = gSimulationFactory->createGeometry(static_cast<crmpm::GeometryType>(geometryType), mesh, sdfDesc);
    return geom;
}

CrGeometryHandle CrCreateConnectedLineSegmentsGeometry(
    int numPoints,
    CrVec3f3 *points,
    float fattenBounds)
{
    crmpm::Vec3f *convertedPoints = new crmpm::Vec3f[numPoints];
    for (int i = 0; i < numPoints; ++i)
    {
        convertedPoints[i] = toEngine(points[i]);
    }
    crmpm::Geometry *geom = gSimulationFactory->createGeometry(crmpm::GeometryType::eConnectedLineSegments, numPoints, convertedPoints, fattenBounds);
    delete[] convertedPoints;
    return geom;
}

CrParticleObjectHandle CrCreateParticleObject(
    float particleSpacing,
    float particleMass,
    CrGeometryHandle geometry,
    const CrVec3f3 *position,
    const CrQuat *rotation,
    const CrVec3f3 *invScale,
    CrParticleMaterialType materialType,
    CrVec4f *materialParams,
    unsigned int *outNumParticles)
{
    CR_DEBUG_LOG_INFO("%s", "Create ParticleObject");
    crmpm::ParticleObjectDesc poDesc;
    poDesc.particleSpacing = particleSpacing;
    poDesc.particleMass = particleMass;
    poDesc.geometry = static_cast<crmpm::Geometry *>(geometry);
    poDesc.position = toEngine(*position);
    poDesc.rotation = toEngine(*rotation);
    poDesc.invScale = toEngine(*invScale);
    poDesc.materialType = static_cast<crmpm::ParticleMaterialType>(materialType);
    poDesc.materialParams = toEngine(*materialParams);

    crmpm::ParticleObject *po = gSimulationFactory->createParticleObject(poDesc);
    *outNumParticles = po->getNumParticles();
    return po;
}

/* Simulation object management */

void CrAddParticleObjectToScene(
    CrParticleObjectHandle particleObject,
    CrSceneHandle scene,
    unsigned int *outOffset)
{
    crmpm::ParticleObject *cppParticleObject = static_cast<crmpm::ParticleObject *>(particleObject);
    crmpm::Scene *cppScene = static_cast<crmpm::Scene *>(scene);
    cppParticleObject->addToScene(*cppScene);
    *outOffset = cppParticleObject->getIndexOffset();
}

void CrResetParticleObject(
    CrParticleObjectHandle particleObject)
{
    crmpm::ParticleObject *cppParticleObject = static_cast<crmpm::ParticleObject *>(particleObject);
    cppParticleObject->reset();
}

void CrAddShapeToScene(
    CrShapeHandle shape,
    CrSceneHandle scene)
{
    crmpm::Shape *cppShape = static_cast<crmpm::Shape *>(shape);
    crmpm::Scene *cppScene = static_cast<crmpm::Scene *>(scene);
    cppScene->addShape(cppShape);
}

void CrSetShapeKinematicTarget(
    CrShapeHandle shape,
    CrVec3f3 *targetPosition,
    CrQuat *targetRotation,
    float dt)
{
    crmpm::Shape *cppShape = static_cast<crmpm::Shape *>(shape);
    crmpm::Transform transform;
    transform.position = toEngine(*targetPosition);
    transform.rotation = toEngine(*targetRotation);
    cppShape->setKinematicTarget(transform, dt);
}

/* Object release */

void CrReleaseScene(CrSceneHandle scene)
{
    crmpm::Scene *cppScene = static_cast<crmpm::Scene *>(scene);
    cppScene->release();
}

void CrReleaseShape(CrShapeHandle shape)
{
    crmpm::Shape *cppShape = static_cast<crmpm::Shape *>(shape);
    cppShape->release();
}

void CrReleaseGeometry(CrGeometryHandle geometry)
{
    crmpm::Geometry *cppGeometry = static_cast<crmpm::Geometry *>(geometry);
    cppGeometry->release();
}

void CrReleaseParticleObject(CrParticleObjectHandle particleObject)
{
    crmpm::ParticleObject *cppParticleObject = static_cast<crmpm::ParticleObject *>(particleObject);
    cppParticleObject->release();
}

/* Fast copy */

void CrFastCopy(void *src, void *dst, int size)
{
    memcpy(dst, src, size);
}
