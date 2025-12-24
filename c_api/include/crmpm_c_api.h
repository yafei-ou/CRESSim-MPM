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

#ifndef CRMPM_C_API_H
#define CRMPM_C_API_H

#ifdef _WIN32
#ifdef CRMPM_C_API_EXPORTS
#define CRMPM_C_API __declspec(dllexport)
#else
#define CRMPM_C_API __declspec(dllimport)
#endif
#elif defined(__linux__)
#ifdef CRMPM_C_API_EXPORTS
#define CRMPM_C_API __attribute__((visibility("default")))
#else
#define CRMPM_C_API
#endif
#else
#define CRMPM_C_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /* Handles */

    typedef void *CrSceneHandle;
    typedef void *CrShapeHandle;
    typedef void *CrGeometryHandle;
    typedef void *CrParticleObjectHandle;

    typedef int CrMpmSolverType;
    typedef int CrShapeType;
    typedef int CrGeometryType;
    typedef int CrParticleMaterialType;

    /* Structs */

    typedef struct CrVec3f3
    {
        float x, y, z;
    } CrVec3f3;

    typedef struct CrVec4f
    {
        float x, y, z, w;
    } CrVec4f;

    typedef struct CrQuat
    {
        float x, y, z, w;
    } CrQuat;

    typedef struct CrBounds3
    {
        CrVec3f3 center;
        CrVec3f3 extents;
    } CrBounds3;

    typedef struct CrParticleData
    {
        unsigned int numParticles;
        void *positionMass; // float4 (16-byte aligned)
        void *velocity;     // float4 (16-byte aligned)
    } CrParticleData;

    /* Enums */

    typedef enum CrSceneDataDirtyFlags
    {
        CR_SCENE_DATA_DIRTY_NONE                 = 0,
        CR_SCENE_DATA_DIRTY_PARTICLE_POS_MASS    = 1 << 0,
        CR_SCENE_DATA_DIRTY_PARTICLE_VELOCITY    = 1 << 1,
        CR_SCENE_DATA_DIRTY_PARTICLE_MATERIAL0   = 1 << 2,
        CR_SCENE_DATA_DIRTY_PARTICLE_TYPE        = 1 << 3,
        CR_SCENE_DATA_DIRTY_NUM_PARTICLES        = 1 << 4,
        CR_SCENE_DATA_DIRTY_COMPUTE_INITIAL_DATA = 1 << 5,

        CR_SCENE_DATA_DIRTY_ALL =
            CR_SCENE_DATA_DIRTY_PARTICLE_POS_MASS |
            CR_SCENE_DATA_DIRTY_PARTICLE_VELOCITY |
            CR_SCENE_DATA_DIRTY_PARTICLE_MATERIAL0 |
            CR_SCENE_DATA_DIRTY_PARTICLE_TYPE |
            CR_SCENE_DATA_DIRTY_NUM_PARTICLES |
            CR_SCENE_DATA_DIRTY_COMPUTE_INITIAL_DATA
    } CrSceneDataDirtyFlags;

    typedef enum CrShapeContactModel
    {
        CR_SHAPE_CONTACT_MODEL_KINEMATIC,
        CR_SHAPE_CONTACT_MODEL_EFFECTIVE_MASS,
    } CrShapeContactModel;

    /* Engine management */

    CRMPM_C_API void CrInitializeEngine(
        unsigned int particleCapacity,
        unsigned int shapeCapacity,
        unsigned int geometryCapacity,
        unsigned int sdfDataCapacity,
        int buildGpuData,
        int numCudaStreams);

    CRMPM_C_API int CrGetInitializationStatus(void);

    CRMPM_C_API void CrReleaseEngine(void);

    /* Physics step */

    CRMPM_C_API void CrAdvance(
        CrSceneHandle scene,
        float dt);

    CRMPM_C_API void CrFetchResults(
        CrSceneHandle scene);

    CRMPM_C_API void CrAdvanceAll(float dt);

    CRMPM_C_API void CrFetchResultsAll();

    /* Simulation data */

    CRMPM_C_API CrParticleData CrGetParticleData(
        CrSceneHandle scene);

    CRMPM_C_API CrParticleData CrGetParticleDataGpu(
        CrSceneHandle scene);

    CRMPM_C_API CrParticleData CrGetParticleDataAll();

    CRMPM_C_API int CrGetSceneParticleDataGlobalOffset(CrSceneHandle scene);

    CRMPM_C_API void CrGetShapeCouplingForceTorque(
        CrShapeHandle shape,
        CrVec4f *outForce,
        CrVec4f *outTorque);

    CRMPM_C_API void CrSceneMarkDirty(CrSceneHandle scene, CrSceneDataDirtyFlags flags);

    CRMPM_C_API void CrSyncSceneDataIfNeeded(
        CrSceneHandle scene);

    /* Object creation */

    CRMPM_C_API CrSceneHandle CrCreateScene(
        CrMpmSolverType solverType,
        int numMaxParticles,
        CrVec3f3 gravity,
        CrBounds3 gridBounds,
        float gridCellSize,
        float solverIntegrationStepSize,
        int solverIterations,
        CrShapeContactModel contactModel);

    CRMPM_C_API CrShapeHandle CrCreateShape(
        CrGeometryHandle geometry,
        CrVec3f3 position,
        CrQuat rotation,
        CrShapeType shapeType,
        float sdfSmoothDistance,
        float sdfFatten,
        float drag,
        float friction,
        CrVec3f3 invScale,
        CrVec4f comInvMass,
        CrVec4f inertiaInv0,
        CrVec4f inertiaInv1);

    CRMPM_C_API CrGeometryHandle CrCreateGeometry(
        CrGeometryType geometryType,
        CrVec4f geometryParams);

    CRMPM_C_API CrGeometryHandle CrCreateTrimeshGeometry(
        CrGeometryType geometryType,
        void *points, /* Must be a CrVec3f3-like array */
        int numPoints,
        unsigned int *triangles,
        int numTriangles,
        float sdfCellSize,
        float sdfFatten,
        CrVec4f sdfSlicerSpineArea);

    CRMPM_C_API CrGeometryHandle CrCreateConnectedLineSegmentsGeometry(
        int numPoints,
        CrVec3f3 *points,
        float fattenBounds);

    CRMPM_C_API CrParticleObjectHandle CrCreateParticleObject(
        float particleSpacing,
        float particleMass,
        CrGeometryHandle geometry,
        const CrVec3f3 *position,
        const CrQuat *rotation,
        const CrVec3f3 *invScale,
        CrParticleMaterialType materialType,
        CrVec4f *materialParams,
        unsigned int *outNumParticles);

    /* Simulation object management */

    CRMPM_C_API void CrAddParticleObjectToScene(
        CrParticleObjectHandle particleObject,
        CrSceneHandle scene,
        unsigned int *outOffset);

    CRMPM_C_API void CrResetParticleObject(
        CrParticleObjectHandle particleObject
    );

    CRMPM_C_API void CrAddShapeToScene(
        CrShapeHandle shape,
        CrSceneHandle scene);

    CRMPM_C_API void CrSetShapePose(
        CrShapeHandle shape,
        CrVec3f3 *position,
        CrQuat *rotation);

    CRMPM_C_API void CrSetShapeVelocity(
        CrShapeHandle shape,
        CrVec3f3 *linear,
        CrVec3f3 *angular);

    CRMPM_C_API void CrSetShapeKinematicTarget(
        CrShapeHandle shape,
        CrVec3f3 *targetPosition,
        CrQuat *targetRotation,
        float dt);

    /* Object release */

    CRMPM_C_API void CrReleaseScene(
        CrSceneHandle scene);

    CRMPM_C_API void CrReleaseShape(
        CrShapeHandle shape);

    CRMPM_C_API void CrReleaseGeometry(
        CrGeometryHandle geometry);

    CRMPM_C_API void CrReleaseParticleObject(
        CrParticleObjectHandle pObj);

    /* Fast copy */

    CRMPM_C_API void CrFastCopy(
        void *src, void *dst, int size);

#ifdef __cplusplus
}
#endif

#endif // !CRMPM_C_API_H