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

#include <cuda_runtime.h>

#include "simulation_factory_impl.h"
#include "scene_impl.h"
#include "geometry_impl.h"
#include "shape_impl.h"
#include "particle_object_impl.h"

#include "check_cuda.cuh"

#include "grow_data_block.cuh"
#include "type_punning.h"

#include "mpm_solver_base.h"
#include "cpu_mls_mpm_solver.h"
#include "gpu_mls_mpm_solver.cuh"
#include "cpu_pb_mpm_solver.h"
#include "gpu_pb_mpm_solver.cuh"
#include "cpu_standard_mpm_solver.h"
#include "gpu_standard_mpm_solver.cuh"

#include "gpu_data_dirty_flags.h"
#include "simulation_factory.h"

namespace crmpm
{
    SimulationFactoryImpl::SimulationFactoryImpl(int particleCapacity,
                                                 int shapeCapacity,
                                                 int geometryCapacity,
                                                 int sdfDataCapacity,
                                                 bool buildGpuData,
                                                 int numCudaStreams)
        : mShapeCapacity(shapeCapacity),
          mGeometryCapacity(geometryCapacity),
          mShapeIndices(shapeCapacity),
          mGeometryIndices(geometryCapacity),
          mIsGpu(buildGpuData),
          mNumCudaStreams(numCudaStreams),
          mSdfDataCapacity(sdfDataCapacity),
          mParticleCapacity(particleCapacity),
          mCpuParticleData(particleCapacity),
          mParticleCpuDataManager(particleCapacity),
          mStatus(SimulationStatus::eIdle),
          mBusySceneCount(0),
          mAdvanceAllTaskFn(std::bind(&SimulationFactoryImpl::_advanceAllTaskFn, this, std::placeholders::_1))
    {
        mThreadManager.start(); // Seperate thread for advanceAll

        mCpuShapeData.size = shapeCapacity;
        mCpuShapeData.type = new ShapeType[mShapeCapacity];
        mCpuShapeData.geometryIdx = new int[mShapeCapacity];
        mCpuShapeData.position = new Vec3f[mShapeCapacity];
        mCpuShapeData.rotation = new Quat[mShapeCapacity];
        mCpuShapeData.linearVelocity = new Vec3f[mShapeCapacity];
        mCpuShapeData.angularVelocity = new Vec3f[mShapeCapacity];
        mCpuShapeData.invScale = new Vec3f[mShapeCapacity];
        mCpuShapeData.params0 = new float4[mShapeCapacity];

        mCpuGeometryData.size = geometryCapacity;
        mCpuGeometryData.type = new GeometryType[mGeometryCapacity];
        mCpuGeometryData.params0 = new float4[mGeometryCapacity];

        mCpuSdfData.dimemsion = new Vec3i[mGeometryCapacity];
        mCpuSdfData.lowerBoundCellSize = new float4[mGeometryCapacity];

        if (mIsGpu)
        {
            // Create CUDA streams
            for (int i = 0; i < mNumCudaStreams; ++i)
            {
                cudaStream_t stream;
                CR_CHECK_CUDA(cudaStreamCreate(&stream));
                mCudaStreams.pushBack(Pair<cudaStream_t, int>(stream, 0));
            }

            // Use pinned host memory for particle data
            CR_CHECK_CUDA(cudaMallocHost<float4>(&mCpuParticlePositionMass, mParticleCapacity * sizeof(float4)));
            CR_CHECK_CUDA(cudaMallocHost<Vec3f>(&mCpuParticleVelocity, mParticleCapacity * sizeof(Vec3f)));

            // Use pinned host memory for large SDF data
            CR_CHECK_CUDA(cudaMallocHost<float4>(&mCpuSdfData.gradientSignedDistance, sizeof(float4) * mSdfDataCapacity));

            // Use pinned host memory for Shape force/torque coupling
            CR_CHECK_CUDA(cudaMallocHost<float4>(&mCpuShapeData.force, sizeof(float4) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMallocHost<float4>(&mCpuShapeData.torque, sizeof(float4) * mShapeCapacity));

            CR_CHECK_CUDA(cudaMalloc<ShapeType>(&mGpuShapeData.type, sizeof(ShapeType) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMalloc<int>(&mGpuShapeData.geometryIdx, sizeof(int) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMalloc<Vec3f>(&mGpuShapeData.position, sizeof(Vec3f) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMalloc<Quat>(&mGpuShapeData.rotation, sizeof(Quat) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMalloc<Vec3f>(&mGpuShapeData.linearVelocity, sizeof(Vec3f) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMalloc<Vec3f>(&mGpuShapeData.angularVelocity, sizeof(Vec3f) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMalloc<Vec3f>(&mGpuShapeData.invScale, sizeof(Vec3f) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMalloc<float4>(&mGpuShapeData.params0, sizeof(float4) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMalloc<float4>(&mGpuShapeData.force, sizeof(float4) * mShapeCapacity));
            CR_CHECK_CUDA(cudaMalloc<float4>(&mGpuShapeData.torque, sizeof(float4) * mShapeCapacity));

            CR_CHECK_CUDA(cudaMalloc<GeometryType>(&mGpuGeometryData.type, sizeof(GeometryType) * mGeometryCapacity));
            CR_CHECK_CUDA(cudaMalloc<float4>(&mGpuGeometryData.params0, sizeof(float4) * mGeometryCapacity));

            CR_CHECK_CUDA(cudaMalloc<Vec3i>(&mGpuSdfData.dimemsion, sizeof(Vec3i) * mGeometryCapacity));
            CR_CHECK_CUDA(cudaMalloc<float4>(&mGpuSdfData.lowerBoundCellSize, sizeof(float4) * mGeometryCapacity));
            CR_CHECK_CUDA(cudaMalloc<float4>(&mGpuSdfData.gradientSignedDistance, sizeof(float4) * mSdfDataCapacity));
        }
        else
        {
            // Particle data block
            mCpuParticlePositionMass = new float4[mParticleCapacity];
            mCpuParticleVelocity = new Vec3f[mParticleCapacity];

            mCpuSdfData.gradientSignedDistance = new float4[mSdfDataCapacity];
            mCpuShapeData.force = new float4[mShapeCapacity];
            mCpuShapeData.torque = new float4[mShapeCapacity];
        }
        mCpuParticleData.positionMass = mCpuParticlePositionMass;
        mCpuParticleData.velocity = mCpuParticleVelocity;

        // Initialize SDF free list
        FreeSdfDataBlock freeSdfBlock;
        freeSdfBlock.offset = 0;
        freeSdfBlock.size = mSdfDataCapacity;
        mFreeSdfDataBlock.pushBack(freeSdfBlock);
    }

        SimulationFactoryImpl::~SimulationFactoryImpl()
        {
            // Must release all scenes, shapes, and geometries first.

            delete[] mCpuShapeData.type;
            delete[] mCpuShapeData.geometryIdx;
            delete[] mCpuShapeData.position;
            delete[] mCpuShapeData.rotation;
            delete[] mCpuShapeData.linearVelocity;
            delete[] mCpuShapeData.angularVelocity;
            delete[] mCpuShapeData.invScale;
            delete[] mCpuShapeData.params0;

            delete[] mCpuGeometryData.type;
            delete[] mCpuGeometryData.params0;

            delete[] mCpuSdfData.dimemsion;
            delete[] mCpuSdfData.lowerBoundCellSize;

            if (mIsGpu)
            {
                // Destroy CUDA streams
                for (Pair<cudaStream_t, int> stream : mCudaStreams)
                {
                    CR_CHECK_CUDA(cudaStreamDestroy(stream.first));
                }
                mCudaStreams.clear();

                // Free pinned host memory
                CR_CHECK_CUDA(cudaFreeHost(mCpuParticlePositionMass));
                CR_CHECK_CUDA(cudaFreeHost(mCpuParticleVelocity));
                CR_CHECK_CUDA(cudaFreeHost(mCpuSdfData.gradientSignedDistance));
                CR_CHECK_CUDA(cudaFreeHost(mCpuShapeData.force));
                CR_CHECK_CUDA(cudaFreeHost(mCpuShapeData.torque));

                // Free GPU memory
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.type));
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.geometryIdx));
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.position));
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.rotation));
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.linearVelocity));
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.angularVelocity));
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.invScale));
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.params0));
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.force));
                CR_CHECK_CUDA(cudaFree(mGpuShapeData.torque));

                CR_CHECK_CUDA(cudaFree(mGpuGeometryData.type));
                CR_CHECK_CUDA(cudaFree(mGpuGeometryData.params0));

                CR_CHECK_CUDA(cudaFree(mGpuSdfData.dimemsion));
                CR_CHECK_CUDA(cudaFree(mGpuSdfData.lowerBoundCellSize));
                CR_CHECK_CUDA(cudaFree(mGpuSdfData.gradientSignedDistance));
            }
            else
            {
                delete[] mCpuParticlePositionMass;
                delete[] mCpuParticleVelocity;
                delete[] mCpuSdfData.gradientSignedDistance;
                delete[] mCpuShapeData.force;
                delete[] mCpuShapeData.torque;
            }
        }

        bool SimulationFactoryImpl::isGpu() const
        {
            return mIsGpu;
        }

        void SimulationFactoryImpl::advanceAll(float dt)
        {
            bool isSceneIdle = true;
            for (SceneImpl *scene : mSceneList)
            {
                if (scene->getSimulationStatus() != SimulationStatus::eIdle)
                {
                    isSceneIdle = false;
                }
            }

            if (isSceneIdle && mStatus == SimulationStatus::eIdle)
            {
                mThreadManager.submitTask(mAdvanceAllTaskFn, &dt, sizeof(float));
                mStatus = SimulationStatus::eBusy;

                // Set all scenes to be busy
                for (SceneImpl *scene : mSceneList)
                {
                    scene->setSimulationStatus(SimulationStatus::eBusy);
                }
            }
            else
            {
                CR_DEBUG_LOG_WARNING("%s", "SimulationFactory is busy.");
            }
        }

        void SimulationFactoryImpl::fetchResultsAll()
        {
            if (mStatus != SimulationStatus::eBusy)
            {
                CR_DEBUG_LOG_WARNING("%s", "advanceAll() has not been called");
                return;
            }

            mThreadManager.waitForIdle();
            mStatus = SimulationStatus::eIdle;

            // Set all scene to be idle
            for (SceneImpl *scene : mSceneList)
            {
                scene->setSimulationStatus(SimulationStatus::eIdle);
            }
        }

        Scene *SimulationFactoryImpl::createScene(const SceneDesc &desc)
        {
            // Check if running out of particle capacity
            int particleDataOffset = mParticleCpuDataManager.request(desc.numMaxParticles);
            if (particleDataOffset < 0)
            {
                CR_DEBUG_LOG_ERROR("%s", "No enough particle capacity in SimulationFactory.");
                return nullptr;
            }

            MpmSolverBase *solver = nullptr;
            const float cellSize = desc.gridCellSize;
            Bounds3 gridBounds = desc.gridBounds;
            bool isSolverGpu;

            if (mIsGpu)
            {
                switch (desc.solverType)
                {
                case MpmSolverType::eCpuMlsMpmSolver:
                {
                    solver = new CpuMlsMpmSolver(desc.numMaxParticles, cellSize, gridBounds);
                    isSolverGpu = false;
                    break;
                }
                case MpmSolverType::eGpuMlsMpmSolver:
                {
                    solver = new GpuMlsMpmSolver(desc.numMaxParticles, cellSize, gridBounds);
                    isSolverGpu = true;
                    break;
                }
                case MpmSolverType::eCpuPbMpmSolver:
                {
                    solver = new CpuPbMpmSolver(desc.numMaxParticles, cellSize, gridBounds);
                    isSolverGpu = false;
                    break;
                }
                case MpmSolverType::eGpuPbMpmSolver:
                {
                    solver = new GpuPbMpmSolver(desc.numMaxParticles, cellSize, gridBounds);
                    isSolverGpu = true;
                    break;
                }
                case MpmSolverType::eCpuStandardMpmSolver:
                {
                    solver = new CpuStandardMpmSolver(desc.numMaxParticles, cellSize, gridBounds);
                    isSolverGpu = false;
                    break;
                }
                case MpmSolverType::eGpuStandardMpmSolver:
                {
                    solver = new GpuStandardMpmSolver(desc.numMaxParticles, cellSize, gridBounds);
                    isSolverGpu = true;
                    break;
                }

                default:
                    break;
                }

                if (solver && isSolverGpu)
                {
                    int streamIdx = getCudaStreamByLoad();

                    // GPU solvers must inherit from MpmSolverGpu
                    MpmSolverGpu *_solver = static_cast<MpmSolverGpu *>(solver);
                    _solver->setCudaStream(mCudaStreams[streamIdx].first);
                    mCudaStreams[streamIdx].second++; // Incrememt CUDA stream load.
                }
            }
            else
            {
                // If the factory doesn't build GPU data, fallback to CPU for all scenes.
                switch (desc.solverType)
                {
                case MpmSolverType::eGpuMlsMpmSolver:
                    CR_DEBUG_LOG_WARNING("%s", "Using GPU solver with a CPU-only factory. Fallback to CPU solver.");
                case MpmSolverType::eCpuMlsMpmSolver:
                {
                    solver = new CpuMlsMpmSolver(desc.numMaxParticles, cellSize, gridBounds);
                    isSolverGpu = false;
                    break;
                }
                case MpmSolverType::eGpuPbMpmSolver:
                    CR_DEBUG_LOG_WARNING("%s", "Using GPU solver with a CPU-only factory. Fallback to CPU solver.");
                case MpmSolverType::eCpuPbMpmSolver:
                {
                    solver = new CpuPbMpmSolver(desc.numMaxParticles, cellSize, gridBounds);
                    isSolverGpu = false;
                    break;
                }
                case MpmSolverType::eGpuStandardMpmSolver:
                    CR_DEBUG_LOG_WARNING("%s", "Using GPU solver with a CPU-only factory. Fallback to CPU solver.");
                case MpmSolverType::eCpuStandardMpmSolver:
                    solver = new CpuStandardMpmSolver(desc.numMaxParticles, cellSize, gridBounds);
                    isSolverGpu = false;
                    break;

                default:
                    break;
                }
            }

            SceneImpl *scene = nullptr;
            if (solver)
            {
                // Bind rigid data with solver
                if (isSolverGpu)
                {
                    solver->bindShapeData(mGpuShapeData);
                    solver->bindGeometryData(mGpuGeometryData);
                    solver->bindGeometrySdfData(mGpuSdfData);
                }
                else
                {
                    solver->bindShapeData(mCpuShapeData);
                    solver->bindGeometryData(mCpuGeometryData);
                    solver->bindGeometrySdfData(mCpuSdfData);
                }
                solver->setGravity(desc.gravity);
                solver->setIntegrationStepSize(desc.solverIntegrationStepSize);
                solver->setSolverIterations(desc.solverIterations);
                
                scene = new SceneImpl(desc.numMaxParticles, mShapeCapacity, mCpuParticlePositionMass, mCpuParticleVelocity, particleDataOffset);
                scene->setFactory(*this);
                scene->setSolver(*solver, isSolverGpu);
                scene->initialize();
            }

            if (scene && solver)
            {
                // Add to Scene list
                mSceneList.pushBack(scene);
                mSolverList.pushBack(solver);
            }
            else
            {
                // If anything fails, release requested particle data
                mParticleCpuDataManager.release(particleDataOffset, desc.numMaxParticles);
            }

            return scene;
        }

        Shape *SimulationFactoryImpl::createShape(Geometry &geom, const ShapeDesc &shapeDesc)
        {
            ShapeImpl *shape = nullptr;
            int nextShapeId;

            if (mShapeIndices.size() >= mShapeCapacity)
            {
                CR_DEBUG_LOG_WARNING("%s", "No enough capacity for shape data. Reallocating.");
                // Double the shape capacity
                int newCapacity = mShapeCapacity * 1.5 + 1;

                growCpuData(mCpuShapeData.type, mShapeCapacity, newCapacity);
                growCpuData(mCpuShapeData.geometryIdx, mShapeCapacity, newCapacity);
                growCpuData(mCpuShapeData.position, mShapeCapacity, newCapacity);
                growCpuData(mCpuShapeData.rotation, mShapeCapacity, newCapacity);
                growCpuData(mCpuShapeData.invScale, mShapeCapacity, newCapacity);
                growCpuData(mCpuShapeData.linearVelocity, mShapeCapacity, newCapacity);
                growCpuData(mCpuShapeData.angularVelocity, mShapeCapacity, newCapacity);
                growCpuData(mCpuShapeData.params0, mShapeCapacity, newCapacity);

                if (mIsGpu)
                {
                    // We need to do CPU->GPU copy anyway. Discard old data.
                    growGpuDataDiscardOld(mGpuShapeData.type, newCapacity);
                    growGpuDataDiscardOld(mGpuShapeData.geometryIdx, newCapacity);
                    growGpuDataDiscardOld(mGpuShapeData.position, newCapacity);
                    growGpuDataDiscardOld(mGpuShapeData.rotation, newCapacity);
                    growGpuDataDiscardOld(mGpuShapeData.invScale, newCapacity);
                    growGpuDataDiscardOld(mGpuShapeData.linearVelocity, newCapacity);
                    growGpuDataDiscardOld(mGpuShapeData.angularVelocity, newCapacity);
                    growGpuDataDiscardOld(mGpuShapeData.params0, newCapacity);
                }

                mShapeCapacity = newCapacity;
                mShapeIndices.reserve(mShapeCapacity);
                markDirty(SimulationFactoryGpuDataDirtyFlags::eShape);
            }

            for (nextShapeId = 0; nextShapeId < mShapeCapacity; ++nextShapeId)
            {
                if (mShapeIndices.find(nextShapeId) == mShapeIndices.end())
                {
                    // Initialize CPU data
                    mCpuShapeData.geometryIdx[nextShapeId] = geom.getId();
                    mCpuShapeData.type[nextShapeId] = shapeDesc.type;
                    mCpuShapeData.position[nextShapeId] = shapeDesc.transform.position;
                    mCpuShapeData.rotation[nextShapeId] = shapeDesc.transform.rotation;
                    mCpuShapeData.linearVelocity[nextShapeId] = Vec3f();
                    mCpuShapeData.angularVelocity[nextShapeId] = Vec3f();
                    mCpuShapeData.invScale[nextShapeId] = shapeDesc.invScale;
                    mCpuShapeData.params0[nextShapeId] = make_float4(shapeDesc.sdfSmoothDistance, shapeDesc.drag, shapeDesc.friction, shapeDesc.sdfFatten);

                    shape = new ShapeImpl();

                    // Bind data
                    shape->setFactory(*this);
                    shape->setId(nextShapeId);
                    shape->bindShapeData(mCpuShapeData);

                    // Link the geometry
                    shape->setGeometry(geom);

                    mShapeIndices.pushBack(nextShapeId);

                    // Mark shape data in GPU dirty
                    markDirty(SimulationFactoryGpuDataDirtyFlags::eShape);
                    break;
                }
            }

            return shape;
        }

        Geometry *SimulationFactoryImpl::createGeometry(GeometryType type, float4 params0)
        {
            checkAndGrowGeometryData();

            GeometryImpl *geom = nullptr;
            int nextGeometryId;
            for (nextGeometryId = 0; nextGeometryId < mGeometryCapacity; ++nextGeometryId)
            {
                // if not in the list
                if (mGeometryIndices.find(nextGeometryId) == mGeometryIndices.end())
                {
                    switch (type)
                    {
                    case GeometryType::eBox:
                    {
                        geom = new GeometryImpl();
                        Bounds3 bounds;
                        bounds.minimum = Vec3f(-params0.x, -params0.y, -params0.z);
                        bounds.maximum = Vec3f(params0.x, params0.y, params0.z);
                        geom->setBounds(bounds);
                        break;
                    }

                    case GeometryType::eSphere:
                    {
                        geom = new GeometryImpl();
                        Bounds3 bounds;
                        bounds.minimum = Vec3f(-params0.x, -params0.x, -params0.x);
                        bounds.maximum = Vec3f(params0.x, params0.x, params0.x);
                        geom->setBounds(bounds);
                        break;
                    }

                    case GeometryType::ePlane:
                    {
                        geom = new GeometryImpl();
                        Bounds3 bounds;
                        bounds.minimum = Vec3f(-CR_MAX_F32, -CR_MAX_F32, -CR_MAX_F32);
                        bounds.maximum = Vec3f(CR_MAX_F32, 0, CR_MAX_F32);
                        geom->setBounds(bounds);
                        break;
                    }

                    case GeometryType::eCapsule:
                    {
                        geom = new GeometryImpl();
                        Bounds3 bounds;
                        bounds.minimum = Vec3f(-params0.x, -params0.x - params0.y, -params0.x);
                        bounds.maximum = Vec3f(params0.x, params0.x + params0.y, params0.x);
                        geom->setBounds(bounds);
                        break;
                    }

                    case GeometryType::eQuadSlicer:
                    {
                        geom = new GeometryImpl();
                        Bounds3 bounds;
                        bounds.minimum = Vec3f(-params0.x, 0, -params0.y);
                        bounds.maximum = Vec3f(params0.x, 0, params0.y);
                        geom->setBounds(bounds);
                        break;
                    }

                    case GeometryType::eArc:
                    {
                        geom = new GeometryImpl();
                        Bounds3 bounds;
                        bounds.minimum = Vec3f(-CR_MAX_F32, -CR_MAX_F32, -CR_MAX_F32);
                        bounds.maximum = Vec3f(CR_MAX_F32, CR_MAX_F32, CR_MAX_F32);
                        geom->setBounds(bounds);
                        break;
                    }

                    default:
                        // Invalid type, return nullptr
                        return geom;
                    }

                    mCpuGeometryData.params0[nextGeometryId] = params0;
                    mCpuGeometryData.type[nextGeometryId] = type;
                    geom->setFactory(*this);
                    geom->setId(nextGeometryId);
                    geom->bindGeometryData(mCpuGeometryData, mCpuSdfData);
                    mGeometryIndices.tryPushBack(nextGeometryId);

                    // Mark geometry data in GPU dirty
                    markDirty(SimulationFactoryGpuDataDirtyFlags::eGeometry);
                    break;
                }
            }

            return geom;
        }

        Geometry *SimulationFactoryImpl::createGeometry(GeometryType type, TriangleMesh &mesh, SdfDesc &sdfDesc)
        {
            checkAndGrowGeometryData();

            GeometryImpl *geom = nullptr;
            int nextGeometryId;
            for (nextGeometryId = 0; nextGeometryId < mGeometryCapacity; ++nextGeometryId)
            {
                // if not in the list
                if (mGeometryIndices.find(nextGeometryId) == mGeometryIndices.end())
                {
                    switch (type)
                    {
                    case GeometryType::eTriangleMesh:
                    {
                        geom = new GeometryImpl();
                        mCpuGeometryData.type[nextGeometryId] = type;

                        int sdfDataId = getNewSdfIndex();
                        if (sdfDataId < 0)
                        {
                            // It shouldn't reach here but just in case.
                            CR_DEBUG_LOG_WARNING("%s", "Can't find an SDF ID.");
                            break;
                        }
                        mGeometrySdfIndices.pushBack(sdfDataId);
                        mCpuGeometryData.params0[nextGeometryId].y = CR_INT_AS_FLOAT(sdfDataId);

                        SdfBuilder::precomputeSdfDesc(sdfDesc, mesh);
                        unsigned int requiredSize = sdfDesc.sdfDimension.x() * sdfDesc.sdfDimension.y() * sdfDesc.sdfDimension.z();

                        int sdfOffset = getSdfDataOffset(requiredSize);
                        mCpuGeometryData.params0[nextGeometryId].x = CR_INT_AS_FLOAT(sdfOffset);
                        mCpuGeometryData.params0[nextGeometryId].z = CR_INT_AS_FLOAT(requiredSize);

                        mCpuSdfData.dimemsion[sdfDataId] = sdfDesc.sdfDimension;
                        mCpuSdfData.lowerBoundCellSize[sdfDataId] = sdfDesc.lowerBound.data;
                        mCpuSdfData.lowerBoundCellSize[sdfDataId].w = sdfDesc.cellSize;
                        SdfBuilder::computeSdf(sdfDesc, mesh, mCpuSdfData.gradientSignedDistance + sdfOffset);
                        SdfBuilder::computeSdfGradient(sdfDesc, mCpuSdfData.gradientSignedDistance + sdfOffset);

                        Bounds3 geomBounds;
                        geomBounds.minimum = sdfDesc.lowerBound;
                        geomBounds.maximum = sdfDesc.lowerBound + sdfDesc.boundingSize;

                        geom->setFactory(*this);
                        geom->setId(nextGeometryId);
                        geom->setBounds(geomBounds);
                        geom->bindGeometryData(mCpuGeometryData, mCpuSdfData);
                        mGeometryIndices.tryPushBack(nextGeometryId);
                        break;
                    }

                    case GeometryType::eTriangleMeshSlicer:
                    {
                        geom = new GeometryImpl();
                        mCpuGeometryData.type[nextGeometryId] = type;
                        int sdfDataId = getNewSdfIndex();
                        if (sdfDataId < 0)
                        {
                            // It shouldn't reach here but just in case.
                            CR_DEBUG_LOG_WARNING("%s", "Can't find an SDF ID.");
                            break;
                        }
                        mGeometrySdfIndices.pushBack(sdfDataId);
                        mCpuGeometryData.params0[nextGeometryId].y = CR_INT_AS_FLOAT(sdfDataId);

                        SdfBuilder::precomputeSdfDesc(sdfDesc, mesh);
                        unsigned int requiredSize = sdfDesc.sdfDimension.x() * sdfDesc.sdfDimension.y() * sdfDesc.sdfDimension.z() + 1; // The first element is the slicer spine area

                        int sdfOffset = getSdfDataOffset(requiredSize);
                        mCpuGeometryData.params0[nextGeometryId].x = CR_INT_AS_FLOAT(sdfOffset);
                        mCpuGeometryData.params0[nextGeometryId].z = CR_INT_AS_FLOAT(requiredSize);

                        mCpuSdfData.dimemsion[sdfDataId] = sdfDesc.sdfDimension;
                        mCpuSdfData.lowerBoundCellSize[sdfDataId] = sdfDesc.lowerBound.data;
                        mCpuSdfData.lowerBoundCellSize[sdfDataId].w = sdfDesc.cellSize;
                        mCpuSdfData.gradientSignedDistance[sdfOffset] = sdfDesc.slicerSpineArea;

                        Array<bool> projectionOnMeshList;
                        SdfBuilder::computeModSdf(sdfDesc, mesh, mCpuSdfData.gradientSignedDistance + sdfOffset + 1, projectionOnMeshList);
                        SdfBuilder::computeSdfGradient(sdfDesc, mCpuSdfData.gradientSignedDistance + sdfOffset + 1);
                        SdfBuilder::postProcessModSdfGradient(sdfDesc, mCpuSdfData.gradientSignedDistance + sdfOffset + 1, projectionOnMeshList);

                        geom->setFactory(*this);
                        geom->setId(nextGeometryId);
                        geom->bindGeometryData(mCpuGeometryData, mCpuSdfData);
                        mGeometryIndices.tryPushBack(nextGeometryId);
                        break;
                    }

                    default:
                        break;
                    }

                    // Mark geometry data in GPU dirty
                    markDirty(SimulationFactoryGpuDataDirtyFlags::eGeometry);
                    markDirty(SimulationFactoryGpuDataDirtyFlags::eGeometrySdfData);
                    break;
                }
            }

            return geom;
        }

        Geometry *SimulationFactoryImpl::createGeometry(GeometryType type, int numPoints, Vec3f *points, float fattenBounds)
        {
            checkAndGrowGeometryData();

            GeometryImpl *geom = nullptr;
            int nextGeometryId;
            for (nextGeometryId = 0; nextGeometryId < mGeometryCapacity; ++nextGeometryId)
            {
                // if not in the list
                if (mGeometryIndices.find(nextGeometryId) == mGeometryIndices.end())
                {
                    switch (type)
                    {
                    case GeometryType::eConnectedLineSegments:
                    {
                        geom = new GeometryImpl();
                        mCpuGeometryData.type[nextGeometryId] = type;

                        int sdfDataId = getNewSdfIndex();
                        if (sdfDataId < 0)
                        {
                            // It shouldn't reach here but just in case.
                            CR_DEBUG_LOG_WARNING("%s", "Can't find an SDF ID.");
                            break;
                        }
                        mGeometrySdfIndices.pushBack(sdfDataId);
                        mCpuGeometryData.params0[nextGeometryId].y = CR_INT_AS_FLOAT(sdfDataId);

                        unsigned int requiredSize = numPoints + 1; // Add one for storing the upper bound
                        int sdfOffset = getSdfDataOffset(requiredSize);
                        mCpuGeometryData.params0[nextGeometryId].x = CR_INT_AS_FLOAT(sdfOffset);
                        mCpuGeometryData.params0[nextGeometryId].z = CR_INT_AS_FLOAT(requiredSize);

                        Bounds3 bounds;
                        for (int p = 0; p < numPoints; ++p)
                        {
                            bounds.include(points[p]);
                        }
                        bounds.fattenFast(fattenBounds);

                        mCpuSdfData.lowerBoundCellSize[sdfDataId] = bounds.minimum.data;
                        mCpuSdfData.gradientSignedDistance[sdfOffset] = bounds.maximum.data;
                        for (int p = 0; p < numPoints; ++p)
                        {
                            mCpuSdfData.gradientSignedDistance[sdfOffset + p + 1] = points[p].data;
                        }

                        geom->setFactory(*this);
                        geom->setId(nextGeometryId);
                        geom->setBounds(bounds);
                        geom->bindGeometryData(mCpuGeometryData, mCpuSdfData);
                        mGeometryIndices.tryPushBack(nextGeometryId);
                        break;
                    }

                    default:
                        break;
                    }

                    // Mark geometry data in GPU dirty
                    markDirty(SimulationFactoryGpuDataDirtyFlags::eGeometry);
                    markDirty(SimulationFactoryGpuDataDirtyFlags::eGeometrySdfData);
                    break;
                }
            }

            return geom;
        }

        ParticleObject *SimulationFactoryImpl::createParticleObject(const ParticleObjectDesc &desc)
        {
            ParticleObjectImpl *po = new ParticleObjectImpl();
            po->setParticleMass(desc.particleMass);
            po->setMaterial(desc.materialType, desc.materialParams);
            po->initialize(desc.particleSpacing, *desc.geometry, desc.position, desc.rotation, desc.invScale);
            return po;
        }

#ifdef _DEBUG
        // Debug print geometry ids
        void SimulationFactoryImpl::printGeometries()
        {
            for (int *id = mGeometryIndices.begin(); id != mGeometryIndices.end(); ++id)
            {
                CR_DEBUG_LOG_INFO("%d", *id);
            }
        }

        // Debug print shape ids
        void SimulationFactoryImpl::printShapes()
        {
            for (int *id = mShapeIndices.begin(); id != mShapeIndices.end(); ++id)
            {
                CR_DEBUG_LOG_INFO("%d", *id);
                CR_DEBUG_LOG_INFO("%d", mCpuShapeData.geometryIdx[*id]);
            }
        }
#endif

        void SimulationFactoryImpl::releaseScene(Scene *scene)
        {
            SceneImpl *sceneImpl = static_cast<SceneImpl *>(scene);

            // Free the particle data
            int particleDataOffset = sceneImpl->getParticleDataGlobalOffset();
            int maxNumParticles = sceneImpl->getMaxNumParticles();
            mParticleCpuDataManager.release(particleDataOffset, maxNumParticles);

            // Remove from scene/solver lists
            int index = mSceneList.find(sceneImpl) - mSceneList.begin();
            mSceneList.remove(index);
            mSolverList.remove(index);
        }

        void SimulationFactoryImpl::releaseShape(Shape *shape)
        {
            int id = shape->getId();
            mShapeIndices.remove(mShapeIndices.find(id) - mShapeIndices.begin());
        }

        void SimulationFactoryImpl::releaseGeometry(Geometry *geom)
        {
            int id = geom->getId();
            mGeometryIndices.remove(mGeometryIndices.find(id) - mGeometryIndices.begin());

            // For geometry that stores SDF data
            GeometryType type = geom->getType();
            if (type == GeometryType::eTriangleMesh || type == GeometryType::eTriangleMeshSlicer || type == GeometryType::eConnectedLineSegments)
            {
                float4 geomParams = geom->getParams();
                int sdfOffset = CR_FLOAT_AS_INT(geomParams.x);
                int sdfId = CR_FLOAT_AS_INT(geomParams.y);
                int sdfSize = CR_FLOAT_AS_INT(geomParams.z);
                mGeometrySdfIndices.remove(mGeometrySdfIndices.find(sdfId) - mGeometrySdfIndices.begin());
                mFreeSdfDataBlock.pushBack({sdfOffset, sdfSize});

                // Sort the offset
                std::sort(mFreeSdfDataBlock.begin(),
                          mFreeSdfDataBlock.end(),
                          [](const FreeSdfDataBlock &a, const FreeSdfDataBlock &b)
                          { return a.offset <= b.offset; });

                // Merge adjacent
                Array<FreeSdfDataBlock> mergedList;
                long long currentOffset = mFreeSdfDataBlock[0].offset;
                long long currentSize = mFreeSdfDataBlock[0].size;

                for (int i = 1; i < mFreeSdfDataBlock.size(); i++)
                {
                    if (currentOffset + currentSize == mFreeSdfDataBlock[i].offset)
                    {
                        // Merge adjacent blocks
                        currentSize += mFreeSdfDataBlock[i].size;
                    }
                    else
                    {
                        // Store merged block and move to the next
                        mergedList.pushBack({currentOffset, currentSize});
                        currentOffset = mFreeSdfDataBlock[i].offset;
                        currentSize = mFreeSdfDataBlock[i].size;
                    }
                }

                // Add the last merged block
                mergedList.pushBack({currentOffset, currentSize});

                // Copy merged back
                mFreeSdfDataBlock.forceSizeUnsafe(0);
                for (int i = 0; i < mergedList.size(); ++i)
                {
                    mFreeSdfDataBlock.pushBack(mergedList[i]);
                }
            }
        }

        void SimulationFactoryImpl::markDirty(SimulationFactoryGpuDataDirtyFlags flags)
        {
            mGpuDataDirtyFlags |= flags;
        }

        ParticleData &SimulationFactoryImpl::getParticleDataAll()
        {
            return mCpuParticleData;
        }

        void SimulationFactoryImpl::resetDirtyFlags()
        {
            mGpuDataDirtyFlags = SimulationFactoryGpuDataDirtyFlags::eNone;
        }

        void SimulationFactoryImpl::syncCpuToGpuIfNeeded()
        {
            if (!mIsGpu || mGpuDataDirtyFlags == SimulationFactoryGpuDataDirtyFlags::eNone)
                return;

            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapeType)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.type, mCpuShapeData.type, mShapeCapacity * sizeof(ShapeType), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapeGeometryIdx)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.geometryIdx, mCpuShapeData.geometryIdx, mShapeCapacity * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapePosition)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.position, mCpuShapeData.position, mShapeCapacity * sizeof(Vec3f), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapeRotation)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.rotation, mCpuShapeData.rotation, mShapeCapacity * sizeof(Quat), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapeLinearVelocity)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.linearVelocity, mCpuShapeData.linearVelocity, mShapeCapacity * sizeof(Vec3f), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapeAngularVelocity)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.angularVelocity, mCpuShapeData.angularVelocity, mShapeCapacity * sizeof(Vec3f), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapeScale)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.invScale, mCpuShapeData.invScale, mShapeCapacity * sizeof(Vec3f), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapeParams0)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.params0, mCpuShapeData.params0, mShapeCapacity * sizeof(float4), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapeForce)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.force, mCpuShapeData.force, mShapeCapacity * sizeof(float4), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eShapeTorque)
                CR_CHECK_CUDA(cudaMemcpy(mGpuShapeData.torque, mCpuShapeData.torque, mShapeCapacity * sizeof(float4), cudaMemcpyKind::cudaMemcpyHostToDevice));

            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eGeometryType)
                CR_CHECK_CUDA(cudaMemcpy(mGpuGeometryData.type, mCpuGeometryData.type, mGeometryCapacity * sizeof(GeometryType), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eGeometryParams0)
                CR_CHECK_CUDA(cudaMemcpy(mGpuGeometryData.params0, mCpuGeometryData.params0, mGeometryCapacity * sizeof(float4), cudaMemcpyKind::cudaMemcpyHostToDevice));
            if (mGpuDataDirtyFlags & SimulationFactoryGpuDataDirtyFlags::eGeometrySdfData)
            {
                CR_CHECK_CUDA(cudaMemcpy(mGpuSdfData.dimemsion, mCpuSdfData.dimemsion, mGeometryCapacity * sizeof(Vec3i), cudaMemcpyKind::cudaMemcpyHostToDevice));
                CR_CHECK_CUDA(cudaMemcpy(mGpuSdfData.lowerBoundCellSize, mCpuSdfData.lowerBoundCellSize, mGeometryCapacity * sizeof(float4), cudaMemcpyKind::cudaMemcpyHostToDevice));
                CR_CHECK_CUDA(cudaMemcpy(mGpuSdfData.gradientSignedDistance, mCpuSdfData.gradientSignedDistance, mSdfDataCapacity * sizeof(float4), cudaMemcpyKind::cudaMemcpyHostToDevice));
            }
        }

        void SimulationFactoryImpl::beforeSceneAdvance()
        {
            syncCpuToGpuIfNeeded();
            resetDirtyFlags();
        }

        void SimulationFactoryImpl::afterSceneAdvance(bool isGpuScene)
        {
            if (!(mIsGpu && isGpuScene))
                return;

            // Copy GPU shape coupling data to CPU
            CR_CHECK_CUDA(cudaMemcpy(mCpuShapeData.force, mGpuShapeData.force, mShapeCapacity * sizeof(float4), cudaMemcpyKind::cudaMemcpyDeviceToHost));
            CR_CHECK_CUDA(cudaMemcpy(mCpuShapeData.torque, mGpuShapeData.torque, mShapeCapacity * sizeof(float4), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }

        SimulationStatus SimulationFactoryImpl::getSimulationStatus()
        {
            return mStatus;
        }

        void SimulationFactoryImpl::checkAndGrowGeometryData()
        {
            if (mGeometryIndices.size() >= mGeometryCapacity)
            {
                CR_DEBUG_LOG_WARNING("%s", "No enough capacity for geometry data. Reallocating.");
                // Double the geometry capacity
                int newCapacity = mGeometryCapacity * 1.5 + 1;

                growCpuData(mCpuGeometryData.type, mGeometryCapacity, newCapacity);
                growCpuData(mCpuGeometryData.params0, mGeometryCapacity, newCapacity);
                growCpuData(mCpuSdfData.dimemsion, mGeometryCapacity, newCapacity);
                growCpuData(mCpuSdfData.lowerBoundCellSize, mGeometryCapacity, newCapacity);

                if (mIsGpu)
                {
                    // We need to do CPU->GPU copy anyway. Discard old data.
                    growGpuDataDiscardOld(mGpuGeometryData.type, newCapacity);
                    growGpuDataDiscardOld(mGpuGeometryData.params0, newCapacity);
                    growGpuDataDiscardOld(mGpuSdfData.dimemsion, newCapacity);
                    growGpuDataDiscardOld(mGpuSdfData.lowerBoundCellSize, newCapacity);
                }

                mGeometryCapacity = newCapacity;
                mGeometryIndices.reserve(mGeometryCapacity);
                markDirty(SimulationFactoryGpuDataDirtyFlags::eGeometry);
            }
        }

        /**
         * @brief get SDF data offset from the allocated SDF data buffer.
         * Use freed space that fits in the block first. Otherwise, append to the end.
         * If there is no enough space, the SDF data buffer will be reallocated.
         */
        size_t SimulationFactoryImpl::getSdfDataOffset(size_t requestedSize)
        {
            size_t allocOffset;

            long long bestIndex = -1;
            long long bestFitSize = SIZE_MAX;
            long long lastFreeBlockIndex = -1;
            long long lastFreeBlockOffset = -1;
        
            // Find the smallest free block that fits
            for (size_t i = 0; i < mFreeSdfDataBlock.size(); i++)
            {
                if (mFreeSdfDataBlock[i].size >= requestedSize && mFreeSdfDataBlock[i].size < bestFitSize)
                {
                    bestFitSize = mFreeSdfDataBlock[i].size;
                    bestIndex = i;
                }
                if (mFreeSdfDataBlock[i].offset > lastFreeBlockOffset)
                {
                    lastFreeBlockIndex = i;
                    lastFreeBlockOffset = mFreeSdfDataBlock[i].offset;
                }
            }

            if (bestIndex == -1)
            {
                // No enough space, reallocate the buffer
                size_t oldCapacity = mSdfDataCapacity;
                mSdfDataCapacity += requestedSize * 2;

                if (mIsGpu)
                {
                    growPinnedData<float4>(mCpuSdfData.gradientSignedDistance, oldCapacity, mSdfDataCapacity);

                    // We need to do CPU->GPU copy anyway. Discard old data.
                    growGpuDataDiscardOld<float4>(mGpuSdfData.gradientSignedDistance, mSdfDataCapacity);
                }
                else
                {
                    growCpuData<float4>(mCpuSdfData.gradientSignedDistance, oldCapacity, mSdfDataCapacity);
                }

                if (lastFreeBlockOffset != -1 && mFreeSdfDataBlock[lastFreeBlockIndex].offset + mFreeSdfDataBlock[lastFreeBlockIndex].size == oldCapacity)
                {
                    // The last free block is at the end of the old buffer
                    allocOffset = mFreeSdfDataBlock[lastFreeBlockIndex].offset;

                    mFreeSdfDataBlock[lastFreeBlockIndex].offset += requestedSize;
                    mFreeSdfDataBlock[lastFreeBlockIndex].size = mSdfDataCapacity - mFreeSdfDataBlock[lastFreeBlockIndex].offset;
                }
                else
                {
                    // The end of the old buffer was occupied, or the whole buffer was occupied
                    allocOffset = oldCapacity;

                    FreeSdfDataBlock newBlock;
                    newBlock.offset = oldCapacity + requestedSize;
                    newBlock.size = mSdfDataCapacity - newBlock.offset;
                    mFreeSdfDataBlock.pushBack(newBlock);
                }
            }
            else
            {
                // Allocate memory from the best-fit block
                allocOffset = mFreeSdfDataBlock[bestIndex].offset;

                if (mFreeSdfDataBlock[bestIndex].size == requestedSize)
                {
                    // Perfect match, remove the block
                    mFreeSdfDataBlock.remove(bestIndex);
                }
                else
                {
                    // Split the block, keep the remaining part
                    mFreeSdfDataBlock[bestIndex].offset += requestedSize;
                    mFreeSdfDataBlock[bestIndex].size -= requestedSize;
                }
            }

            return allocOffset;
        }

        int SimulationFactoryImpl::getNewSdfIndex()
        {
            for (int nextId = 0; nextId < mGeometryCapacity; ++nextId)
            {
                // if not in the list
                if (mGeometrySdfIndices.find(nextId) == mGeometrySdfIndices.end())
                {
                    return nextId;
                }
            }
            return -1;
        }

        /**
         * Get the stream index with minimum load
         */
        int SimulationFactoryImpl::getCudaStreamByLoad()
        {
            int best = 0;
            for (int i = 1; i < mNumCudaStreams; ++i)
            {
                if (mCudaStreams[i].second < mCudaStreams[best].second)
                {
                    best = i;
                }
            }
            return best;
        }

        void SimulationFactoryImpl::_advanceAllTaskFn(void *data)
        {
            for (SceneImpl *scene : mSceneList)
            {
                scene->beforeAdvance();
            }

            // SimulationFactory uses default stream.
            // Device synced before any cudaMemcpy unless
            // Using per-thread default stream
            // TODO: we should use it for better practice.
            // But if we stick to only use the default stream
            // in the factory, this should be okay, because
            // all Scenes/Solvers use their own streams.
            beforeSceneAdvance();

            float dt = *((float *)data);
            for (MpmSolverBase *solver : mSolverList)
            {
                float timeAdvanced = 0;
                while (timeAdvanced < dt)
                {
                    // Step all solvers iteratively
                    // This won't help anything on the CPU,
                    // but should reduce GPU sync.
                    // Can be further optimized with CUDA streams
                    {
                        timeAdvanced += solver->step();
                    }
                }
            }

            // Fetch results
            if (mIsGpu)
            {
                cudaDeviceSynchronize();
            }

            for (SceneImpl *scene : mSceneList)
            {
                scene->afterAdvance();
            }

            afterSceneAdvance(true);
        }

} // namespace crmpm
