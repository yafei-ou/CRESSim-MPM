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

#include "scene_impl.h"
#include "shape_impl.h"
#include "check_cuda.cuh"
#include "simulation_factory_impl.h"
#include "grow_data_block.cuh"
#include "material_data.h"
#include "mpm_solver_gpu.h"

namespace crmpm
{
    void SceneImpl::initialize()
    {
        // Initialize a separate thread
        mThreadManager.start();

        mCpuParticleData.positionMass = mParticlePositionMass;
        mCpuParticleData.velocity = mParticleVelocity;

        mParticleMaterialTypes = new ParticleMaterialType[mMaxNumParticles];
        mParticleMaterialPropertiesGroup1 = new float4[mMaxNumParticles];
        mCpuParticleMaterialData.params0 = mParticleMaterialPropertiesGroup1;
        mCpuParticleMaterialData.type = mParticleMaterialTypes;

        mActiveParticleMask = new unsigned char[mMaxNumParticles];

        // For GPU solvers
        if (mIsGpu)
        {
            CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticlePositionMass, mMaxNumParticles * sizeof(float4), mCudaStream));
            CR_CHECK_CUDA(cudaMallocAsync<Vec3f>(&dmParticleVelocity, mMaxNumParticles * sizeof(Vec3f), mCudaStream));
            CR_CHECK_CUDA(cudaMallocAsync<float4>(&dmParticleMaterialPropertiesGroup1, mMaxNumParticles * sizeof(float4), mCudaStream));
            CR_CHECK_CUDA(cudaMallocAsync<ParticleMaterialType>(&dmParticleMaterialTypes, mMaxNumParticles * sizeof(ParticleMaterialType), mCudaStream));

            CR_CHECK_CUDA(cudaMemsetAsync(dmParticlePositionMass, 0, mMaxNumParticles * sizeof(float4), mCudaStream));
            CR_CHECK_CUDA(cudaMemsetAsync(dmParticleVelocity, 0, mMaxNumParticles * sizeof(Vec3f), mCudaStream));
            CR_CHECK_CUDA(cudaMemsetAsync(dmParticleMaterialPropertiesGroup1, 0, mMaxNumParticles * sizeof(float4), mCudaStream));
            CR_CHECK_CUDA(cudaMemsetAsync(dmParticleMaterialTypes, 0, mMaxNumParticles * sizeof(ParticleMaterialType), mCudaStream));

            mGpuParticleData.positionMass = dmParticlePositionMass;
            mGpuParticleData.velocity = dmParticleVelocity;
            mGpuParticleMaterialData.params0 = dmParticleMaterialPropertiesGroup1;
            mGpuParticleMaterialData.type = dmParticleMaterialTypes;

            CR_CHECK_CUDA(cudaMallocAsync<int>(&dmShapeIds, mShapeCapacity * sizeof(int), mCudaStream));
            CR_CHECK_CUDA(cudaMallocAsync<unsigned char>(&dmActiveParticleMask, mMaxNumParticles * sizeof(unsigned char), mCudaStream));
            CR_CHECK_CUDA(cudaMallocAsync<unsigned int>(&dmComputeInitialDataIndices, mMaxNumParticles * sizeof(int), mCudaStream));

            mSolver->bindParticleData(mGpuParticleData);
            mSolver->bindParticleMaterials(mGpuParticleMaterialData);
            mSolver->bindShapeIds(dmShapeIds);
            mSolver->bindActiveMask(dmActiveParticleMask);
        }
        else
        {
            mSolver->bindParticleData(mCpuParticleData);
            mSolver->bindParticleMaterials(mCpuParticleMaterialData);
            mSolver->bindShapeIds(mShapeIds.begin());
            mSolver->bindActiveMask(mActiveParticleMask);
        }

        mSolver->initialize();
    }

    void SceneImpl::advance(float dt)
    {
        // Only advance if both the scene and the factory (advanceAll) are idle
        if (mStatus == SimulationStatus::eIdle &&
            mFactory->getSimulationStatus() == SimulationStatus::eIdle)
        {
            mThreadManager.submitTask(mAdvanceTaskFn, &dt, sizeof(float));
            mStatus = SimulationStatus::eBusy;
        }
        else
        {
            CR_DEBUG_LOG_WARNING("%s", "Results have not been fetched after advance() or advanceAll().");
        }
    }

    void SceneImpl::fetchResults()
    {
        if (mFactory->getSimulationStatus() == SimulationStatus::eBusy)
        {
            // If the factory is busy due to advanceAll(),
            // all scenes must be busy.
            CR_DEBUG_LOG_WARNING("%s", "SimulationFactory->advanceAll() was called. All results will be fetched.");
            mFactory->fetchResultsAll();
        }
        else if (mStatus == SimulationStatus::eBusy)
        {
            mThreadManager.waitForIdle();
            mStatus = SimulationStatus::eIdle;
        }
        else
        {
            CR_DEBUG_LOG_WARNING("%s", "Scene->advance() has not been called");
        }
    }

    void SceneImpl::setActiveMaskRange(unsigned int start, unsigned int length, char value)
    {
        if (mIsGpu)
        {
            CR_CHECK_CUDA(cudaMemsetAsync(dmActiveParticleMask + start, value, length, mCudaStream))
        }

        // Always set host array in case user wants to read using getActiveMask().
        memset(mActiveParticleMask + start, value, length);
    }

    void SceneImpl::addShape(Shape *shape)
    {
        // Prevent adding a shape twice
        if (mShapeIds.find(shape->getId()) != mShapeIds.end())
        {
            CR_DEBUG_LOG_WARNING("%s", "Attempted to add the same shape to a scene twice. Discarded.");
            return;
        }

        // Grow shape capacity if needed
        if (mShapeIds.size() == mShapeCapacity)
        {
            int newCapacity = mShapeCapacity * 1.5 + 1;
            mShapeIds.reserve(newCapacity);

            if (mIsGpu)
            {
                growGpuData(dmShapeIds, mShapeCapacity, newCapacity);
                mSolver->bindShapeIds(dmShapeIds);
            }
            else
            {
                mSolver->bindShapeIds(mShapeIds.begin());
            }

            mShapeCapacity = newCapacity;
        }

        // Add to scene
        static_cast<ShapeImpl *>(shape)->incrementRef();
        mShapeIds.tryPushBack(shape->getId());
        mShapes.pushBack(static_cast<ShapeImpl *>(shape));
        if (mIsGpu)
        {
            CR_CHECK_CUDA(cudaMemcpyAsync(dmShapeIds, mShapeIds.begin(), mShapeCapacity * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice, mCudaStream));
        }
        mSolver->setNumShapes(mShapeIds.size());
    }

    void SceneImpl::syncDataIfNeeded()
    {
        if (mDataDirtyFlags == SceneDataDirtyFlags::eNone)
            return;

        if (mDataDirtyFlags & SceneDataDirtyFlags::eNumParticles)
        {
            mSolver->setNumActiveParticles(mNumAllocatedParticles);
        }

        if (mIsGpu && mDataDirtyFlags & SceneDataDirtyFlags::eParticlePositionMass)
            CR_CHECK_CUDA(cudaMemcpyAsync(mGpuParticleData.positionMass, mCpuParticleData.positionMass, mMaxNumParticles * sizeof(float4), cudaMemcpyKind::cudaMemcpyHostToDevice, mCudaStream));

        if (mIsGpu && mDataDirtyFlags & SceneDataDirtyFlags::eParticleVelocity)
            CR_CHECK_CUDA(cudaMemcpyAsync(mGpuParticleData.velocity, mCpuParticleData.velocity, mMaxNumParticles * sizeof(Vec3f), cudaMemcpyKind::cudaMemcpyHostToDevice, mCudaStream));

        if (mIsGpu && mDataDirtyFlags & SceneDataDirtyFlags::eParticleMaterialParams0)
            CR_CHECK_CUDA(cudaMemcpyAsync(mGpuParticleMaterialData.params0, mCpuParticleMaterialData.params0, mMaxNumParticles * sizeof(float4), cudaMemcpyKind::cudaMemcpyHostToDevice, mCudaStream));

        if (mIsGpu && mDataDirtyFlags & SceneDataDirtyFlags::eParticleMaterialType)
            CR_CHECK_CUDA(cudaMemcpyAsync(mGpuParticleMaterialData.type, mCpuParticleMaterialData.type, mMaxNumParticles * sizeof(ParticleMaterialType), cudaMemcpyKind::cudaMemcpyHostToDevice, mCudaStream));

        if (mDataDirtyFlags & SceneDataDirtyFlags::eComputeInitialData)
        {
            if (mComputeInitialDataIndices.size() > mMaxNumParticles)
            {
                CR_DEBUG_LOG_WARNING("%s", "Compute initial data particle num > max particle num!");
            }
            if (mIsGpu)
            {
                CR_CHECK_CUDA(cudaMemcpyAsync(dmComputeInitialDataIndices, mComputeInitialDataIndices.begin(), mComputeInitialDataIndices.size() * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice, mCudaStream));
                mSolver->computeInitialData(mComputeInitialDataIndices.size(),
                                            dmComputeInitialDataIndices);
            }
            else
            {
                mSolver->computeInitialData(mComputeInitialDataIndices.size(),
                                            mComputeInitialDataIndices.begin());
            }
            mComputeInitialDataIndices.clear();
        }

        mDataDirtyFlags = SceneDataDirtyFlags::eNone;
    }

    void SceneImpl::setSolver(MpmSolverBase &solver, bool isGpu)
    {
        mIsGpu = isGpu;
        mSolver = &solver;
        mSolver->setNumShapes(0);

        if (mIsGpu)
        {
            // Always use the same cuda stream as the solver
            mCudaStream = static_cast<MpmSolverGpu &>(solver).getCudaStream();
        }
    }

    void SceneImpl::beforeAdvance()
    {
        // Set all shape data to be zero
        for (ShapeImpl *shape : mShapes)
        {
            shape->resetCouplingForce();
        }

        syncDataIfNeeded();
    }

    void SceneImpl::afterAdvance()
    {
        // TODO: this can be moved to factory if we share a big memory block for all scenes
        if (mIsGpu)
        {
            // GPU->CPU
            CR_CHECK_CUDA(cudaMemcpyAsync(mCpuParticleData.positionMass, dmParticlePositionMass, mMaxNumParticles * sizeof(float4), cudaMemcpyKind::cudaMemcpyDeviceToHost, mCudaStream));
            CR_CHECK_CUDA(cudaMemcpyAsync(mCpuParticleData.velocity, dmParticleVelocity, mMaxNumParticles * sizeof(Vec3f), cudaMemcpyKind::cudaMemcpyDeviceToHost, mCudaStream));
        }

        for (ShapeImpl *shape : mShapes)
        {
            shape->resetKinematicTarget();
        }
    }

    SimulationStatus SceneImpl::getSimulationStatus()
    {
        return mStatus;
    }

    void SceneImpl::setSimulationStatus(SimulationStatus status)
    {
        mStatus = status;
    }

    void SceneImpl::_release()
    {
        CR_DEBUG_LOG_INFO("%s", "Releasing Scene.");
        // In case solver still running
        if (mStatus == SimulationStatus::eBusy)
            fetchResults(); // This also works if advanceAll() was called
        
        // Release all referred shapes
        for (ShapeImpl *shape : mShapes)
        {
            shape->release();
        }
        mShapes.clear();

        mSolver->release();
        delete[] mParticleMaterialTypes;
        delete[] mParticleMaterialPropertiesGroup1;
        delete[] mActiveParticleMask;

        mFactory->releaseScene(this);

        if (mIsGpu)
        {
            CR_CHECK_CUDA(cudaFreeAsync(dmParticlePositionMass, mCudaStream));
            CR_CHECK_CUDA(cudaFreeAsync(dmParticleVelocity, mCudaStream));
            CR_CHECK_CUDA(cudaFreeAsync(dmParticleMaterialPropertiesGroup1, mCudaStream));
            CR_CHECK_CUDA(cudaFreeAsync(dmParticleMaterialTypes, mCudaStream));
            CR_CHECK_CUDA(cudaFreeAsync(dmShapeIds, mCudaStream));
            CR_CHECK_CUDA(cudaFreeAsync(dmActiveParticleMask, mCudaStream));
            CR_CHECK_CUDA(cudaFreeAsync(dmComputeInitialDataIndices, mCudaStream));
        }
    }

    void SceneImpl::_advanceTaskFn(void *data)
    {
        beforeAdvance();
        float dt = *((float *)data);
        mFactory->beforeSceneAdvance(); // Note: a second scene advance will not trigger copy again.
        float timeAdvanced = 0;
        while (timeAdvanced < dt)
        {
            timeAdvanced += mSolver->step();
        }

        // We don't need to call mSolver->fetchResults() here

        afterAdvance();

        if (mIsGpu)
        { 
            // Final stream synchronization if using GPU
            CR_CHECK_CUDA(cudaStreamSynchronize(mCudaStream));
        }

        mFactory->afterSceneAdvance(mIsGpu);
    }

} // namespace crmpm
