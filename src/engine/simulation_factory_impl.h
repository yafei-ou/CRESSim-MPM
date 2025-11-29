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

#ifndef CR_SIMULATION_FACTORY_IMPL_H
#define CR_SIMULATION_FACTORY_IMPL_H

#include "type_aliases.h"
#include "simulation_factory.h"
#include "simulation_status.h"
#include "single_thread_manager.h"
#include "cpu_data_allocation_manager.h"

// Forward declare CUstream_st and cudaStream_t
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace crmpm
{
    class SceneImpl;
    class MpmSolverBase;
    class CpuDataAllocationManager;

    class SimulationFactoryImpl : public SimulationFactory
    {
    public:
        SimulationFactoryImpl(int particleCapacity,
                              int shapeCapacity,
                              int geometryCapacity,
                              int sdfDataCapacity = 10000,
                              bool isGpu = true,
                              int numCudaStreams = 1);
        ~SimulationFactoryImpl();

        bool isGpu() const override;

        void advanceAll(float dt) override;

        void fetchResultsAll() override;

        Scene *createScene(const SceneDesc &desc) override;

        Shape *createShape(Geometry &geom, const ShapeDesc &shapeDesc) override;

        // Overloads for creating geometries
        Geometry *createGeometry(GeometryType type, float4 params0) override;
        Geometry *createGeometry(GeometryType type, TriangleMesh &mesh, SdfDesc &sdfDesc) override;
        Geometry *createGeometry(GeometryType type, int numPoints, Vec3f *points, float fattenBounds = 0.5f) override;

        ParticleObject *createParticleObject(const ParticleObjectDesc &desc) override;

        void markDirty(SimulationFactoryGpuDataDirtyFlags flags) override;

        ParticleData &getParticleDataAll() override;

#ifdef _DEBUG
        void printGeometries();
        void printShapes();
#endif

    protected:
        void releaseScene(Scene *scene);
        void releaseShape(Shape *shape);
        void releaseGeometry(Geometry *geom);

        void resetDirtyFlags();
        void syncCpuToGpuIfNeeded();
        void beforeSceneAdvance();
        void afterSceneAdvance(bool isGpuScene);

        SimulationStatus getSimulationStatus();

    private:
        bool mIsGpu;

        Array<SceneImpl *> mSceneList;
        Array<MpmSolverBase *> mSolverList;

        int mParticleCapacity;
        ParticleData mCpuParticleData;
        float4 *mCpuParticlePositionMass;
        Vec3f *mCpuParticleVelocity;
        CpuDataAllocationManager mParticleCpuDataManager; // This manages both postitionMass and velocity data

        int mShapeCapacity;
        ShapeData mCpuShapeData;
        ShapeData mGpuShapeData;

        int mGeometryCapacity;
        GeometryData mCpuGeometryData;
        GeometryData mGpuGeometryData;

        GeometrySdfData mCpuSdfData;
        GeometrySdfData mGpuSdfData;
        unsigned int mSdfDataCapacity;

        SimulationFactoryGpuDataDirtyFlags mGpuDataDirtyFlags;

        Array<int> mShapeIndices;
        Array<int> mGeometryIndices;
        Array<int> mGeometrySdfIndices;

        struct FreeSdfDataBlock
        {
            long long offset;
            long long size;
        };
        Array<FreeSdfDataBlock> mFreeSdfDataBlock;

        SimulationStatus mStatus;
        int mBusySceneCount;
        SingleThreadManager mThreadManager;
        Task mAdvanceAllTaskFn;

        // CUDA related
        int mNumCudaStreams;
        Array<Pair<cudaStream_t, int>> mCudaStreams;

        void checkAndGrowGeometryData();
        size_t getSdfDataOffset(size_t requestedSize);
        int getNewSdfIndex();

        int getCudaStreamByLoad();

        void _advanceAllTaskFn(void *data);

        friend class FactoryControlledImpl;
        friend class SceneImpl;
        friend class ShapeImpl;
        friend class GeometryImpl;
    };
} // namespace crmpm

#endif // !CR_SIMULATION_FACTORY_IMPL_H