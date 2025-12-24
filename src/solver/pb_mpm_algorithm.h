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

#ifndef CR_PB_MPM_ALGORITHM_H
#define CR_PB_MPM_ALGORITHM_H

#include "preprocessor.h"
#include "vec3.h"
#include "mat33.h"
#include "material_data.h"
#include "constitutive_models.h"
#include "shape_functions.h"
#include "mpm_rigid_contact.h"

namespace crmpm
{
    /*-------------------------*/
    /// PB-MPM
    /*-------------------------*/

    template <bool IsCuda>
    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void pbMpmComputeInitialGridMass(
        const Vec3f gridBoundMin,
        const float invCellSize,
        const Vec3i numNodesPerDim,
        const float gridVolume,
        const float4 &particlePositionMass,
        float4 *CR_RESTRICT nodeMomentumVelocityMass)
    {
        // Scatter particle mass to nodes
        Vec3f scaledPos = (particlePositionMass - gridBoundMin) * invCellSize;
        Vec3i baseCoord = (scaledPos - Vec3f(0.5f)).cast<int>();
        Vec3f fx = scaledPos - baseCoord.cast<float>();

        Mat3x3f weights;
        computeQuadraticBSplineWeights(fx, weights);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Vec3i nodeCoord = baseCoord + make_int4(i, j, k, 0);
                    if (nodeCoord.x() < 0 || nodeCoord.x() >= numNodesPerDim.x() ||
                        nodeCoord.y() < 0 || nodeCoord.y() >= numNodesPerDim.y() ||
                        nodeCoord.z() < 0 || nodeCoord.z() >= numNodesPerDim.z())
                    {
                        continue;
                    }
                    int nodeIdx = nodeCoord.x() + nodeCoord.y() * numNodesPerDim.x() + nodeCoord.z() * numNodesPerDim.x() * numNodesPerDim.y();

                    float weight = weights(i, 0) * weights(j, 1) * weights(k, 2);

                    // Scatter mass to grid
                    float weighted_mass = weight * particlePositionMass.w;
                    if constexpr (IsCuda)
                    {
                        atomicAdd(&nodeMomentumVelocityMass[nodeIdx].w, weighted_mass);
                    }
                    else
                    {
                        nodeMomentumVelocityMass[nodeIdx].w += weighted_mass;
                    }
                }
            }
        }
    }

    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void pbMpmComputeInitialVolume(
        const Vec3f gridBoundMin,
        const float invCellSize,
        const Vec3i numNodesPerDim,
        const float gridVolume,
        const float4 &particlePositionMass,
        const float4 *CR_RESTRICT nodeMomentumVelocityMass,
        float &particleInitialVolume)
    {
        Vec3f scaledPos = (particlePositionMass - gridBoundMin) * invCellSize;
        Vec3i baseCoord = (scaledPos - Vec3f(0.5f)).cast<int>();
        Vec3f fx = scaledPos - baseCoord.cast<float>();

        Mat3x3f weights;
        computeQuadraticBSplineWeights(fx, weights);

        float density = 0.0f;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Vec3i nodeCoord = baseCoord + Vec3i{i, j, k};

                    if (nodeCoord.x() < 0 || nodeCoord.x() >= numNodesPerDim.x() ||
                        nodeCoord.y() < 0 || nodeCoord.y() >= numNodesPerDim.y() ||
                        nodeCoord.z() < 0 || nodeCoord.z() >= numNodesPerDim.z())
                    {
                        continue;
                    }

                    int nodeIdx = nodeCoord.x() + nodeCoord.y() * numNodesPerDim.x() + nodeCoord.z() * numNodesPerDim.x() * numNodesPerDim.y();
                    float weight = weights(i, 0) * weights(j, 1) * weights(k, 2);

                    float nodeDensity = nodeMomentumVelocityMass[nodeIdx].w / gridVolume;
                    density += nodeDensity * weight;
                }
            }
        }
        // printf("GPU Density: %f\n", density);

        if (density > 1e-8f)
        {
            particleInitialVolume = particlePositionMass.w / density;
        }
        else
        {
            particleInitialVolume = 1e-8f;
        }
    }

    template <bool IsCuda>
    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void pbMpmParticleToGrid(
        const float integrationStepSize,
        const Vec3f &gridBoundMin,
        const float invCellSize,
        const float cellSize,
        const Vec3i &numNodesPerDim,
        const float4 &particlePositionMass,
        const Vec3f &particleVelocity,
        const Mat3x3f &gradientDeformationTensor,
        const float4 &particleMaterialProperties0,
        const ParticleMaterialType &particleMaterialType,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData &shapeData,
        const GeometryData &geometryData,
        const GeometrySdfData &geometrySdfData,
        Mat3x3f &particleAffineMomentum,
        float4 *CR_RESTRICT nodeMomentumVelocityMass)
    {
        // Material constraints
        solveMaterialConstraints(integrationStepSize, particleMaterialProperties0, particleMaterialType, gradientDeformationTensor, particleAffineMomentum);

        // Weights
        Mat3x3f weights;
        const Vec3f scaledPos = ( particlePositionMass - gridBoundMin) * invCellSize;
        const Vec3i baseCoord = ( scaledPos - Vec3f(0.5f)).cast<int>(); // Lower-left
        const Vec3f fx = scaledPos - baseCoord.cast<float>();
        computeQuadraticBSplineWeights(fx, weights);

        const float particleMass = particlePositionMass.w;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Vec3i nodeCoord = baseCoord + Vec3i(i, j, k);
                    if (nodeCoord.x() < 0 || nodeCoord.x() >= numNodesPerDim.x() ||
                        nodeCoord.y() < 0 || nodeCoord.y() >= numNodesPerDim.y() ||
                        nodeCoord.z() < 0 || nodeCoord.z() >= numNodesPerDim.z())
                    {
                        continue;
                    }

                    const Vec3f nodeParticleDistance = (Vec3f(i, j, k) - fx) * cellSize;

                    // For slicer geometries, do not transfer velocity
                    if (checkOnDifferentSides<false>(
                            particlePositionMass,
                            (particlePositionMass + nodeParticleDistance).data,
                            numShapes,
                            shapeIds,
                            shapeData,
                            geometryData,
                            geometrySdfData))
                    {
                        continue;
                    }

                    const int nodeIdx = nodeCoord.x() + nodeCoord.y() * numNodesPerDim.x() + nodeCoord.z() * numNodesPerDim.x() * numNodesPerDim.y();

                    const float weight = weights(i, 0) * weights(j, 1) * weights(k, 2);

                    // Scatter mass
                    const float weightedMass = weight * particleMass;

                    if constexpr (IsCuda)
                    {
                        atomicAdd(&nodeMomentumVelocityMass[nodeIdx].w, weightedMass);
                        atomicAddVec3f(&nodeMomentumVelocityMass[nodeIdx], (particleVelocity + particleAffineMomentum * nodeParticleDistance) * weightedMass);
                    }
                    else
                    {
                        nodeMomentumVelocityMass[nodeIdx].w += weightedMass;
                        nodeMomentumVelocityMass[nodeIdx] += (particleVelocity + particleAffineMomentum * nodeParticleDistance) * weightedMass;
                    }
                }
            }
        }
    }

    template <bool IsCuda, bool UseEffectiveMass>
    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void pbMpmUpdateGrid(
        const float integrationStepSize,
        const int nodeIdx,
        const Vec3f gridBoundMin,
        const float cellSize,
        const Vec3i numNodesPerDim,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData &shapeData,
        const GeometryData &geometryData,
        const GeometrySdfData &geometrySdfData,
        float4 *CR_RESTRICT nodeMomentumVelocityMass)
    {
        int numNodesX = numNodesPerDim.x();
        int numNodesY = numNodesPerDim.y();

        int i = nodeIdx % numNodesX;
        int j = (nodeIdx / numNodesX) % numNodesY;
        int k = nodeIdx / (numNodesX * numNodesY);

        // Small mass cutoff
        if (nodeMomentumVelocityMass[nodeIdx].w < 1e-8f)
            return;

        // Convert to velocity
        nodeMomentumVelocityMass[nodeIdx].x /= nodeMomentumVelocityMass[nodeIdx].w;
        nodeMomentumVelocityMass[nodeIdx].y /= nodeMomentumVelocityMass[nodeIdx].w;
        nodeMomentumVelocityMass[nodeIdx].z /= nodeMomentumVelocityMass[nodeIdx].w;

        // Collision with rigid bodies
        // TODO: this can use shared memory
        Vec3f nodePosition = Vec3f(i * cellSize, j * cellSize, k * cellSize) + gridBoundMin;
        nodePosition += nodeMomentumVelocityMass[nodeIdx] * integrationStepSize;
        resolveRigidCollision<false, IsCuda, UseEffectiveMass>(
            nodeMomentumVelocityMass[nodeIdx],
            nodePosition.data,
            numShapes,
            shapeIds,
            shapeData,
            geometryData,
            geometrySdfData);

        // Boundary conditions
        int numNodesZ = numNodesPerDim.z();
        if (i < 2 || i > numNodesX - 3)
            nodeMomentumVelocityMass[nodeIdx].x = 0;
        if (j < 2 || j > numNodesY - 3)
            nodeMomentumVelocityMass[nodeIdx].y = 0;
        if (k < 2 || k > numNodesZ - 3)
            nodeMomentumVelocityMass[nodeIdx].z = 0;
    }

    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void pbMpmGridToParticle(
        const Vec3f gridBoundMin,
        const Vec3f gridBoundMax,
        const float invCellSize,
        const float cellSize,
        const float mCellVolume,
        const Vec3i numNodesPerDim,
        const float integrationStepSize,
        const float4 *CR_RESTRICT nodeMomentumVelocityMass,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData &shapeData,
        const GeometryData &geometryData,
        const GeometrySdfData &geometrySdfData,
        float4 &particlePositionMass,
        Vec3f &particleVelocity,
        Mat3x3f &particleAffineMomentum)
    {
        Vec3f scaledPos = (particlePositionMass - gridBoundMin) * invCellSize;
        Vec3i baseCoord = (scaledPos - Vec3f(0.5f)).cast<int>();
        Vec3f fx = scaledPos - baseCoord.cast<float>();

        Mat3x3f weights;
        computeQuadraticBSplineWeights(fx, weights);

        Vec3f particleVelocityCumulation;
        Mat3x3f affineMomentumCumulation;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Vec3i nodeCoord = baseCoord + Vec3i(i, j, k);
                    if (nodeCoord.x() < 0 || nodeCoord.x() >= numNodesPerDim.x() ||
                        nodeCoord.y() < 0 || nodeCoord.y() >= numNodesPerDim.y() ||
                        nodeCoord.z() < 0 || nodeCoord.z() >= numNodesPerDim.z())
                    {
                        continue;
                    }

                    int nodeIdx = nodeCoord.x() + nodeCoord.y() * numNodesPerDim.x() + nodeCoord.z() * numNodesPerDim.x() * numNodesPerDim.y();
                    Vec3f nodeVel = nodeMomentumVelocityMass[nodeIdx];
                    Vec3f nodeParticleDistance = (Vec3f(i, j, k) - fx) * cellSize;

                    // For slicer geometries, do not transfer velocity
                    Vec3f normal;
                    if (checkOnDifferentSides<true>(
                            particlePositionMass,
                            (particlePositionMass + nodeParticleDistance).data,
                            numShapes,
                            shapeIds,
                            shapeData,
                            geometryData,
                            geometrySdfData,
                            &normal))
                    {
                        nodeVel = particleVelocity - normal * particleVelocity.dot(normal) * 2.0f; // particle reflected velocity
                    }

                    float weight = weights(i, 0) * weights(j, 1) * weights(k, 2);

                    // Update particle velocity in the PIC way
                    particleVelocityCumulation += nodeVel * weight;

                    // Update affine momentum
                    affineMomentumCumulation += Vec3f(nodeVel * weight).outerProduct(nodeParticleDistance);
                }
            }
        }
        // Update velocity
        particleVelocity = particleVelocityCumulation;

        // Update APIC affine momentum
        particleAffineMomentum = 4 * invCellSize * affineMomentumCumulation;
    }

    template <bool IsCuda>
    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void pbMpmIntegrateParticle(
        const Vec3f gridBoundMin,
        const Vec3f gridBoundMax,
        const float integrationStepSize,
        const Vec3f &gravity,
        const Mat3x3f &particleAffineMomentum,
        const ParticleMaterialType &particleMaterialType,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData &shapeData,
        const GeometryData &geometryData,
        const GeometrySdfData &geometrySdfData,
        float4 &particlePositionMass,
        Vec3f &particleVelocity,
        Mat3x3f &gradientDeformationTensor)
    {
        // Position integration (symplectic)
        particlePositionMass += particleVelocity * integrationStepSize;
        clamp(particlePositionMass, gridBoundMin, gridBoundMax);

        particleVelocity += gravity * integrationStepSize; // gravity

        // Particle-level collision response, in case particles are inside a shape.
        // Most of the collision resolution has been done at grid level.
        // TODO: correcting position sometimes seem to do too much and particles can get deformed without generating force
        resolveRigidCollision<true, IsCuda, false>(
            particleVelocity.data,
            particlePositionMass,
            numShapes,
            shapeIds,
            shapeData,
            geometryData,
            geometrySdfData);

        // TODO: this clamping actually makes simulation less stable.
        // Mat3x3f u, vt, sigma;
        // svd3(gradientDeformationTensor, u, vt, sigma);
        // clamp(sigma.col0.x, 0.2f, 10000.0f);
        // clamp(sigma.col1.y, 0.2f, 10000.0f);
        // clamp(sigma.col2.z, 0.2f, 10000.0f);
        // gradientDeformationTensor = u * sigma * vt;

        // Update gradient deformation tensor F
        gradientDeformationTensor.leftMulInPlace(Mat3x3f::Identity() + particleAffineMomentum * integrationStepSize);
    }
} // namespace crmpm

#endif // !CR_PB_MPM_ALGORITHM_H