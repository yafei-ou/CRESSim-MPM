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

#ifndef CR_MPM_RIGID_CONTACT_H
#define CR_MPM_RIGID_CONTACT_H

#include "cuda_compat.h"
#include "preprocessor.h"
#include "vec3.h"
#include "quat.h"
#include "geometry.h"
#include "shape.h"
#include "sdf.h"
#include "type_punning.h"

namespace crmpm
{
    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE Vec3f computeVelocityAtRigidBody(
        const Vec3f &linearVelocity,
        const Vec3f &angularVelocity,
        const Vec3f &worldRelativePosition)
    {
        Vec3f angularContribution = angularVelocity.cross(worldRelativePosition);

        return linearVelocity + angularContribution;
    }

    /**
     * Accumulate the momentum on the rigid body. Torques are converted to relative to the origin.
     */
    template <bool IsCuda>
    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void addMomentumToRigidBody(
        const Vec3f &momentum,
        const Vec3f &localPosition,
        float4 &shapeForce,
        float4 &shapeTorque)
    {
        if constexpr (IsCuda)
        {
            atomicAdd(&shapeForce.x, momentum.x());
            atomicAdd(&shapeForce.y, momentum.y());
            atomicAdd(&shapeForce.z, momentum.z());

            Vec3f torque = localPosition.cross(momentum);
            atomicAdd(&shapeTorque.x, torque.x());
            atomicAdd(&shapeTorque.y, torque.y());
            atomicAdd(&shapeTorque.z, torque.z());
        }
        else
        {
            shapeForce += momentum;
            shapeTorque += localPosition.cross(momentum);
        }
    }

    /**
     * Modify the velocity if inputPosition is inside a shape.
     * If IsParticle = true, inputPosition will be pushed to the 
     * nearest shape boundary (used for particles)
     */
    template <bool IsParticle, bool IsCuda>
    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void resolveRigidCollision(
        float4 &inputVelocity,
        float4 &inputPosition,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData &shapeData,
        const GeometryData &geometryData,
        const GeometrySdfData &geometrySdfData)
    {
        const ShapeType *CR_RESTRICT shapeType = shapeData.type;
        const Vec3f *CR_RESTRICT shapePosition = shapeData.position;
        const Quat *CR_RESTRICT shapeRotation = shapeData.rotation;
        const Vec3f *CR_RESTRICT shapeLinearVelocity = shapeData.linearVelocity;
        const Vec3f *CR_RESTRICT shapeAngularVelocity = shapeData.angularVelocity;
        const Vec3f *CR_RESTRICT shapeInvScale = shapeData.invScale;
        const float4 *CR_RESTRICT shapeParams0 = shapeData.params0;
        const int *CR_RESTRICT geomIds = shapeData.geometryIdx;

        const GeometryType *CR_RESTRICT geomType = geometryData.type;
        const float4 *CR_RESTRICT geomParams0 = geometryData.params0;

        const Vec3i *CR_RESTRICT sdfDimension = geometrySdfData.dimemsion;
        const float4 *CR_RESTRICT sdfLowerBoundCellSize = geometrySdfData.lowerBoundCellSize;
        const float4 *CR_RESTRICT sdfGridData = geometrySdfData.gradientSignedDistance;

        for (int shapeIdx = 0; shapeIdx < numShapes; shapeIdx++)
        {
            // The shape ID in the scene
            const int _shapeId = shapeIds[shapeIdx];
            const ShapeType _shapeType = shapeType[_shapeId];
            const Vec3f _shapeInvScale = shapeInvScale[_shapeId];

            // The geometry ID linked to the shape
            const int _geomId = geomIds[_shapeId];

            // Geometry type
            GeometryType _geomType = geomType[_geomId];

            if constexpr (IsParticle)
            {
                if (_geomType == GeometryType::eConnectedLineSegments || _geomType == GeometryType::eArc)
                {
                    // We do not modify particles for line segments or arc
                    continue;
                }
            }

            // Local position
            const Vec3f _shapePosition = shapePosition[_shapeId];
            const Quat _shapeRotation = shapeRotation[_shapeId];
            Vec3f worldRelativePositionn = inputPosition - _shapePosition;
            Vec3f relativePosition = _shapeRotation.rotateInv(worldRelativePositionn);
            Vec3f localPosition = relativePosition * _shapeInvScale; // inverse scale to local position

            // Node velocity correction based on rigid contact
            Vec3f rigidVelocity = _shapeType == ShapeType::eStatic ? Vec3f() : computeVelocityAtRigidBody(shapeLinearVelocity[_shapeId], shapeAngularVelocity[_shapeId], worldRelativePositionn);

            float4 sdfGradientDistance;
            Vec3f tangentDirection;
            bool onSpine = getGeometrySdf(localPosition, _geomType, geomParams0[_geomId], sdfDimension, sdfLowerBoundCellSize, sdfGridData, sdfGradientDistance, &tangentDirection);
            Vec3f gradient = Vec3f(sdfGradientDistance);
            float distance = sdfGradientDistance.w;

            if (isnan(distance))
                continue;

            gradient *= _shapeInvScale; // scaled gradient
            const float invScale = gradient.invNorm();
            distance = distance * invScale;
            gradient = gradient * invScale;

            float friction;
            if (onSpine)
            {
                friction = 1e2; // Large friction to ensure sticking response for a slicer spine
            }
            else
            {
                friction = shapeParams0[_shapeId].z;
            }

            distance -= shapeParams0[_shapeId].w; // fatten SDF size

            float smoothDistance = 0;
            if constexpr (!IsParticle)
            {
                smoothDistance = shapeParams0[_shapeId].x;
            }

            if (distance < smoothDistance) // cutoff for further distances
            {
                Vec3f normal = _shapeRotation.rotate(gradient);

                Vec3f relativeVelocity = inputVelocity - rigidVelocity;
                float normalVelocity = relativeVelocity.dot(normal);;
                Vec3f normalComponent = normal * normalVelocity;
                
                if constexpr (IsParticle)
                {
                    // For particles, push out if inside the rigid shape.

                    // TODO: apart from the undesired (not restorable) deformation,
                    // this also causes a suddenly appearing shape to push all particles
                    // inside the shape to the boundary.
                    float inside = fminf(distance, 0.0f);
                    Vec3f correctedPosition = normal * inside;
                    inputPosition.x -= correctedPosition.x();
                    inputPosition.y -= correctedPosition.y();
                    inputPosition.z -= correctedPosition.z();
                }

                // Skip if they are separating
                if (normalVelocity >= 0)
                {
                    continue;
                }

                // Apply velocity correction
                Vec3f tangentialComponent = relativeVelocity - normalComponent;
                Vec3f collisionResponse;
                const float stickyScale = shapeParams0[_shapeId].y; // A damping coefficient

                // For line segments, we only allow motion along the line
                if (_geomType == GeometryType::eConnectedLineSegments || _geomType == GeometryType::eArc)
                {
                    tangentDirection *= _shapeInvScale;
                    tangentDirection.normalize();
                    tangentDirection = _shapeRotation.rotate(tangentDirection);
                    tangentialComponent = tangentDirection * tangentialComponent.dot(tangentDirection);
                }

                // v = v_t + mu * v_n * vt / norm(v_t)
                float tangentNorm = tangentialComponent.norm();
                float frictionCorrection = fmaxf(friction * normalVelocity / (tangentNorm + CR_EPS), -1);

                collisionResponse = tangentialComponent + tangentialComponent * frictionCorrection;
                collisionResponse *= stickyScale;
                collisionResponse += rigidVelocity;

                if constexpr (!IsParticle)
                {
                    if (_shapeType == ShapeType::eDynamic)
                    {
                        float4 *CR_RESTRICT shapeCouplingForce = shapeData.force;
                        float4 *CR_RESTRICT shapeCouplingTorque = shapeData.torque;
                        // Rigid coupling when set as dynamic
                        Vec3f momentumChange = inputVelocity - collisionResponse;
                        momentumChange *= inputVelocity.w; // At the grid level, input is velocity mass
                        addMomentumToRigidBody<IsCuda>(momentumChange, inputPosition - _shapePosition, shapeCouplingForce[_shapeId], shapeCouplingTorque[_shapeId]);
                    }
                }

                inputVelocity.x = collisionResponse.x();
                inputVelocity.y = collisionResponse.y();
                inputVelocity.z = collisionResponse.z();
            }
        }
    }

    template <bool ComputeNormal>
    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE bool checkOnDifferentSides(
        const float4 &p1,
        const float4 &p2,
        const int numShapes,
        const int *CR_RESTRICT shapeIds,
        const ShapeData &shapeData,
        const GeometryData &geometryData,
        const GeometrySdfData &geometrySdfData,
        Vec3f *normal = nullptr)
    {
        const Vec3f *CR_RESTRICT shapePosition = shapeData.position;
        const Quat *CR_RESTRICT shapeRotation = shapeData.rotation;
        const Vec3f *CR_RESTRICT shapeInvScale = shapeData.invScale;
        const int *CR_RESTRICT geomIds = shapeData.geometryIdx;

        const GeometryType *CR_RESTRICT geomType = geometryData.type;
        const float4 *CR_RESTRICT geomParams0 = geometryData.params0;

        const Vec3i *CR_RESTRICT sdfDimension = geometrySdfData.dimemsion;
        const float4 *CR_RESTRICT sdfLowerBoundCellSize = geometrySdfData.lowerBoundCellSize;
        const float4 *CR_RESTRICT sdfGridData = geometrySdfData.gradientSignedDistance;

        for (int shapeIdx = 0; shapeIdx < numShapes; shapeIdx++)
        {
            // The shape ID in the scene
            const int _shapeId = shapeIds[shapeIdx];
            const Vec3f _shapeInvScale = shapeInvScale[_shapeId];

            // The geometry ID linked to the shape
            const int _geomId = geomIds[_shapeId];

            // Geometry type
            GeometryType _geomType = geomType[_geomId];

            // Only check for slicer geometries
            if (_geomType != GeometryType::eQuadSlicer && _geomType != GeometryType::eTriangleMeshSlicer)
            {
                continue;
            }

            // Local positions
            const Vec3f _shapePosition = shapePosition[_shapeId];
            const Quat _shapeRotation = shapeRotation[_shapeId];

            Vec3f relativePosition1 = _shapeRotation.rotateInv(p1 - _shapePosition);
            relativePosition1 *= _shapeInvScale; // inverse scale to local position
            Vec3f relativePosition2 = _shapeRotation.rotateInv(p2 - _shapePosition);
            relativePosition2 *= _shapeInvScale;

            float4 sdfGradientDistance1;
            bool p1OnGeom = getGeometryModSdf(relativePosition1, _geomType, geomParams0[_geomId], sdfDimension, sdfLowerBoundCellSize, sdfGridData, sdfGradientDistance1);
            
            if (!p1OnGeom)
            {
                continue;
            }
            
            float4 sdfGradientDistance2;
            bool p2OnGeom = getGeometryModSdf(relativePosition2, _geomType, geomParams0[_geomId], sdfDimension, sdfLowerBoundCellSize, sdfGridData, sdfGradientDistance2);
            
            if (p2OnGeom && !(signbit(sdfGradientDistance1.w) == signbit(sdfGradientDistance2.w)))
            {
                if constexpr (ComputeNormal)
                {
                    Vec3f gradient = Vec3f(sdfGradientDistance2);

                    gradient *= _shapeInvScale; // scaled gradient
                    const float invScale = gradient.invNorm();
                    gradient = gradient * invScale;
                    *normal = gradient; // no safety check
                }
                return true;
            }
        }

        return false;
    }

} // namespace crmpm

#endif // !CR_MPM_RIGID_CONTACT_H