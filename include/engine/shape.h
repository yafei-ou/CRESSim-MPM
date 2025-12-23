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

#ifndef CR_SHAPE_H
#define CR_SHAPE_H

#include "transform.h"
#include "releasable.h"
#include "engine_export.h"

namespace crmpm
{
    class Geometry;

    /**
     * @brief Shape type enum.
     *
     * If a Shape is set to ShapeType::eDynamic, contact force will be recorded
     */
    enum class ShapeType : int
    {
        eStatic,
        eKinematic,
        eDynamic,
    };

    /**
     * @brief Shape data SOA
     */
    struct ShapeData
    {
        ShapeType *type = nullptr;
        int *geometryIdx = nullptr; // Each shape is linked with a geometry data, but multiple shapes can link to the same geometry.
        Vec3f *position = nullptr;
        Quat *rotation = nullptr;
        Vec3f *invScale = nullptr; // Inverse scale

        // Rigid body velocity

        Vec3f *linearVelocity = nullptr;
        Vec3f *angularVelocity = nullptr;

        float4 *force;  // Output coupling force
        float4 *torque; // Output coupling torque

        /**
         * Shape params0
         *    - x: sdf smooth distance parameter
         *    - y: damping/drag coefficient
         *    - z: friction
         *    - w: fatten SDF
         */
        float4 *params0 = nullptr;

        int size;
    };

    /**
     * @brief Rigid body shape. For now, it is either static or kinematic.
     */
    class CRMPM_ENGINE_API Shape : virtual public Releasable
    {
    public:
        /**
         * @brief Set the Geometry of the Shape.
         */
        virtual void setGeometry(Geometry &geom) = 0;

        /**
         * @brief Get the Geometry of the Shape.
         */
        virtual Geometry *getGeometry() = 0;

        /**
         * @brief Get the global ID of the Shape.
         */
        virtual int getId() const = 0;

        /**
         * @brief Get the type of the shape.
         */
        virtual ShapeType getType() = 0;

        /**
         * Set (teleport) the shape to a certain pose.
         */
        virtual void setPose(const Transform &transform) = 0;

        /**
         * Set the velocity (both linear and angular) of the shape.
         * For dynamic shapes, `setPose` and `setVelocity` can be used together.
         */
        virtual void setVelocity(const Vec3f &linear, const Vec3f &angular) = 0;

        /**
         * @brief Set a kinematic target of the Shape.
         * 
         * This works for both kinematic.
         * Velocity will be automatically calculated based on the current pose
         * and the target.
         */
        virtual void setKinematicTarget(const Transform &transform, const float dt) = 0;

        /**
         * @brief Set the 3-dimensional scale of the shape.
         */
        virtual void setScale(const Vec3f &scale) = 0;

        /**
         * @brief Get the contact force of the shape during the last simulation step.
         */
        virtual const float4 &getShapeForce() const = 0;

        /**
         * Get the contact torque of the shape during the last simulation step.
         */
        virtual const float4 &getShapeTorque() const = 0;

    protected:
        Shape() {};

        virtual ~Shape() {};
    };

} // namespace crmpm

#endif // !CR_SHAPE_H