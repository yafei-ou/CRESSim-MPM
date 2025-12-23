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

#ifndef CR_SHAPE_IMPL_H
#define CR_SHAPE_IMPL_H

#include "shape.h"
#include "factory_controlled.h"
#include "ref_counted.h"

namespace crmpm
{
    class SimulationFactoryImpl;

    /**
     * Rigid body shape. For now, it is either static or kinematic.
     */
    class ShapeImpl : public Shape, public FactoryControlled, public RefCounted
    {
    public:
        void setGeometry(Geometry &geom) override;

        CR_FORCE_INLINE Geometry *getGeometry() override
        {
            return mGeometry;
        }

        CR_FORCE_INLINE int getId() const override { return mId; }

        CR_FORCE_INLINE ShapeType getType() override { return mShapeData->type[mId]; }

        void setPose(const Transform &transform) override;

        void setVelocity(const Vec3f &linear, const Vec3f &angular) override;

        void setKinematicTarget(const Transform &transform, const float dt) override;

        CR_FORCE_INLINE void setScale(const Vec3f &scale) override
        {
            _invScale() = 1.0f / scale;
            markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeScale);
        }

        CR_FORCE_INLINE const float4 &getShapeForce() const override { return mShapeData->force[mId]; }

        CR_FORCE_INLINE const float4 &getShapeTorque() const override { return mShapeData->torque[mId]; }

        CR_FORCE_INLINE void setId(int id) { mId = id; }

        CR_FORCE_INLINE void bindShapeData(ShapeData &shapeData) { mShapeData = &shapeData; }

        void resetKinematicTarget();

        void resetCouplingForce();

    protected:
        ShapeImpl() {};

        ~ShapeImpl() {};

    private:
        int mId;

        // Linked data in factory
        ShapeData *mShapeData = nullptr;

        CR_FORCE_INLINE int &_geometryId() { return mShapeData->geometryIdx[mId]; }
        CR_FORCE_INLINE ShapeType &_type() { return mShapeData->type[mId]; }
        CR_FORCE_INLINE Vec3f &_position() { return mShapeData->position[mId]; }
        CR_FORCE_INLINE Quat &_rotation() { return mShapeData->rotation[mId]; }
        CR_FORCE_INLINE Vec3f &_linearVelocity() { return mShapeData->linearVelocity[mId]; }
        CR_FORCE_INLINE Vec3f &_angularVelocity() { return mShapeData->angularVelocity[mId]; }
        CR_FORCE_INLINE Vec3f &_invScale() { return mShapeData->invScale[mId]; }
        CR_FORCE_INLINE float4 &_force() { return mShapeData->force[mId]; }
        CR_FORCE_INLINE float4 &_torque() { return mShapeData->torque[mId]; }

        Geometry *mGeometry = nullptr; // Address to linked geometry.

        void _release() override;

        friend class SimulationFactoryImpl;
    };

} // namespace crmpm

#endif // !CR_SHAPE_IMPL_H