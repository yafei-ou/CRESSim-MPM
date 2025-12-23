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

#include "shape_impl.h"
#include "geometry_impl.h"
#include "simulation_factory_impl.h"

namespace crmpm
{
    void ShapeImpl::setGeometry(Geometry &geom)
    {
        // Decrease existing geometry's ref count
        if (mGeometry)
        {
            mGeometry->release();
        }

        // Update the linked geometry index in the factory data
        _geometryId() = geom.getId();

        mGeometry = &geom;
        static_cast<GeometryImpl &>(geom).incrementRef();

        // Mark factory GPU data dirty
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeGeometryIdx);
    }

    void ShapeImpl::setPose(const Transform &transform)
    {
        _position() = transform.position;
        _rotation() = transform.rotation;

        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapePosition);
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeRotation);
    }

    void ShapeImpl::setVelocity(const Vec3f &linear, const Vec3f &angular)
    {
        _linearVelocity() = linear;
        _angularVelocity() = angular;

        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeLinearVelocity);
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeAngularVelocity);
    }

    void ShapeImpl::setKinematicTarget(const Transform &transform, const float dt)
    {
        if (getType() == ShapeType::eStatic)
            return;

        // Linear increment
        _linearVelocity() = (transform.position - _position()) / dt;

        // Angular increment
        Quat qCurrent = _rotation();
        Quat qTarget = transform.rotation;

        Quat qDelta = qTarget * qCurrent.getConjugate(); // Relative rotation

        Vec3f angularVelocity;
        float angle;
        Vec3f axis;
        qDelta.toRadiansAndUnitAxis(angle, axis);
        _angularVelocity() = (axis * angle) / dt;

        _position() = transform.position;
        _rotation() = transform.rotation;

        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapePosition);
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeRotation);
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeLinearVelocity);
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeAngularVelocity);
    }

    void ShapeImpl::resetKinematicTarget()
    {
        _linearVelocity() = Vec3f();
        _angularVelocity() = Vec3f();
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeLinearVelocity);
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeAngularVelocity);
    }

    void ShapeImpl::resetCouplingForce()
    {
        _force() = make_float4(0, 0, 0, 0);
        _torque() = make_float4(0, 0, 0, 0);
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeForce);
        mFactory->markDirty(SimulationFactoryGpuDataDirtyFlags::eShapeTorque);
    }

    void ShapeImpl::_release()
    {
        CR_DEBUG_LOG_INFO("%s", "Releasing Shape.");
        if (mGeometry)
        {
            mGeometry->release();
            mGeometry = nullptr;
        }
        mFactory->releaseShape(this);
    }
} // namespace crmpm
