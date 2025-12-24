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

#ifndef CR_GPU_DATA_DIRTY_FLAGS_H
#define CR_GPU_DATA_DIRTY_FLAGS_H

#include "preprocessor.h"

#define CR_DEFINE_ENUM_BIT_OPERATORS(EnumType)                                       \
    constexpr EnumType operator|(EnumType lhs, EnumType rhs)                         \
    {                                                                                \
        return static_cast<EnumType>(static_cast<int>(lhs) | static_cast<int>(rhs)); \
    }                                                                                \
                                                                                     \
    constexpr EnumType &operator|=(EnumType &lhs, EnumType rhs)                      \
    {                                                                                \
        lhs = lhs | rhs;                                                             \
        return lhs;                                                                  \
    }                                                                                \
                                                                                     \
    constexpr bool operator&(EnumType lhs, EnumType rhs)                             \
    {                                                                                \
        return (static_cast<int>(lhs) & static_cast<int>(rhs)) != 0;                 \
    }

namespace crmpm
{
    enum class SimulationFactoryGpuDataDirtyFlags
    {
        eNone = 0,

        eShapeType = 1 << 0,
        eShapeGeometryIdx = 1 << 1,
        eShapePosition = 1 << 2,
        eShapeRotation = 1 << 3,
        eShapeLinearVelocity = 1 << 4,
        eShapeAngularVelocity = 1 << 5,
        eShapeComInvMass = 1 << 6,
        eShapeInertia0 = 1 << 7,
        eShapeInertia1 = 1 << 8,
        eShapeScale = 1 << 9,
        eShapeParams0 = 1 << 10,
        eShapeForce = 1 << 11,
        eShapeTorque = 1 << 12,

        eGeometryType = 1 << 13,
        eGeometryParams0 = 1 << 14,
        eGeometrySdfData = 1 << 15,

        eShape = eShapeType | eShapeGeometryIdx | eShapePosition | eShapeRotation | eShapeLinearVelocity |
                 eShapeAngularVelocity | eShapeComInvMass | eShapeInertia0 | eShapeInertia1 | eShapeParams0 |
                 eShapeScale | eShapeForce | eShapeTorque,
        eGeometry = eGeometryType | eGeometryParams0 | eGeometrySdfData,

        eAll = eShape | eGeometry,
    };

    enum class SceneDataDirtyFlags
    {
        eNone = 0,

        eParticlePositionMass = 1 << 0,
        eParticleVelocity = 1 << 1,
        eParticleMaterialParams0 = 1 << 2,
        eParticleMaterialType = 1 << 3,
        eNumParticles = 1 << 4,       // Change of allocated particle number
        eComputeInitialData = 1 << 5, // Compute initial particle data, e.g. volume

        eAll = eParticlePositionMass | eParticleVelocity | eParticleMaterialParams0 | eParticleMaterialType |
               eNumParticles | eComputeInitialData
    };

    CR_DEFINE_ENUM_BIT_OPERATORS(SimulationFactoryGpuDataDirtyFlags);
    CR_DEFINE_ENUM_BIT_OPERATORS(SceneDataDirtyFlags);

} // namespace crmpm

#endif // !CR_GPU_DATA_DIRTY_FLAGS_H