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

#ifndef CR_CONSTITUTIVE_MODELS_H
#define CR_CONSTITUTIVE_MODELS_H

#include "cuda_compat.h"
#include "preprocessor.h"
#include "mat33.h"
#include "material_data.h"
#include "svd3.h"

#include "debug_logger.h"

namespace crmpm
{
    /*-------------------------*/
    /// Standard and MLS MPM
    /*-------------------------*/

    CR_CUDA_HOST CR_CUDA_DEVICE CR_FORCE_INLINE void computeLinearElasticStress(
        const Mat3x3f &gradientDeformationTensor,
        const float &lambda,  // Lame param lambda
        const float &mu, // Lame param mu
        Mat3x3f &particleStressTensor)
    {
        const Mat3x3f strain = 0.5f * (gradientDeformationTensor.transpose() * gradientDeformationTensor - Mat3x3f::Identity()); // Green-Lagrange strain tensor
        particleStressTensor = lambda * Mat3x3f::Scalar(strain.trace()) + 2 * mu * strain; // 2nd PK stress

        // Convert to Kirchhoff stress
        particleStressTensor = gradientDeformationTensor * particleStressTensor * gradientDeformationTensor.transpose();
    }

    CR_CUDA_HOST CR_CUDA_DEVICE CR_FORCE_INLINE void computeNeoHookeanStress(
        const Mat3x3f &gradientDeformationTensor,
        const float &lambda,
        const float &mu,
        Mat3x3f &particleStressTensor)
    {
        float J = gradientDeformationTensor.determinant();
        if (J < 1e-6)
            J = 1e-6;
        const Mat3x3f b = gradientDeformationTensor * gradientDeformationTensor.transpose(); // Left Cauchy-Green tensor

        // Convert to Kirchhoff stress
        particleStressTensor = mu * (b - Mat3x3f::Identity()) + Mat3x3f::Scalar(lambda * log(J)); // Convert to Kirchhoff stress
    }

    CR_CUDA_HOST CR_CUDA_DEVICE CR_FORCE_INLINE void computeCoRotationalStress(
        const Mat3x3f &gradientDeformationTensor,
        const float &lambda,
        const float &mu,
        Mat3x3f &particleStressTensor)
    {
        Mat3x3f u, vt, sigma;
        svd3(gradientDeformationTensor, u, vt, sigma);

        float J = gradientDeformationTensor.determinant();
        if (J < 1e-6)
            J = 1e-6;

        // Kirchhoff stress.
        // See A. Stomakhin et al., Energetically Consistent Invertible Elasticity.
        particleStressTensor = 2 * mu * (gradientDeformationTensor - u * vt) * gradientDeformationTensor.transpose() + Mat3x3f::Scalar(lambda * J * (J - 1));
    }

    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void computeStress(
        const float4 &particleMaterialProperties0,
        const ParticleMaterialType &particleMaterialType,
        const Mat3x3f &gradientDeformationTensor,
        const Mat3x3f &gradientVelocity,
        const float currentVolume,
        const float initialVolume,
        Mat3x3f &stress)
    {
        switch (particleMaterialType)
        {
        case ParticleMaterialType::eNeoHookean:
            computeNeoHookeanStress(gradientDeformationTensor, particleMaterialProperties0.x, particleMaterialProperties0.y, stress);
            break;
        case ParticleMaterialType::eCoRotational:
            computeCoRotationalStress(gradientDeformationTensor, particleMaterialProperties0.x, particleMaterialProperties0.y, stress);
        case ParticleMaterialType::eLinearElastic:
            computeLinearElasticStress(gradientDeformationTensor, particleMaterialProperties0.x, particleMaterialProperties0.y, stress);
            break;

        default:
            break;
        }
    }

    /*-------------------------*/
    /// PB-MPM
    /*-------------------------*/
    CR_FORCE_INLINE CR_CUDA_HOST CR_CUDA_DEVICE void solveMaterialConstraints(
        const float integrationStepSize,
        const float4 &particleMaterialProperties0,
        const ParticleMaterialType &particleMaterialType,
        const Mat3x3f &gradientDeformationTensor,
        Mat3x3f &particleAffineMomentum)
    {
        switch (particleMaterialType)
        {
        // PB-MPM does not consider physical properties
        case ParticleMaterialType::eNeoHookean:
        case ParticleMaterialType::eCoRotational:
        {
            Mat3x3f targetF = (Mat3x3f::Identity() + particleAffineMomentum * integrationStepSize) * gradientDeformationTensor;
            Mat3x3f u, vt, sigma;
            svd3(targetF, u, vt, sigma);

            float detF = targetF.determinant();
            float clampDetF = clamp(fabsf(detF), 0.1f, 1000.0f);
            Mat3x3f Q = (1.0f / (copysignf(1.0f, detF) * sqrtf(clampDetF))) * targetF;
            const float elasticityRatio = particleMaterialProperties0.y;
            targetF = elasticityRatio * (u * vt) + (1.0f - elasticityRatio) * Q;

            // Actual elasticRelaxation is scaled by integration step size.
            // Original: particleAffineMomentum += elasticRelaxation * ((targetF * gradientDeformationTensor.inverse() - Mat3x3f::Identity()) * invIntegrationStepSize - particleAffineMomentum);
            const float elasticRelaxation = particleMaterialProperties0.x;
            particleAffineMomentum += elasticRelaxation * ((targetF * gradientDeformationTensor.inverse() - Mat3x3f::Identity()) - particleAffineMomentum * integrationStepSize);
            break;
        }

        default:
            break;
        }
    }
}

#endif //!CR_CONSTITUTIVE_MODELS_H