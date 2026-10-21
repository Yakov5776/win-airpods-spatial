#include "atmos_panner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace MagicAapSpatial
{
namespace
{
constexpr float kPi = 3.14159265358979323846f;
constexpr float kRadiansPerDegree = kPi / 180.0f;
constexpr float kMetersPerSofaRadius = 1.4f;
constexpr std::size_t kMaxConvolutionTaps = 2048;

float SoftLimit(float value)
{
    const float magnitude = std::abs(value);
    if (magnitude <= 0.85f) return value;
    const float limited = 0.85f + 0.15f * std::tanh((magnitude - 0.85f) / 0.15f);
    return std::copysign(limited, value);
}
}

AtmosPanner::AtmosPanner(AtmosPannerSettings settings)
{
    SetSettings(settings);
}

void AtmosPanner::SetSettings(AtmosPannerSettings settings)
{
    settings.objectWidth = std::isfinite(settings.objectWidth)
        ? std::clamp(settings.objectWidth, 0.5f, 2.0f)
        : 1.0f;
    settings.diffuseBedLevel = std::isfinite(settings.diffuseBedLevel)
        ? std::clamp(settings.diffuseBedLevel, 0.0f, 0.7f)
        : 0.35f;
    settings_ = settings;
}

void AtmosPanner::SetHrtfProfile(std::shared_ptr<const HrtfProfile> profile)
{
    hrtfProfile_ = std::move(profile);
}

void AtmosPanner::SetHrtfBlend(float blend)
{
    hrtfBlend_ = std::isfinite(blend) ? std::clamp(blend, 0.0f, 1.0f) : 0.85f;
}

void AtmosPanner::Reset()
{
    std::fill(tailLeft_.begin(), tailLeft_.end(), 0.0f);
    std::fill(tailRight_.begin(), tailRight_.end(), 0.0f);
}

void AtmosPanner::Process(
    const CavernObjectBlock& block,
    float listenerYawDegrees,
    float listenerPitchDegrees,
    float* outputLeft,
    float* outputRight,
    std::size_t outputCapacity)
{
    if (outputLeft == nullptr || outputRight == nullptr ||
        outputCapacity < static_cast<std::size_t>(block.frameCount))
    {
        throw std::invalid_argument("Atmos panner output buffer is too small.");
    }
    if (block.frameCount <= 0) return;

    const auto frameCount = static_cast<std::size_t>(block.frameCount);
    const float yaw = std::isfinite(listenerYawDegrees) ? listenerYawDegrees : 0.0f;
    const float pitch = std::isfinite(listenerPitchDegrees) ? listenerPitchDegrees : 0.0f;
    const float pitchRadians = pitch * kRadiansPerDegree;
    const float cosPitch = std::cos(pitchRadians);
    const float sinPitch = std::sin(pitchRadians);

    const bool useHrtf = (hrtfProfile_ != nullptr && hrtfBlend_ > 0.001f);

    std::vector<float> pannedLeft;
    std::vector<float> pannedRight;
    if (!useHrtf || hrtfBlend_ < 0.999f)
    {
        pannedLeft.assign(frameCount, 0.0f);
        pannedRight.assign(frameCount, 0.0f);
        for (const auto& object : block.objects)
        {
            const auto lockMode = object.isDynamic ? settings_.dynamicObjects : settings_.bed;
            if (object.isLfe)
            {
                for (std::size_t frame = 0; frame < frameCount && frame < object.samples.size(); frame++)
                {
                    const float sample = object.samples[frame] * 0.20f;
                    pannedLeft[frame] += sample;
                    pannedRight[frame] += sample;
                }
                continue;
            }

            if (lockMode == ObjectLockMode::Diffuse)
            {
                for (std::size_t frame = 0; frame < frameCount && frame < object.samples.size(); frame++)
                {
                    const float sample = object.samples[frame] * settings_.diffuseBedLevel;
                    pannedLeft[frame] += sample;
                    pannedRight[frame] += sample;
                }
                continue;
            }

            const float x = object.x;
            const float y = object.y;
            const float z = object.z;
            const float relativeYaw = lockMode == ObjectLockMode::WorldLocked ? -yaw : 0.0f;
            const float yawRadians = relativeYaw * kRadiansPerDegree;
            const float rotatedX = (std::cos(yawRadians) * x + std::sin(yawRadians) * z) * settings_.objectWidth;
            const float rotatedZ = (-std::sin(yawRadians) * x + std::cos(yawRadians) * z);
            const float rotatedY = y * cosPitch - rotatedZ * sinPitch;
            const float depth = std::max(0.25f, std::sqrt(rotatedX * rotatedX + rotatedY * rotatedY + rotatedZ * rotatedZ));
            const float azimuth = std::atan2(rotatedX, std::max(0.001f, rotatedZ));
            const float side = std::sin(azimuth);
            const float elevation = std::clamp(rotatedY / depth, -1.0f, 1.0f);
            const float distanceGain = 1.0f / (1.0f + 0.12f * std::max(0.0f, depth - 1.0f));
            const float elevationGain = 1.0f - std::abs(elevation) * 0.06f;
            const float leftGain = std::sqrt(std::clamp(0.5f * (1.0f - side), 0.0f, 1.0f)) * distanceGain * elevationGain;
            const float rightGain = std::sqrt(std::clamp(0.5f * (1.0f + side), 0.0f, 1.0f)) * distanceGain * elevationGain;

            for (std::size_t frame = 0; frame < frameCount && frame < object.samples.size(); frame++)
            {
                const float sample = object.samples[frame];
                pannedLeft[frame] += sample * leftGain;
                pannedRight[frame] += sample * rightGain;
            }
        }
    }

    if (!useHrtf)
    {
        for (std::size_t frame = 0; frame < frameCount; frame++)
        {
            outputLeft[frame] = SoftLimit(pannedLeft[frame]);
            outputRight[frame] = SoftLimit(pannedRight[frame]);
        }
        return;
    }

    if (tailLeft_.size() < kMaxConvolutionTaps)
    {
        tailLeft_.assign(kMaxConvolutionTaps, 0.0f);
        tailRight_.assign(kMaxConvolutionTaps, 0.0f);
    }

    std::fill_n(outputLeft, frameCount, 0.0f);
    std::fill_n(outputRight, frameCount, 0.0f);

    const std::size_t tailFramesToAdd = std::min(frameCount, tailLeft_.size());
    for (std::size_t i = 0; i < tailFramesToAdd; ++i)
    {
        outputLeft[i] += tailLeft_[i];
        outputRight[i] += tailRight_[i];
    }
    if (frameCount < tailLeft_.size())
    {
        std::copy(tailLeft_.begin() + frameCount, tailLeft_.end(), tailLeft_.begin());
        std::fill(tailLeft_.end() - frameCount, tailLeft_.end(), 0.0f);
        std::copy(tailRight_.begin() + frameCount, tailRight_.end(), tailRight_.begin());
        std::fill(tailRight_.end() - frameCount, tailRight_.end(), 0.0f);
    }
    else
    {
        std::fill(tailLeft_.begin(), tailLeft_.end(), 0.0f);
        std::fill(tailRight_.begin(), tailRight_.end(), 0.0f);
    }

    std::array<float, kMaxConvolutionTaps> filterLeft{};
    std::array<float, kMaxConvolutionTaps> filterRight{};

    for (const auto& object : block.objects)
    {
        const auto lockMode = object.isDynamic ? settings_.dynamicObjects : settings_.bed;
        if (object.isLfe)
        {
            for (std::size_t frame = 0; frame < frameCount && frame < object.samples.size(); frame++)
            {
                const float sample = object.samples[frame] * 0.20f;
                outputLeft[frame] += sample;
                outputRight[frame] += sample;
            }
            continue;
        }

        if (lockMode == ObjectLockMode::Diffuse)
        {
            for (std::size_t frame = 0; frame < frameCount && frame < object.samples.size(); frame++)
            {
                const float sample = object.samples[frame] * settings_.diffuseBedLevel;
                outputLeft[frame] += sample;
                outputRight[frame] += sample;
            }
            continue;
        }

        float maxAmp = 0.0f;
        for (std::size_t frame = 0; frame < frameCount && frame < object.samples.size(); frame++)
        {
            maxAmp = std::max(maxAmp, std::abs(object.samples[frame]));
        }
        if (maxAmp < 1e-6f) continue;

        const float x = object.x;
        const float y = object.y;
        const float z = object.z;
        const float relativeYaw = lockMode == ObjectLockMode::WorldLocked ? -yaw : 0.0f;
        const float yawRadians = relativeYaw * kRadiansPerDegree;
        const float rotatedX = (std::cos(yawRadians) * x + std::sin(yawRadians) * z) * settings_.objectWidth;
        const float rotatedZ = (-std::sin(yawRadians) * x + std::cos(yawRadians) * z);
        const float rotatedY = y * cosPitch - rotatedZ * sinPitch;
        const float depth = std::max(0.25f, std::sqrt(rotatedX * rotatedX + rotatedY * rotatedY + rotatedZ * rotatedZ));
        const float distanceGain = 1.0f / (1.0f + 0.12f * std::max(0.0f, depth - 1.0f));

        float dirX = rotatedZ / depth;
        float dirY = -rotatedX / depth;
        float dirZ = rotatedY / depth;

        const float sofaPos[3] = {
            dirX * kMetersPerSofaRadius,
            dirY * kMetersPerSofaRadius,
            dirZ * kMetersPerSofaRadius
        };

        std::size_t tapCount = 0;
        if (!hrtfProfile_->Evaluate(
                sofaPos[0], sofaPos[1], sofaPos[2],
                filterLeft.data(), filterRight.data(),
                kMaxConvolutionTaps, tapCount) || tapCount == 0)
        {
            const float azimuth = std::atan2(rotatedX, std::max(0.001f, rotatedZ));
            const float side = std::sin(azimuth);
            const float elevation = std::clamp(rotatedY / depth, -1.0f, 1.0f);
            const float elevationGain = 1.0f - std::abs(elevation) * 0.06f;
            const float leftGain = std::sqrt(std::clamp(0.5f * (1.0f - side), 0.0f, 1.0f)) * distanceGain * elevationGain;
            const float rightGain = std::sqrt(std::clamp(0.5f * (1.0f + side), 0.0f, 1.0f)) * distanceGain * elevationGain;
            for (std::size_t frame = 0; frame < frameCount && frame < object.samples.size(); frame++)
            {
                outputLeft[frame] += object.samples[frame] * leftGain;
                outputRight[frame] += object.samples[frame] * rightGain;
            }
            continue;
        }

        for (std::size_t frame = 0; frame < frameCount && frame < object.samples.size(); frame++)
        {
            const float sample = object.samples[frame] * distanceGain;
            if (std::abs(sample) < 1e-7f) continue;

            for (std::size_t tap = 0; tap < tapCount; tap++)
            {
                const std::size_t dest = frame + tap;
                if (dest < frameCount)
                {
                    outputLeft[dest] += sample * filterLeft[tap];
                    outputRight[dest] += sample * filterRight[tap];
                }
                else
                {
                    const std::size_t tailIdx = dest - frameCount;
                    if (tailIdx < tailLeft_.size())
                    {
                        tailLeft_[tailIdx] += sample * filterLeft[tap];
                        tailRight_[tailIdx] += sample * filterRight[tap];
                    }
                }
            }
        }
    }

    if (hrtfBlend_ < 0.999f)
    {
        for (std::size_t frame = 0; frame < frameCount; frame++)
        {
            outputLeft[frame] = pannedLeft[frame] + hrtfBlend_ * (outputLeft[frame] - pannedLeft[frame]);
            outputRight[frame] = pannedRight[frame] + hrtfBlend_ * (outputRight[frame] - pannedRight[frame]);
        }
    }

    for (std::size_t frame = 0; frame < frameCount; frame++)
    {
        outputLeft[frame] = SoftLimit(outputLeft[frame]);
        outputRight[frame] = SoftLimit(outputRight[frame]);
    }
}
}
