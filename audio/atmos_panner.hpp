#pragma once

#include "cavern_object_stream.hpp"
#include "spatializer.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace MagicAapSpatial
{
enum class ObjectLockMode
{
    WorldLocked,
    HeadLocked,
    Diffuse
};

struct AtmosPannerSettings
{
    ObjectLockMode dynamicObjects = ObjectLockMode::WorldLocked;
    ObjectLockMode bed = ObjectLockMode::Diffuse;
    float objectWidth = 1.0f;
    float diffuseBedLevel = 0.35f;
};

class AtmosPanner final
{
public:
    explicit AtmosPanner(AtmosPannerSettings settings = {});
    void SetSettings(AtmosPannerSettings settings);
    void SetHrtfProfile(std::shared_ptr<const HrtfProfile> profile);
    void SetHrtfBlend(float blend);
    void Reset();

    void Process(
        const CavernObjectBlock& block,
        float listenerYawDegrees,
        float listenerPitchDegrees,
        float listenerRollDegrees,
        float* outputLeft,
        float* outputRight,
        std::size_t outputCapacity);

    void Process(
        const CavernObjectBlock& block,
        float listenerYawDegrees,
        float listenerPitchDegrees,
        float* outputLeft,
        float* outputRight,
        std::size_t outputCapacity)
    {
        Process(block, listenerYawDegrees, listenerPitchDegrees, 0.0f, outputLeft, outputRight, outputCapacity);
    }

private:
    AtmosPannerSettings settings_;
    std::shared_ptr<const HrtfProfile> hrtfProfile_;
    float hrtfBlend_ = 0.85f;
    std::vector<float> tailLeft_;
    std::vector<float> tailRight_;
};
}
