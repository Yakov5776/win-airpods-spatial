#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

namespace MagicAapSpatial
{
enum class Mode
{
    Off,
    Fixed,
    Tracked
};

struct Pose
{
    float yawDegrees = 0.0f;
    float pitchDegrees = 0.0f;
    float rollDegrees = 0.0f;
};

class AtmosPanner;

class HrtfProfile final
{
public:
    static std::shared_ptr<const HrtfProfile> Load(
        const std::filesystem::path& path,
        float sampleRate = 48000.0f);

    const std::filesystem::path& Path() const noexcept;
    bool IsBrir() const noexcept;
    int FilterLength() const noexcept;

    bool Evaluate(
        float x, float y, float z,
        float* leftFilter, float* rightFilter,
        std::size_t maxTaps,
        std::size_t& outTapCount) const;

private:
    struct Impl;
    explicit HrtfProfile(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;

    friend class Spatializer;
    friend class AtmosPanner;
};

class Spatializer final
{
public:
    explicit Spatializer(float sampleRate = 48000.0f);
    ~Spatializer();

    Spatializer(const Spatializer&) = delete;
    Spatializer& operator=(const Spatializer&) = delete;

    void SetMode(Mode mode);
    void SetPose(Pose pose);
    void SetStageWidth(float degrees);
    void SetRoomReflection(float amount);
    void SetHrtfBlend(float amount);
    void SetHrtfProfile(std::shared_ptr<const HrtfProfile> profile);
    void Reset();
    bool IsHrtfReady() const;

    void Process(
        const float* inputLeft,
        const float* inputRight,
        float* outputLeft,
        float* outputRight,
        std::size_t frameCount);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}