#pragma once

#include <filesystem>

namespace MagicAapSpatial
{
std::filesystem::path DecodeToTemporaryWav(
    const std::filesystem::path& sourcePath,
    bool channelBedMode = false);

std::filesystem::path DecodeChannelBedToTemporaryWav(const std::filesystem::path& sourcePath);
}
