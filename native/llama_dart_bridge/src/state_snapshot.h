#ifndef LLAMA_DART_STATE_SNAPSHOT_H_
#define LLAMA_DART_STATE_SNAPSHOT_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace llama_dart_bridge_internal {

constexpr size_t kStateSnapshotHeaderSize = 64;
constexpr uint32_t kStateSnapshotVersion = 1;
constexpr uint32_t kStateSnapshotHasDraft = 1u << 0;
constexpr uint32_t kStateSnapshotHasSpeculativeState = 1u << 1;

struct state_snapshot_layout {
  uint32_t speculative_type = 0;
  uint64_t target_size = 0;
  uint64_t draft_size = 0;
  uint64_t speculative_size = 0;
  uint64_t token_count = 0;
  int64_t position = 0;
};

struct state_snapshot_view {
  state_snapshot_layout layout;
  const uint8_t *target = nullptr;
  const uint8_t *draft = nullptr;
  const uint8_t *speculative = nullptr;
  const uint8_t *tokens = nullptr;
};

enum class state_snapshot_decode_result {
  success,
  not_snapshot,
  invalid,
};

inline void write_u32_le(uint8_t *out, uint32_t value) {
  for (size_t i = 0; i < 4; ++i) {
    out[i] = static_cast<uint8_t>(value >> (i * 8));
  }
}

inline void write_u64_le(uint8_t *out, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    out[i] = static_cast<uint8_t>(value >> (i * 8));
  }
}

inline uint32_t read_u32_le(const uint8_t *data) {
  uint32_t value = 0;
  for (size_t i = 0; i < 4; ++i) {
    value |= static_cast<uint32_t>(data[i]) << (i * 8);
  }
  return value;
}

inline uint64_t read_u64_le(const uint8_t *data) {
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(data[i]) << (i * 8);
  }
  return value;
}

inline bool checked_size_add(size_t left, size_t right, size_t *out) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    return false;
  }
  *out = left + right;
  return true;
}

inline bool state_snapshot_size(const state_snapshot_layout &layout,
                                size_t *out_size) {
  if (layout.target_size == 0 || layout.position < 0 ||
      layout.target_size > std::numeric_limits<size_t>::max() ||
      layout.draft_size > std::numeric_limits<size_t>::max() ||
      layout.speculative_size > std::numeric_limits<size_t>::max() ||
      layout.token_count > std::numeric_limits<size_t>::max() / 4) {
    return false;
  }
  size_t size = kStateSnapshotHeaderSize;
  return checked_size_add(size, static_cast<size_t>(layout.target_size),
                          &size) &&
         checked_size_add(size, static_cast<size_t>(layout.draft_size),
                          &size) &&
         checked_size_add(size, static_cast<size_t>(layout.speculative_size),
                          &size) &&
         checked_size_add(size, static_cast<size_t>(layout.token_count) * 4,
                          out_size);
}

inline void write_state_snapshot_header(uint8_t *out,
                                        const state_snapshot_layout &layout) {
  static constexpr uint8_t magic[8] = {'F', 'L', 'L', 'A',
                                       'M', 'E', 'R', 'S'};
  std::memcpy(out, magic, sizeof(magic));
  write_u32_le(out + 8, kStateSnapshotVersion);
  uint32_t flags = 0;
  if (layout.draft_size > 0) {
    flags |= kStateSnapshotHasDraft;
  }
  if (layout.speculative_size > 0) {
    flags |= kStateSnapshotHasSpeculativeState;
  }
  write_u32_le(out + 12, flags);
  write_u32_le(out + 16, layout.speculative_type);
  // Filled after all payload sections have been written.
  write_u32_le(out + 20, 0);
  write_u64_le(out + 24, layout.target_size);
  write_u64_le(out + 32, layout.draft_size);
  write_u64_le(out + 40, layout.speculative_size);
  write_u64_le(out + 48, layout.token_count);
  write_u64_le(out + 56, static_cast<uint64_t>(layout.position));
}

inline uint32_t state_snapshot_checksum(const uint8_t *data, size_t size) {
  uint32_t checksum = 2166136261u;
  for (size_t i = 0; i < size; ++i) {
    if (i >= 20 && i < 24) {
      continue;
    }
    checksum ^= data[i];
    checksum *= 16777619u;
  }
  return checksum;
}

inline void finalize_state_snapshot(uint8_t *data, size_t size) {
  write_u32_le(data + 20, state_snapshot_checksum(data, size));
}

inline state_snapshot_decode_result decode_state_snapshot(
    const uint8_t *data, size_t size, state_snapshot_view *out,
    std::string *error) {
  static constexpr uint8_t magic[8] = {'F', 'L', 'L', 'A',
                                       'M', 'E', 'R', 'S'};
  if (size < sizeof(magic) || std::memcmp(data, magic, sizeof(magic)) != 0) {
    return state_snapshot_decode_result::not_snapshot;
  }
  auto invalid = [&](const char *message) {
    if (error != nullptr) {
      *error = message;
    }
    return state_snapshot_decode_result::invalid;
  };
  if (size < kStateSnapshotHeaderSize) {
    return invalid("snapshot header is truncated");
  }
  if (read_u32_le(data + 8) != kStateSnapshotVersion) {
    return invalid("snapshot version is unsupported");
  }
  const uint32_t flags = read_u32_le(data + 12);
  if ((flags & ~(kStateSnapshotHasDraft |
                 kStateSnapshotHasSpeculativeState)) != 0) {
    return invalid("snapshot header flags are invalid");
  }

  state_snapshot_layout layout;
  layout.speculative_type = read_u32_le(data + 16);
  layout.target_size = read_u64_le(data + 24);
  layout.draft_size = read_u64_le(data + 32);
  layout.speculative_size = read_u64_le(data + 40);
  layout.token_count = read_u64_le(data + 48);
  const uint64_t position = read_u64_le(data + 56);
  if (position > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return invalid("snapshot position is invalid");
  }
  layout.position = static_cast<int64_t>(position);
  if (((flags & kStateSnapshotHasDraft) != 0) !=
          (layout.draft_size > 0) ||
      ((flags & kStateSnapshotHasSpeculativeState) != 0) !=
          (layout.speculative_size > 0)) {
    return invalid("snapshot section flags do not match section sizes");
  }

  size_t expected_size = 0;
  if (!state_snapshot_size(layout, &expected_size) || expected_size != size) {
    return invalid("snapshot section sizes are invalid");
  }
  if (read_u32_le(data + 20) != state_snapshot_checksum(data, size)) {
    return invalid("snapshot checksum does not match its payload");
  }

  out->layout = layout;
  size_t offset = kStateSnapshotHeaderSize;
  out->target = data + offset;
  offset += static_cast<size_t>(layout.target_size);
  out->draft = layout.draft_size == 0 ? nullptr : data + offset;
  offset += static_cast<size_t>(layout.draft_size);
  out->speculative =
      layout.speculative_size == 0 ? nullptr : data + offset;
  offset += static_cast<size_t>(layout.speculative_size);
  out->tokens = layout.token_count == 0 ? nullptr : data + offset;
  return state_snapshot_decode_result::success;
}

} // namespace llama_dart_bridge_internal

#endif // LLAMA_DART_STATE_SNAPSHOT_H_
