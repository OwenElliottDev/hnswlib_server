#ifndef WAL_HPP
#define WAL_HPP

#include "field_value.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static constexpr uint32_t WAL_MAGIC = 0x57414C31; // "WAL1"
static constexpr uint32_t WAL_VERSION = 1;
static constexpr size_t WAL_HEADER_SIZE = 256;
static constexpr size_t WAL_COMPACT_THRESHOLD = 64 * 1024 * 1024; // 64MB

enum class WalOpType : uint8_t { ADD = 0x01, DELETE = 0x02 };

enum class WalSpaceType : uint8_t { L2 = 0, IP = 1 };

enum class WalVectorType : uint8_t { FLOAT32 = 0, FLOAT16 = 1, BFLOAT16 = 2 };

// WAL binary format:
//
// header (256 bytes, zero-padded):
//   [magic:4] [version:4] [dimension:4] [M:4] [efConstruction:4]
//   [spaceType:1] [vectorType:1] [reserved:234]
//
// each entry:
//   [entryLength:4] [payload...] [crc32:4]
//
// ADD payload (opType=0x01):
//   [opType:1] [docId:4] [vectorSize:4] [floatVector:vectorSize]
//   [metaCount:4] [for each: keyLen:4, key:bytes, variantIdx:4, value:...]
//   value encoding: long=8B, double=8B, string=len:4+data
//
// DELETE payload (opType=0x02):
//   [opType:1] [docId:4]

struct WalHeader {
  uint32_t magic = WAL_MAGIC;
  uint32_t version = WAL_VERSION;
  int32_t dimension = 0;
  int32_t M = 16;
  int32_t efConstruction = 512;
  WalSpaceType spaceType = WalSpaceType::IP;
  WalVectorType vectorType = WalVectorType::FLOAT32;
};

struct WalEntry {
  WalOpType opType;
  uint32_t docId;
  std::vector<float> vector;                  // ADD only
  std::map<std::string, FieldValue> metadata; // ADD only
};

uint32_t crc32(const uint8_t *data, size_t length);

class WriteAheadLog {
public:
  WriteAheadLog(const std::string &path, const WalHeader &header);
  ~WriteAheadLog();

  void logAdd(uint32_t docId, const std::vector<float> &vector, const std::map<std::string, FieldValue> &metadata);
  void logDelete(uint32_t docId);

  static std::pair<WalHeader, std::vector<WalEntry>> readAll(const std::string &path);

  void truncate();
  bool tryCompact();
  size_t approxSize() const { return approxSize_.load(std::memory_order_relaxed); }
  bool hasDeletes() const { return deleteCount_.load(std::memory_order_relaxed) > 0; }

  void startFsyncThread(int intervalMs);
  void stopFsyncThread();

private:
  void writeHeader(FILE *f, const WalHeader &header);
  static WalHeader readHeader(FILE *f);
  static std::vector<uint8_t> serializeAddPayload(uint32_t docId, const std::vector<float> &vector,
                                                  const std::map<std::string, FieldValue> &metadata);
  static std::vector<uint8_t> serializeDeletePayload(uint32_t docId);
  void appendEntry(const std::vector<uint8_t> &payload);

  std::string path_;
  WalHeader header_;
  FILE *file_ = nullptr;
  std::mutex writeMutex_;

  std::atomic<size_t> approxSize_{0};
  std::atomic<uint32_t> deleteCount_{0};
  std::atomic<bool> compacting_{false};

  std::atomic<bool> fsyncRunning_{false};
  std::thread fsyncThread_;
};

#endif // WAL_HPP
