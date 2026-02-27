#include "wal.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

#ifdef __APPLE__
#include <fcntl.h>
#else
#include <unistd.h>
#endif

// crc32 with standard reflected polynomial (0xEDB88320)
static uint32_t crc32_table[256];
static bool crc32_table_init = false;

static void init_crc32_table() {
    if (crc32_table_init) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) {
            if (c & 1)
                c = 0xEDB88320 ^ (c >> 1);
            else
                c >>= 1;
        }
        crc32_table[i] = c;
    }
    crc32_table_init = true;
}

uint32_t crc32(const uint8_t* data, size_t length) {
    init_crc32_table();
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

static void pushU32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v));
    buf.push_back(static_cast<uint8_t>(v >> 8));
    buf.push_back(static_cast<uint8_t>(v >> 16));
    buf.push_back(static_cast<uint8_t>(v >> 24));
}

static void pushU8(std::vector<uint8_t>& buf, uint8_t v) {
    buf.push_back(v);
}

static void pushBytes(std::vector<uint8_t>& buf, const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + len);
}

static uint32_t readU32(const uint8_t* data, size_t& off) {
    uint32_t v = static_cast<uint32_t>(data[off])
               | (static_cast<uint32_t>(data[off+1]) << 8)
               | (static_cast<uint32_t>(data[off+2]) << 16)
               | (static_cast<uint32_t>(data[off+3]) << 24);
    off += 4;
    return v;
}

static uint8_t readU8(const uint8_t* data, size_t& off) {
    return data[off++];
}

// matches DataStore's serializeFieldValue convention
static void serializeFieldValueToWal(std::vector<uint8_t>& buf, const FieldValue& value) {
    int variantIdx = static_cast<int>(value.index());
    pushU32(buf, static_cast<uint32_t>(variantIdx));

    switch (variantIdx) {
        case 0: { // long
            long v = std::get<long>(value);
            pushBytes(buf, &v, sizeof(v));
            break;
        }
        case 1: { // double
            double v = std::get<double>(value);
            pushBytes(buf, &v, sizeof(v));
            break;
        }
        case 2: { // string
            const std::string& s = std::get<std::string>(value);
            uint32_t len = static_cast<uint32_t>(s.size());
            pushU32(buf, len);
            pushBytes(buf, s.data(), s.size());
            break;
        }
    }
}

static FieldValue deserializeFieldValueFromWal(const uint8_t* data, size_t& off) {
    uint32_t variantIdx = readU32(data, off);

    switch (variantIdx) {
        case 0: { // long
            long v;
            std::memcpy(&v, data + off, sizeof(v));
            off += sizeof(v);
            return v;
        }
        case 1: { // double
            double v;
            std::memcpy(&v, data + off, sizeof(v));
            off += sizeof(v);
            return v;
        }
        case 2: { // string
            uint32_t len = readU32(data, off);
            std::string s(reinterpret_cast<const char*>(data + off), len);
            off += len;
            return s;
        }
        default:
            throw std::runtime_error("Unknown variant index in WAL entry");
    }
}

WriteAheadLog::WriteAheadLog(const std::string& path, const WalHeader& header)
    : path_(path), header_(header)
{
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    file_ = std::fopen(path.c_str(), "r+b");
    if (file_) {
        WalHeader existing = readHeader(file_);
        if (existing.magic != WAL_MAGIC) {
            std::fclose(file_);
            throw std::runtime_error("WAL file has invalid magic");
        }
        std::fseek(file_, 0, SEEK_END);
        approxSize_.store(static_cast<size_t>(std::ftell(file_)), std::memory_order_relaxed);
    } else {
        file_ = std::fopen(path.c_str(), "w+b");
        if (!file_) {
            throw std::runtime_error("Failed to create WAL file: " + path);
        }
        writeHeader(file_, header_);
        std::fflush(file_);
        approxSize_.store(WAL_HEADER_SIZE, std::memory_order_relaxed);
    }
}

WriteAheadLog::~WriteAheadLog() {
    stopFsyncThread();
    if (file_) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

void WriteAheadLog::writeHeader(FILE* f, const WalHeader& header) {
    uint8_t buf[WAL_HEADER_SIZE] = {};
    size_t off = 0;
    std::memcpy(buf + off, &header.magic, 4); off += 4;
    std::memcpy(buf + off, &header.version, 4); off += 4;
    std::memcpy(buf + off, &header.dimension, 4); off += 4;
    std::memcpy(buf + off, &header.M, 4); off += 4;
    std::memcpy(buf + off, &header.efConstruction, 4); off += 4;
    buf[off++] = static_cast<uint8_t>(header.spaceType);
    buf[off++] = static_cast<uint8_t>(header.vectorType);
    // rest is zero-padded

    std::fseek(f, 0, SEEK_SET);
    std::fwrite(buf, 1, WAL_HEADER_SIZE, f);
}

WalHeader WriteAheadLog::readHeader(FILE* f) {
    uint8_t buf[WAL_HEADER_SIZE];
    std::fseek(f, 0, SEEK_SET);
    size_t read = std::fread(buf, 1, WAL_HEADER_SIZE, f);
    if (read < WAL_HEADER_SIZE) {
        throw std::runtime_error("WAL file too small for header");
    }

    WalHeader h;
    size_t off = 0;
    std::memcpy(&h.magic, buf + off, 4); off += 4;
    std::memcpy(&h.version, buf + off, 4); off += 4;
    std::memcpy(&h.dimension, buf + off, 4); off += 4;
    std::memcpy(&h.M, buf + off, 4); off += 4;
    std::memcpy(&h.efConstruction, buf + off, 4); off += 4;
    h.spaceType = static_cast<WalSpaceType>(buf[off++]);
    h.vectorType = static_cast<WalVectorType>(buf[off++]);
    return h;
}

std::vector<uint8_t> WriteAheadLog::serializeAddPayload(uint32_t docId,
    const std::vector<float>& vector,
    const std::map<std::string, FieldValue>& metadata)
{
    std::vector<uint8_t> payload;
    pushU8(payload, static_cast<uint8_t>(WalOpType::ADD));
    pushU32(payload, docId);

    uint32_t vectorSize = static_cast<uint32_t>(vector.size() * sizeof(float));
    pushU32(payload, vectorSize);
    pushBytes(payload, vector.data(), vectorSize);

    uint32_t metaCount = static_cast<uint32_t>(metadata.size());
    pushU32(payload, metaCount);
    for (const auto& [key, value] : metadata) {
        uint32_t keyLen = static_cast<uint32_t>(key.size());
        pushU32(payload, keyLen);
        pushBytes(payload, key.data(), keyLen);
        serializeFieldValueToWal(payload, value);
    }
    return payload;
}

std::vector<uint8_t> WriteAheadLog::serializeDeletePayload(uint32_t docId) {
    std::vector<uint8_t> payload;
    pushU8(payload, static_cast<uint8_t>(WalOpType::DELETE));
    pushU32(payload, docId);
    return payload;
}

void WriteAheadLog::appendEntry(const std::vector<uint8_t>& payload) {
    uint32_t entryLen = static_cast<uint32_t>(payload.size());
    uint32_t checksum = crc32(payload.data(), payload.size());

    size_t bytesWritten = sizeof(entryLen) + payload.size() + sizeof(checksum);

    std::lock_guard<std::mutex> lock(writeMutex_);
    std::fwrite(&entryLen, sizeof(entryLen), 1, file_);
    std::fwrite(payload.data(), 1, payload.size(), file_);
    std::fwrite(&checksum, sizeof(checksum), 1, file_);

    approxSize_.fetch_add(bytesWritten, std::memory_order_relaxed);
}

void WriteAheadLog::logAdd(uint32_t docId, const std::vector<float>& vector,
                           const std::map<std::string, FieldValue>& metadata) {
    auto payload = serializeAddPayload(docId, vector, metadata);
    appendEntry(payload);
}

void WriteAheadLog::logDelete(uint32_t docId) {
    auto payload = serializeDeletePayload(docId);
    appendEntry(payload);
    deleteCount_.fetch_add(1, std::memory_order_relaxed);
}

std::pair<WalHeader, std::vector<WalEntry>> WriteAheadLog::readAll(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open WAL file: " + path);
    }

    WalHeader header = readHeader(f);
    if (header.magic != WAL_MAGIC) {
        std::fclose(f);
        throw std::runtime_error("WAL file has invalid magic");
    }

    std::fseek(f, 0, SEEK_END);
    long totalSize = std::ftell(f);
    std::fseek(f, WAL_HEADER_SIZE, SEEK_SET);

    std::vector<WalEntry> entries;
    long pos = WAL_HEADER_SIZE;

    while (pos + 4 <= totalSize) {
        uint32_t entryLen;
        if (std::fread(&entryLen, sizeof(entryLen), 1, f) != 1) break;
        pos += 4;

        if (pos + entryLen + 4 > static_cast<unsigned long>(totalSize)) break;

        std::vector<uint8_t> payload(entryLen);
        if (std::fread(payload.data(), 1, entryLen, f) != entryLen) break;
        pos += entryLen;

        uint32_t storedCrc;
        if (std::fread(&storedCrc, sizeof(storedCrc), 1, f) != 1) break;
        pos += 4;

        uint32_t computedCrc = crc32(payload.data(), payload.size());
        if (storedCrc != computedCrc) break; // corrupt entry, stop

        size_t off = 0;
        uint8_t opByte = readU8(payload.data(), off);
        WalOpType op = static_cast<WalOpType>(opByte);

        if (op == WalOpType::ADD) {
            WalEntry entry;
            entry.opType = WalOpType::ADD;
            entry.docId = readU32(payload.data(), off);

            uint32_t vectorSize = readU32(payload.data(), off);
            uint32_t numFloats = vectorSize / sizeof(float);
            entry.vector.resize(numFloats);
            std::memcpy(entry.vector.data(), payload.data() + off, vectorSize);
            off += vectorSize;

            uint32_t metaCount = readU32(payload.data(), off);
            for (uint32_t m = 0; m < metaCount; m++) {
                uint32_t keyLen = readU32(payload.data(), off);
                std::string key(reinterpret_cast<const char*>(payload.data() + off), keyLen);
                off += keyLen;
                FieldValue value = deserializeFieldValueFromWal(payload.data(), off);
                entry.metadata[key] = value;
            }
            entries.push_back(std::move(entry));
        } else if (op == WalOpType::DELETE) {
            WalEntry entry;
            entry.opType = WalOpType::DELETE;
            entry.docId = readU32(payload.data(), off);
            entries.push_back(std::move(entry));
        }
    }

    std::fclose(f);
    return {header, entries};
}

void WriteAheadLog::truncate() {
    std::lock_guard<std::mutex> lock(writeMutex_);
    // reopen as w+b to truncate, then rewrite header
    std::fclose(file_);
    file_ = std::fopen(path_.c_str(), "w+b");
    if (!file_) {
        throw std::runtime_error("Failed to truncate WAL file: " + path_);
    }
    writeHeader(file_, header_);
    std::fflush(file_);
    approxSize_.store(WAL_HEADER_SIZE, std::memory_order_relaxed);
    deleteCount_.store(0, std::memory_order_relaxed);
}

bool WriteAheadLog::tryCompact() {
    // only one thread compacts at a time; others skip
    bool expected = false;
    if (!compacting_.compare_exchange_strong(expected, true)) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(writeMutex_);
        std::fflush(file_);
    }

    auto [header, entries] = readAll(path_);

    // walk in order: ADD → store in map; DELETE → if in map, cancel both; else keep DELETE
    std::map<uint32_t, size_t> addIndex; // docId → index in survivors
    std::vector<WalEntry> survivors;

    for (auto& entry : entries) {
        if (entry.opType == WalOpType::ADD) {
            addIndex[entry.docId] = survivors.size();
            survivors.push_back(std::move(entry));
        } else if (entry.opType == WalOpType::DELETE) {
            auto it = addIndex.find(entry.docId);
            if (it != addIndex.end()) {
                survivors[it->second].opType = static_cast<WalOpType>(0xFF); // sentinel for removal
                addIndex.erase(it);
            } else {
                // delete refers to snapshot baseline, keep it
                survivors.push_back(std::move(entry));
            }
        }
    }

    survivors.erase(
        std::remove_if(survivors.begin(), survivors.end(),
            [](const WalEntry& e) { return static_cast<uint8_t>(e.opType) == 0xFF; }),
        survivors.end()
    );

    std::string compactPath = path_ + ".compact";
    FILE* cf = std::fopen(compactPath.c_str(), "wb");
    if (!cf) {
        compacting_.store(false);
        return false;
    }

    size_t newSize = WAL_HEADER_SIZE;
    {
        uint8_t buf[WAL_HEADER_SIZE] = {};
        size_t off = 0;
        std::memcpy(buf + off, &header.magic, 4); off += 4;
        std::memcpy(buf + off, &header.version, 4); off += 4;
        std::memcpy(buf + off, &header.dimension, 4); off += 4;
        std::memcpy(buf + off, &header.M, 4); off += 4;
        std::memcpy(buf + off, &header.efConstruction, 4); off += 4;
        buf[off++] = static_cast<uint8_t>(header.spaceType);
        buf[off++] = static_cast<uint8_t>(header.vectorType);
        std::fwrite(buf, 1, WAL_HEADER_SIZE, cf);
    }

    for (const auto& entry : survivors) {
        std::vector<uint8_t> payload;
        if (entry.opType == WalOpType::ADD) {
            payload = serializeAddPayload(entry.docId, entry.vector, entry.metadata);
        } else {
            payload = serializeDeletePayload(entry.docId);
        }
        uint32_t entryLen = static_cast<uint32_t>(payload.size());
        uint32_t checksum = crc32(payload.data(), payload.size());
        std::fwrite(&entryLen, sizeof(entryLen), 1, cf);
        std::fwrite(payload.data(), 1, payload.size(), cf);
        std::fwrite(&checksum, sizeof(checksum), 1, cf);
        newSize += sizeof(entryLen) + payload.size() + sizeof(checksum);
    }

    std::fflush(cf);
#ifdef __APPLE__
    fcntl(fileno(cf), F_FULLFSYNC);
#else
    fdatasync(fileno(cf));
#endif
    std::fclose(cf);

    // atomic rename, then reopen for appending
    {
        std::lock_guard<std::mutex> lock(writeMutex_);
        std::fclose(file_);
        file_ = nullptr;

        std::rename(compactPath.c_str(), path_.c_str());

        file_ = std::fopen(path_.c_str(), "r+b");
        if (!file_) {
            compacting_.store(false);
            throw std::runtime_error("Failed to reopen WAL after compaction");
        }
        std::fseek(file_, 0, SEEK_END);
        // reset counters to reflect compacted state
        approxSize_.store(static_cast<size_t>(std::ftell(file_)), std::memory_order_relaxed);
        deleteCount_.store(0, std::memory_order_relaxed);
    }

    compacting_.store(false);
    return true;
}

void WriteAheadLog::startFsyncThread(int intervalMs) {
    if (fsyncRunning_.load()) return;
    fsyncRunning_.store(true);
    fsyncThread_ = std::thread([this, intervalMs]() {
        while (fsyncRunning_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
            if (!fsyncRunning_.load()) break;
            std::lock_guard<std::mutex> lock(writeMutex_);
            if (file_) {
                std::fflush(file_);
#ifdef __APPLE__
                fcntl(fileno(file_), F_FULLFSYNC);
#else
                fdatasync(fileno(file_));
#endif
            }
        }
    });
}

void WriteAheadLog::stopFsyncThread() {
    if (!fsyncRunning_.load()) return;
    fsyncRunning_.store(false);
    if (fsyncThread_.joinable()) {
        fsyncThread_.join();
    }
}
