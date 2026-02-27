#include <gtest/gtest.h>
#include "wal.hpp"
#include <filesystem>
#include <cstdio>
#include <thread>
#include <vector>
#include <set>

class WalTest : public ::testing::Test {
protected:
    std::string testDir = "test_wal_tmp";

    void SetUp() override {
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        std::filesystem::remove_all(testDir);
    }

    std::string walPath(const std::string& name = "test") {
        return testDir + "/" + name + ".wal";
    }

    WalHeader makeHeader(int dim = 4) {
        WalHeader h;
        h.dimension = dim;
        h.M = 16;
        h.efConstruction = 200;
        h.spaceType = WalSpaceType::IP;
        h.vectorType = WalVectorType::FLOAT32;
        return h;
    }
};

TEST_F(WalTest, HeaderWriteReadRoundtrip) {
    WalHeader h;
    h.dimension = 128;
    h.M = 32;
    h.efConstruction = 400;
    h.spaceType = WalSpaceType::L2;
    h.vectorType = WalVectorType::BFLOAT16;

    {
        WriteAheadLog wal(walPath(), h);
    }

    auto [readH, entries] = WriteAheadLog::readAll(walPath());
    EXPECT_EQ(readH.magic, WAL_MAGIC);
    EXPECT_EQ(readH.version, WAL_VERSION);
    EXPECT_EQ(readH.dimension, 128);
    EXPECT_EQ(readH.M, 32);
    EXPECT_EQ(readH.efConstruction, 400);
    EXPECT_EQ(readH.spaceType, WalSpaceType::L2);
    EXPECT_EQ(readH.vectorType, WalVectorType::BFLOAT16);
    EXPECT_TRUE(entries.empty());
}

TEST_F(WalTest, AddEntryRoundtrip) {
    WalHeader h = makeHeader(3);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    std::map<std::string, FieldValue> meta = {
        {"name", std::string("alice")},
        {"age", 30L},
        {"score", 99.5}
    };

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(42, vec, meta);
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].opType, WalOpType::ADD);
    EXPECT_EQ(entries[0].docId, 42u);
    ASSERT_EQ(entries[0].vector.size(), 3u);
    EXPECT_FLOAT_EQ(entries[0].vector[0], 1.0f);
    EXPECT_FLOAT_EQ(entries[0].vector[1], 2.0f);
    EXPECT_FLOAT_EQ(entries[0].vector[2], 3.0f);

    EXPECT_EQ(std::get<std::string>(entries[0].metadata.at("name")), "alice");
    EXPECT_EQ(std::get<long>(entries[0].metadata.at("age")), 30L);
    EXPECT_DOUBLE_EQ(std::get<double>(entries[0].metadata.at("score")), 99.5);
}

TEST_F(WalTest, DeleteEntryRoundtrip) {
    WalHeader h = makeHeader();

    {
        WriteAheadLog wal(walPath(), h);
        wal.logDelete(7);
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].opType, WalOpType::DELETE);
    EXPECT_EQ(entries[0].docId, 7u);
}

TEST_F(WalTest, MultipleMixedEntries) {
    WalHeader h = makeHeader(2);

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(1, {1.0f, 2.0f}, {{"k", std::string("v")}});
        wal.logAdd(2, {3.0f, 4.0f}, {});
        wal.logDelete(1);
        wal.logAdd(3, {5.0f, 6.0f}, {{"x", 10L}});
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    ASSERT_EQ(entries.size(), 4u);
    EXPECT_EQ(entries[0].opType, WalOpType::ADD);
    EXPECT_EQ(entries[0].docId, 1u);
    EXPECT_EQ(entries[1].opType, WalOpType::ADD);
    EXPECT_EQ(entries[1].docId, 2u);
    EXPECT_EQ(entries[2].opType, WalOpType::DELETE);
    EXPECT_EQ(entries[2].docId, 1u);
    EXPECT_EQ(entries[3].opType, WalOpType::ADD);
    EXPECT_EQ(entries[3].docId, 3u);
}

TEST_F(WalTest, CrcCorruptionDetection) {
    WalHeader h = makeHeader(2);

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(1, {1.0f, 2.0f}, {});
        wal.logAdd(2, {3.0f, 4.0f}, {});
    }

    // corrupt a byte in the first entry payload
    {
        FILE* f = std::fopen(walPath().c_str(), "r+b");
        ASSERT_NE(f, nullptr);
        // header(256) + entryLen(4) + opType(1) = docId byte
        std::fseek(f, WAL_HEADER_SIZE + 4 + 1, SEEK_SET);
        uint8_t garbage = 0xFF;
        std::fwrite(&garbage, 1, 1, f);
        std::fclose(f);
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    EXPECT_EQ(entries.size(), 0u);
}

TEST_F(WalTest, PartialWriteDetection) {
    WalHeader h = makeHeader(2);

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(1, {1.0f, 2.0f}, {});
        wal.logAdd(2, {3.0f, 4.0f}, {});
    }

    {
        FILE* f = std::fopen(walPath().c_str(), "r+b");
        std::fseek(f, 0, SEEK_END);
        long size = std::ftell(f);
        std::fclose(f);
        std::filesystem::resize_file(walPath(), size - 6);
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    EXPECT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].docId, 1u);
}

TEST_F(WalTest, TruncateClearsEntriesKeepsHeader) {
    WalHeader h = makeHeader(2);

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(1, {1.0f, 2.0f}, {});
        wal.logAdd(2, {3.0f, 4.0f}, {});
        wal.truncate();
    }

    auto [readH, entries] = WriteAheadLog::readAll(walPath());
    EXPECT_EQ(readH.dimension, 2);
    EXPECT_TRUE(entries.empty());

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(3, {5.0f, 6.0f}, {});
    }

    auto [_, entries2] = WriteAheadLog::readAll(walPath());
    ASSERT_EQ(entries2.size(), 1u);
    EXPECT_EQ(entries2[0].docId, 3u);
}

TEST_F(WalTest, CompactionCancelsAddDeletePairs) {
    WalHeader h = makeHeader(2);

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(1, {1.0f, 2.0f}, {{"k", std::string("v")}});
        wal.logAdd(2, {3.0f, 4.0f}, {});
        wal.logDelete(1); // cancels ADD id=1
        wal.tryCompact();
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].opType, WalOpType::ADD);
    EXPECT_EQ(entries[0].docId, 2u);
}

TEST_F(WalTest, CompactionPreservesOrphanDelete) {
    WalHeader h = makeHeader(2);

    {
        WriteAheadLog wal(walPath(), h);
        wal.logDelete(5); // no prior ADD — refers to snapshot baseline
        wal.logAdd(1, {1.0f, 2.0f}, {});
        wal.tryCompact();
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    ASSERT_EQ(entries.size(), 2u);
    // delete id=5 should be preserved (snapshot-relative)
    EXPECT_EQ(entries[0].opType, WalOpType::DELETE);
    EXPECT_EQ(entries[0].docId, 5u);
    EXPECT_EQ(entries[1].opType, WalOpType::ADD);
    EXPECT_EQ(entries[1].docId, 1u);
}

TEST_F(WalTest, ConcurrentLogAddFromMultipleThreads) {
    WalHeader h = makeHeader(2);
    const int numThreads = 8;
    const int entriesPerThread = 100;

    {
        WriteAheadLog wal(walPath(), h);

        std::vector<std::thread> threads;
        for (int t = 0; t < numThreads; t++) {
            threads.emplace_back([&wal, t, entriesPerThread]() {
                for (int i = 0; i < entriesPerThread; i++) {
                    uint32_t docId = static_cast<uint32_t>(t * entriesPerThread + i);
                    wal.logAdd(docId, {1.0f, 2.0f}, {});
                }
            });
        }
        for (auto& th : threads) {
            th.join();
        }
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    ASSERT_EQ(entries.size(), static_cast<size_t>(numThreads * entriesPerThread));

    std::set<uint32_t> ids;
    for (const auto& e : entries) {
        EXPECT_EQ(e.opType, WalOpType::ADD);
        ids.insert(e.docId);
    }
    EXPECT_EQ(ids.size(), static_cast<size_t>(numThreads * entriesPerThread));
}

TEST_F(WalTest, EmptyMetadataRoundtrip) {
    WalHeader h = makeHeader(2);

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(1, {1.0f, 2.0f}, {});
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_TRUE(entries[0].metadata.empty());
}

TEST_F(WalTest, ApproxSize) {
    WalHeader h = makeHeader(2);

    WriteAheadLog wal(walPath(), h);
    size_t headerOnly = wal.approxSize();
    EXPECT_EQ(headerOnly, WAL_HEADER_SIZE);

    wal.logAdd(1, {1.0f, 2.0f}, {});
    size_t afterOne = wal.approxSize();
    EXPECT_GT(afterOne, headerOnly);
}

TEST_F(WalTest, ReopenExistingWal) {
    WalHeader h = makeHeader(2);

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(1, {1.0f, 2.0f}, {});
    }

    {
        WriteAheadLog wal(walPath(), h);
        wal.logAdd(2, {3.0f, 4.0f}, {});
    }

    auto [_, entries] = WriteAheadLog::readAll(walPath());
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].docId, 1u);
    EXPECT_EQ(entries[1].docId, 2u);
}
