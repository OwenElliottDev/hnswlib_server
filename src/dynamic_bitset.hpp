#ifndef DYNAMIC_BITSET_HPP
#define DYNAMIC_BITSET_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>

class DynamicBitset {
private:
    std::vector<uint64_t> blocks_;
    size_t numBits_;

    static size_t blockIndex(size_t pos) { return pos / 64; }
    static size_t bitIndex(size_t pos) { return pos % 64; }
    static size_t numBlocks(size_t bits) { return (bits + 63) / 64; }

public:
    DynamicBitset() : numBits_(0) {}

    explicit DynamicBitset(size_t num_bits)
        : blocks_(numBlocks(num_bits), 0), numBits_(num_bits) {}

    void set(size_t pos) {
        if (pos >= numBits_) return;
        blocks_[blockIndex(pos)] |= (uint64_t(1) << bitIndex(pos));
    }

    void clear(size_t pos) {
        if (pos >= numBits_) return;
        blocks_[blockIndex(pos)] &= ~(uint64_t(1) << bitIndex(pos));
    }

    bool test(size_t pos) const {
        if (pos >= numBits_) return false;
        return (blocks_[blockIndex(pos)] >> bitIndex(pos)) & 1;
    }

    size_t count() const {
        size_t c = 0;
        for (auto block : blocks_) {
            c += __builtin_popcountll(block);
        }
        return c;
    }

    size_t size() const { return numBits_; }

    void resize(size_t new_size) {
        if (new_size <= numBits_) return;
        blocks_.resize(numBlocks(new_size), 0);
        numBits_ = new_size;
    }

    DynamicBitset& operator&=(const DynamicBitset& other) {
        size_t len = std::min(blocks_.size(), other.blocks_.size());
        for (size_t i = 0; i < len; ++i) {
            blocks_[i] &= other.blocks_[i];
        }
        for (size_t i = len; i < blocks_.size(); ++i) {
            blocks_[i] = 0;
        }
        return *this;
    }

    DynamicBitset& operator|=(const DynamicBitset& other) {
        if (other.blocks_.size() > blocks_.size()) {
            blocks_.resize(other.blocks_.size(), 0);
            numBits_ = std::max(numBits_, other.numBits_);
        }
        for (size_t i = 0; i < other.blocks_.size(); ++i) {
            blocks_[i] |= other.blocks_[i];
        }
        return *this;
    }

    DynamicBitset andNot(const DynamicBitset& other) const {
        DynamicBitset result = *this;
        size_t len = std::min(result.blocks_.size(), other.blocks_.size());
        for (size_t i = 0; i < len; ++i) {
            result.blocks_[i] &= ~other.blocks_[i];
        }
        return result;
    }

    std::vector<int> to_vector() const {
        std::vector<int> result;
        for (size_t i = 0; i < blocks_.size(); ++i) {
            uint64_t block = blocks_[i];
            while (block) {
                int bit = __builtin_ctzll(block);
                result.push_back(static_cast<int>(i * 64 + bit));
                block &= block - 1; // clear lowest set bit
            }
        }
        return result;
    }
};

#endif // DYNAMIC_BITSET_HPP
