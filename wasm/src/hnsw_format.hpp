#ifndef HNSW_FORMAT_HPP
#define HNSW_FORMAT_HPP

// hnswlib's saveIndex/loadIndex dump size_t header fields and size_t labels
// straight from memory, so the on-disk layout depends on sizeof(size_t). The
// server runs on 64-bit hosts while wasm32 has a 4-byte size_t. To keep a single
// interchangeable file format, index files are always stored in the 64-bit
// (LP64) layout and translated to/from the native layout here.
//
// File layout (64-bit):
//   u64 offsetLevel0, u64 maxElements, u64 curElementCount, u64 sizeDataPerElement,
//   u64 labelOffset, u64 offsetData, i32 maxLevel, u32 enterpointNode,
//   u64 maxM, u64 maxM0, u64 M, f64 mult, u64 efConstruction
//   curElementCount x [ links + vector (labelOffset bytes) | u64 label ]
//   curElementCount x [ u32 linkListSize | linkListSize bytes ]  (copied verbatim)

#include "hnswlib/hnswlib.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnsw_format {

namespace detail {

template <typename T> T readPod(std::istream &in) {
  T v{};
  in.read(reinterpret_cast<char *>(&v), sizeof(T));
  if (!in) {
    throw std::runtime_error("Index file is truncated or corrupted");
  }
  return v;
}

template <typename T> void writePod(std::ostream &out, const T &v) { out.write(reinterpret_cast<const char *>(&v), sizeof(T)); }

template <typename To, typename From> To checkedCast(From v) {
  if (v > static_cast<From>(std::numeric_limits<To>::max())) {
    throw std::runtime_error("Index file value exceeds platform limits");
  }
  return static_cast<To>(v);
}

// Header fields that are size_t in memory, read with width FromSize and
// written with width ToSize.
template <typename FromSize, typename ToSize> struct Translator {
  static size_t translate(const std::string &inPath, const std::string &outPath) {
    std::ifstream in(inPath, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Cannot open index file: " + inPath);
    }
    std::ofstream out(outPath, std::ios::binary);
    if (!out) {
      throw std::runtime_error("Cannot open index file for writing: " + outPath);
    }

    auto sizeField = [&]() {
      FromSize v = readPod<FromSize>(in);
      writePod(out, checkedCast<ToSize>(v));
      return static_cast<uint64_t>(v);
    };

    sizeField();                  // offsetLevel0
    sizeField();                  // maxElements
    uint64_t count = sizeField(); // curElementCount
    uint64_t inSizePerElement = readPod<FromSize>(in);
    uint64_t labelOffset = readPod<FromSize>(in);
    if (inSizePerElement != labelOffset + sizeof(FromSize)) {
      throw std::runtime_error("Unsupported index layout: label is not the last field of an element");
    }
    writePod(out, checkedCast<ToSize>(labelOffset + sizeof(ToSize)));
    writePod(out, checkedCast<ToSize>(labelOffset));
    sizeField();                          // offsetData
    writePod(out, readPod<int32_t>(in));  // maxLevel
    writePod(out, readPod<uint32_t>(in)); // enterpointNode
    sizeField();                          // maxM
    sizeField();                          // maxM0
    sizeField();                          // M
    writePod(out, readPod<double>(in));   // mult
    sizeField();                          // efConstruction

    std::vector<char> element(static_cast<size_t>(labelOffset));
    for (uint64_t i = 0; i < count; i++) {
      in.read(element.data(), element.size());
      if (!in) {
        throw std::runtime_error("Index file is truncated or corrupted");
      }
      out.write(element.data(), element.size());
      writePod(out, checkedCast<ToSize>(readPod<FromSize>(in))); // label
    }

    // upper-layer link lists are u32 sizes and u32 ids, identical on both layouts
    std::vector<char> buf(1 << 16);
    while (in) {
      in.read(buf.data(), buf.size());
      out.write(buf.data(), in.gcount());
    }
    if (!out) {
      throw std::runtime_error("Failed writing index file: " + outPath);
    }
    return static_cast<size_t>(count);
  }
};

} // namespace detail

static_assert(sizeof(hnswlib::labeltype) == sizeof(size_t), "hnswlib labels are expected to be size_t");

// Translates a 64-bit layout index file into one hnswlib::loadIndex can read on
// this platform. Returns the element count stored in the file.
inline size_t portableToNative(const std::string &portablePath, const std::string &nativePath) {
  return detail::Translator<uint64_t, size_t>::translate(portablePath, nativePath);
}

// Translates a file written by hnswlib::saveIndex on this platform into the
// 64-bit layout the server reads.
inline size_t nativeToPortable(const std::string &nativePath, const std::string &portablePath) {
  return detail::Translator<size_t, uint64_t>::translate(nativePath, portablePath);
}

} // namespace hnsw_format

#endif // HNSW_FORMAT_HPP
