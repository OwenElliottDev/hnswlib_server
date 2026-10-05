// Embind surface for EdgeIndex. This is the low-level API; js/hnswlib-edge.mjs
// wraps it with file loading, typed-array handling and JSON (de)serialisation.
// Metadata crosses the boundary as JSON text so the server's FieldValue
// conversions in models.hpp are reused unchanged.

#include "edge_index.hpp"
#include "models.hpp"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <limits>
#include <set>

using namespace emscripten;

namespace {

const char *SCRATCH_DIR = "/tmp";

[[noreturn]] void throwJsError(const std::string &message) {
  val::global("Error").new_(message).throw_();
  __builtin_unreachable();
}

// Runs f, rethrowing C++ exceptions as JS Errors carrying the message.
template <typename F> auto guarded(F &&f) -> decltype(f()) {
  try {
    return f();
  } catch (const std::exception &e) {
    throwJsError(e.what());
  }
}

void checkIntegerRange(const std::string &key, const nlohmann::json &v) {
  if (v.is_number_integer()) {
    auto i = v.get<int64_t>();
    if (i < std::numeric_limits<long>::min() || i > std::numeric_limits<long>::max()) {
      throw std::invalid_argument("Integer metadata value for '" + key +
                                  "' is outside the 32-bit range supported in WASM; store it as a float or string");
    }
  } else if (v.is_array()) {
    for (const auto &el : v) {
      checkIntegerRange(key, el);
    }
  }
}

// JS can't distinguish 3 from 3.0, so fields listed in doubleFields are coerced
// to doubles to keep one numeric type per field (mixed long/double values in a
// field break range filters).
nlohmann::json coerceToDouble(const nlohmann::json &v) {
  if (v.is_number()) {
    return v.get<double>();
  }
  if (v.is_array() && !v.empty() && v[0].is_number()) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &el : v) {
      arr.push_back(el.get<double>());
    }
    return arr;
  }
  return v;
}

std::vector<Metadata> parseMetadatas(const std::string &text, const std::set<std::string> &doubleFields) {
  if (text.empty()) {
    return {};
  }
  auto j = nlohmann::json::parse(text);
  if (j.is_null()) {
    return {};
  }
  if (!j.is_array()) {
    throw std::invalid_argument("metadatas must be an array");
  }
  std::vector<Metadata> out;
  out.reserve(j.size());
  for (const auto &obj : j) {
    Metadata meta;
    if (!obj.is_null()) {
      if (!obj.is_object()) {
        throw std::invalid_argument("each metadata entry must be an object");
      }
      for (const auto &[key, value] : obj.items()) {
        if (value.is_null()) {
          continue;
        }
        nlohmann::json v = doubleFields.count(key) ? coerceToDouble(value) : value;
        checkIntegerRange(key, v);
        // models.hpp's helpers aren't ADL-visible for std::variant, so call them directly
        FieldValue fv;
        from_json(v, fv);
        meta[key] = std::move(fv);
      }
    }
    out.push_back(std::move(meta));
  }
  return out;
}

nlohmann::json metadataToJson(const Metadata &meta) {
  nlohmann::json j = nlohmann::json::object();
  for (const auto &[key, value] : meta) {
    to_json(j[key], value);
  }
  return j;
}

template <typename T> val toTypedArray(const char *ctor, const std::vector<T> &v) {
  // the memory view aliases the wasm heap; constructing a new typed array copies it out
  return val::global(ctor).new_(typed_memory_view(v.size(), v.data()));
}

std::set<std::string> doubleFieldsFrom(const nlohmann::json &settings) {
  std::set<std::string> fields;
  if (settings.contains("doubleFields") && settings["doubleFields"].is_array()) {
    for (const auto &f : settings["doubleFields"]) {
      fields.insert(f.get<std::string>());
    }
  }
  return fields;
}

} // namespace

class Index {
public:
  explicit Index(std::unique_ptr<EdgeIndex> index) : index_(std::move(index)), doubleFields_(doubleFieldsFrom(index_->settings())) {}

  static std::unique_ptr<Index> create(const std::string &settingsJson, int initialCapacity) {
    return guarded([&] {
      auto settings = nlohmann::json::parse(settingsJson);
      return std::make_unique<Index>(std::make_unique<EdgeIndex>(settings, static_cast<size_t>(std::max(initialCapacity, 1))));
    });
  }

  // Paths are in the Emscripten virtual filesystem; the JS wrapper writes the
  // downloaded bytes there first so they never transit the wasm heap.
  static std::unique_ptr<Index> loadFiles(const std::string &binPath, const std::string &settingsJson, const std::string &dataPath) {
    return guarded([&] {
      auto settings = nlohmann::json::parse(settingsJson);
      return std::make_unique<Index>(EdgeIndex::load(binPath, settings, dataPath, SCRATCH_DIR));
    });
  }

  void saveFiles(const std::string &binPath, const std::string &dataPath) {
    guarded([&] { index_->save(binPath, dataPath, SCRATCH_DIR); });
  }

  // ids: array-like of ints; vectors: flat array-like of n * dimension floats.
  void addDocuments(const val &ids, const val &vectors, const std::string &metadatasJson) {
    guarded([&] {
      auto idVec = convertJSArrayToNumberVector<int>(ids);
      auto flat = convertJSArrayToNumberVector<float>(vectors);
      size_t dim = static_cast<size_t>(index_->dimension());
      if (flat.size() != idVec.size() * dim) {
        throw std::invalid_argument("Expected " + std::to_string(idVec.size()) + " vectors of dimension " + std::to_string(dim) + " (" +
                                    std::to_string(idVec.size() * dim) + " floats), got " + std::to_string(flat.size()) + " floats");
      }
      index_->addDocuments(idVec, flat.data(), idVec.size(), parseMetadatas(metadatasJson, doubleFields_));
    });
  }

  void deleteDocuments(const val &ids) {
    guarded([&] { index_->deleteDocuments(convertJSArrayToNumberVector<int>(ids)); });
  }

  // Returns { hits: Int32Array, distances: Float32Array, metadatas?: string (JSON array) }.
  val search(const val &query, int k, int efSearch, const std::string &filter, bool returnMetadata, int rerankSize) {
    return guarded([&] {
      auto q = convertJSArrayToNumberVector<float>(query);
      SearchOptions opts;
      opts.k = static_cast<size_t>(std::max(k, 0));
      opts.efSearch = efSearch;
      opts.filter = filter;
      opts.returnMetadata = returnMetadata;
      opts.rerankSize = rerankSize;
      SearchResult r = index_->search(q.data(), q.size(), opts);

      val out = val::object();
      out.set("hits", toTypedArray("Int32Array", r.hits));
      out.set("distances", toTypedArray("Float32Array", r.distances));
      if (returnMetadata) {
        nlohmann::json metas = nlohmann::json::array();
        for (const auto &m : r.metadatas) {
          metas.push_back(metadataToJson(m));
        }
        out.set("metadatas", metas.dump());
      }
      return out;
    });
  }

  bool contains(int id) { return index_->contains(id); }

  // Returns { vector: Float32Array, metadata: string (JSON) } or null.
  val getDocument(int id) {
    return guarded([&] {
      if (!index_->contains(id)) {
        return val::null();
      }
      val out = val::object();
      out.set("vector", toTypedArray("Float32Array", index_->getVector(id)));
      out.set("metadata", metadataToJson(index_->getMetadata(id)).dump());
      return out;
    });
  }

  std::string settings() const { return index_->settings().dump(); }
  std::string status() const { return index_->status().dump(); }
  int dimension() const { return index_->dimension(); }

private:
  std::unique_ptr<EdgeIndex> index_;
  std::set<std::string> doubleFields_;
};

std::string version() { return HNSWLIB_VERSION; }

EMSCRIPTEN_BINDINGS(hnswlib_edge) {
  function("version", &version);
  class_<Index>("Index")
      .class_function("create", &Index::create)
      .class_function("loadFiles", &Index::loadFiles)
      .function("saveFiles", &Index::saveFiles)
      .function("addDocuments", &Index::addDocuments)
      .function("deleteDocuments", &Index::deleteDocuments)
      .function("search", &Index::search)
      .function("contains", &Index::contains)
      .function("getDocument", &Index::getDocument)
      .function("settings", &Index::settings)
      .function("status", &Index::status)
      .function("dimension", &Index::dimension);
}
