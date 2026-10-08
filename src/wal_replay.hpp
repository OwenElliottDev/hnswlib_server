#ifndef WAL_REPLAY_HPP
#define WAL_REPLAY_HPP

#include "index_context.hpp"
#include "nlohmann/json.hpp"
#include "wal.hpp"
#include <memory>
#include <string>

WalHeader makeWalHeader(const nlohmann::json &settings);

std::shared_ptr<IndexContext> contextFromWalHeader(const WalHeader &header, const std::string &indexName);

void startBackgroundWalReplay(IndexContext *ctx, const std::string &indexName, ResolvedWal resolved, const std::string &walPath,
                              int fsyncIntervalMs);

#endif // WAL_REPLAY_HPP
