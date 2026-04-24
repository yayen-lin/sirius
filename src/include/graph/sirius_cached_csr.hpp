#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cucascade/data/data_batch.hpp>
#include <duckdb/common/shared_ptr.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

struct sirius_cached_csr {
  // CSR arrays managed via rmm device_uvector
  rmm::device_uvector<int64_t> offsets{0, rmm::cuda_stream_default};  // num_vertices + 1
  rmm::device_uvector<int64_t> indices{0, rmm::cuda_stream_default};  // num_edges
  rmm::device_uvector<float> weights{0, rmm::cuda_stream_default};    // num_edges (optional)

  int64_t num_vertices = 0;
  int64_t num_edges    = 0;
  bool is_weighted     = false;
  bool is_built        = false;

  std::string key;  // cache key
  std::mutex build_mutex;

  // pending edge batches for CSR construction
  // accumulated by sirius_physical_csr_construction::execute()
  // consumed and cleared by build_csr_if_needed() after build
  std::vector<std::shared_ptr<cucascade::data_batch>> pending_batches;

  // cache key: "EDGE_TABLE:src_col:dst_col"
  static std::string MakeKey(const std::string& edge_table,
                             const std::string& src_col,
                             const std::string& dst_col)
  {
    std::string upper_table = edge_table;
    std::transform(upper_table.begin(), upper_table.end(), upper_table.begin(), ::toupper);
    return upper_table + ":" + src_col + ":" + dst_col;
  }
};

// global cache manager, stores and retrieves sirius_cached_csr instances
class sirius_csr_cache {
 public:
  static sirius_csr_cache& instance()
  {
    static sirius_csr_cache cache;
    return cache;
  }

  shared_ptr<sirius_cached_csr> find(const std::string& key)
  {
    std::scoped_lock lock(mutex_);
    auto it = map_.find(key);
    return it != map_.end() ? it->second : nullptr;
  }

  void insert(const std::string& key, shared_ptr<sirius_cached_csr> csr)
  {
    std::scoped_lock lock(mutex_);
    map_.emplace(key, std::move(csr));
  }

 private:
  sirius_csr_cache() = default;
  std::mutex mutex_;
  std::unordered_map<std::string, shared_ptr<sirius_cached_csr>> map_;
};

}  // namespace duckdb
