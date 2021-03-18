#include <atomic>
#include <memory>

#include "feature/feature.h"
#include "file_watcher.h"
#include "metrics/metrics.h"
#include "util/log.h"
#include "util/util.h"

namespace ad {

std::string ad_info_filename;
static std::shared_ptr<StoreAdInfo> ad_info;
std::string ad_counter_filename;
static std::shared_ptr<StoreAdCounter> ad_counter;


// 解析配置；加载文件并解析为protobuf；注册filewatcher监控文件变动并解析更新
// conf: 完整的server.json
bool InitFeature(const nlohmann::json& conf) {
  decltype(conf.find("")) it_path, it_s3, it_sub, it_data, it_cnt, it_info;
  if ((it_path = conf.find("data_path")) == conf.end() ||
      (!it_path.value().is_string()) ||
      (it_s3 = conf.find("s3")) == conf.end() ||
      (!it_s3.value().is_object()) ||
      (it_sub = it_s3.value().find("sub_dir")) == it_s3.value().end() ||
      (!it_sub.value().is_string()) ||
      (it_data = it_s3.value().find("data")) == it_s3.value().end() ||
      (!it_data.value().is_object()) ||
      (it_cnt = it_data.value().find("ad_counter")) == it_data.value().end() ||
      (!it_cnt.value().is_string()) ||
      (it_info = it_data.value().find("ad_info")) == it_data.value().end() ||
      (!it_info.value().is_string())
      ) {
    LOG_ERROR("s3 config invalid");
    return false;
  }

  ad_info_filename = it_path.value().get<std::string>() + "/" +
                     it_sub.value().get<std::string>() + "/ad_info.pb";
  ad_counter_filename = it_path.value().get<std::string>() + "/" +
                        it_sub.value().get<std::string>() + "/ad_counter.pb";
  ad_info = std::make_shared<StoreAdInfo>();
  ad_counter = std::make_shared<StoreAdCounter>();
  if (!ad_info->ParseFromString(ReadFile(ad_info_filename)) ||
      !ad_counter->ParseFromString(ReadFile(ad_counter_filename))) {
    LOG_ERROR("parse ad_info or ad_counter failed");
    return false;
  }
  LOG_INFO("ad_info init size=" << ad_info->ad_infos().size()
    << " ad_counter init size=" << ad_counter->store_ad_counter().size());

  bool b_ad_info = common::FileWatcher::Instance()->AddFile(ad_info_filename,
    [] (std::string content) {
      auto p = std::make_shared<StoreAdInfo>();
      if (p->ParseFromString(content)) {
        std::atomic_store_explicit(&ad_info, p, std::memory_order_release);
        LOG_INFO("ad_info parse succ, size=" << p->ad_infos().size());
      } else {
        common::Stats::get()->Incr(adInfoParseError);
        LOG_ERROR("ad_info parse failed");
      }
    });
  bool b_ad_cnt = common::FileWatcher::Instance()->AddFile(ad_counter_filename,
    [] (std::string content) {
      auto p = std::make_shared<StoreAdCounter>();
      if (p->ParseFromString(content)) {
        std::atomic_store_explicit(&ad_counter, p, std::memory_order_release);
        LOG_INFO("ad_counter parse succ, sz=" << p->store_ad_counter().size());
      } else {
        common::Stats::get()->Incr(adCounterParseError);
        LOG_ERROR("ad_counter parse failed");
      }
    });
  return b_ad_info && b_ad_cnt;
}


std::shared_ptr<StoreAdInfo> GetStoreAdInfo() {
  return std::atomic_load_explicit(&ad_info, std::memory_order_acquire);
}


std::shared_ptr<StoreAdCounter> GetStoreAdCounter() {
  return std::atomic_load_explicit(&ad_counter, std::memory_order_acquire);
}

}  // end of namespace