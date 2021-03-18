#pragma once

#include <nlohmann/json.hpp>

#include "ad_model_service.pb.h"
#include "model_feature.pb.h"
#include "store_table.pb.h"

namespace ad {

bool DataToFeatureInput(
  const ad_model::AdRequest& ad_request,
  const StoreUserCounter& user_counter,
  const StoreUserProfile& user_profile,
  const StoreAdInfo& ad_info,
  const StoreAdCounter& ad_counter,
  std::vector<Feature> &feature
);

bool InitFeature(const nlohmann::json& conf);

std::shared_ptr<StoreAdInfo> GetStoreAdInfo();

std::shared_ptr<StoreAdCounter> GetStoreAdCounter();

}  // end of namespace
