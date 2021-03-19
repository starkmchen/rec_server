#pragma once

#include <optional>

#include "ad_model_service.pb.h"
#include "store_table.pb.h"

namespace ad {

class AdRec {
 public:
  AdRec(const ad_model::AdRequest* request) : request_(request) {}
  bool Recommend(std::vector<modelx::Model_result>& ads);

 private:
  std::optional<std::vector<double>> GetModelCtr(const std::vector<Feature>&);
  std::optional<std::vector<double>> GetCtr(const std::vector<Feature>& fs);

  const ad_model::AdRequest* request_;

  StoreUserCounter store_user_counter_;
  StoreUserProfile store_user_profile_;
  void InitShareStoreData();
};

}  // end of namespace
