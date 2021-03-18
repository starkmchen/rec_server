#pragma once

#include "ad_model_service.pb.h"
#include "store_table.pb.h"

namespace ad {

class AdRec {
 public:
  AdRec(const ad_model::AdRequest* request) : request_(request) {}
  bool Recommend(std::vector<modelx::Model_result>& ads);

 private:
  bool GetModelCtr(
      const std::vector<Feature> &fs,
      std::vector<double> &ctr_vec);

  const ad_model::AdRequest* request_;

  StoreUserCounter store_user_counter_;
  StoreUserProfile store_user_profile_;
  void InitShareStoreData();
};

}  // end of namespace
