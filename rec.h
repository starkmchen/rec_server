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
  bool FillScore(
    const std::vector<Feature>& fs,
    const std::vector<double> &ctr_vec,
    const std::vector<double> &cvr_vec,
    std::vector<modelx::Model_result>& ads,
    const modelx::PredictionRequest& ad_request,
    bool is_explore_flow,
    metis::ReqAds& req_ads,
    std::map<std::string, metis::RecAdInfo>& rec_ads);

  std::optional<std::vector<double>> GetModelScore(
      const std::string &model_name,
      const std::string &tf_output,
      const std::vector<Feature>&);
  std::optional<std::vector<double>> GetCtr(const std::vector<Feature>& fs);
  std::optional<std::vector<double>> GetCvr(const std::vector<Feature>& fs);

  const ad_model::AdRequest* request_;

  StoreUserCounter store_user_counter_;
  StoreUserProfile store_user_profile_;
  void InitShareStoreData();
};

}  // end of namespace
