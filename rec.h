#pragma once

#include <future>
#include <memory>
#include <optional>
#include <vector>

#include "ad_model_service.pb.h"
#include "store_table.pb.h"

struct FeatureResult;

namespace ad {

class AdRec {
 public:
  AdRec(const ad_model::AdRequest* request) : request_(request) {}
  bool Recommend(std::vector<modelx::Model_result>& ads);

 private:
  bool FillScore(
    const std::vector<double> &ctr_vec,
    const std::vector<double> &cvr_vec,
    std::vector<modelx::Model_result>& ads,
    const modelx::PredictionRequest& ad_request,
    bool is_explore_flow,
    metis::ReqAds& req_ads,
    std::map<std::string, metis::RecAdInfo>& rec_ads);

  void DelExcessCapAd(std::vector<Feature> &fs);

  std::optional<std::vector<double>> GetModelScore(
      const std::string &model_name,
      const std::string &tf_output);
  std::optional<std::vector<double>> GetCtr();
  std::optional<std::vector<double>> GetCvr();
  using FutureCtr = std::future<std::optional<std::vector<double>>>;
  using FutureCvr = std::future<std::optional<std::vector<double>>>;
  std::pair<FutureCtr, FutureCvr> GetCtrCvr();
  void InitShareStoreData();

  const ad_model::AdRequest* request_;
  StoreUserCounter store_user_counter_;
  StoreUserProfile store_user_profile_;
  std::vector<Feature> raw_features_;
  std::vector<std::shared_ptr<FeatureResult>> model_features_;
};

}  // end of namespace
