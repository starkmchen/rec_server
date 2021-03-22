#include <algorithm>
#include <chrono>
#include <tuple>

#include "rec/beta_distribution.h"
#include "ads_feature.h"
#include "feature/feature.h"
#include "prediction_service.pb.h"  // tf-serving
#include "metis/metis.h"
#include "metis_kafka.pb.h"
#include "metrics/metrics.h"
#include "rec/rec.h"
#include "sharestore/sharestore.h"
#include "tf/tf.h"
#include "tf/tf_model.h"
#include "util/log.h"

namespace ad {

inline void swap(modelx::Model_result& lhs, modelx::Model_result& rhs) {
  lhs.Swap(&rhs);
}


void AdRec::InitShareStoreData() {
  const auto& user_id = request_->request().user_id();
  std::vector<std::string> keys;
  keys.push_back("nt:ads:user_counter:" + user_id);
  keys.push_back("nt:ads:user_profile:" + user_id);
  std::vector<std::pair<std::string, std::string>> results;
  {
    common::Timer timer(sharestoreMgetMs);
    if (!GetShareStore()->multiGetValue(GetSegment(), keys, &results)
          || results.size() < 2) {
      common::Stats::get()->Incr(sharestoreMgetError);
      LOG_ERROR("sharestore mget failed, or size=" << results.size());
      return;
    }
  }
  if (!results[0].second.empty() &&
      !store_user_counter_.ParseFromString(results[0].second)) {
    common::Stats::get()->Incr(counterParseError);
    LOG_ERROR("parse sharestore counter failed");
  }
  if (!results[1].second.empty() &&
      !store_user_profile_.ParseFromString(results[1].second)) {
    common::Stats::get()->Incr(userProfileParseError);
    LOG_ERROR("parse sharestore user_profile failed");
  }
}

/* ========================================================================== */

inline long Mod(long l, long r) {
  if (r < 2 || l == 0) {
    return 0;
  }
  return l % (r - 1) + 1;
}


void FillSequenceFeature(
    const DnnFieldItem& feature_info,
    const std::vector<FeatureResultPtr>& creatives,
    tensorflow::TensorProto& tensor_proto
    ) {
  auto field_seq_length = feature_info.field_seq_length;
  tensor_proto.set_dtype(tensorflow::DataType::DT_INT64);
  auto tensor_shape_proto = tensor_proto.mutable_tensor_shape();
  tensor_shape_proto->add_dim()->set_size(creatives.size());
  tensor_shape_proto->add_dim()->set_size(field_seq_length);

  auto field_max_length = feature_info.field_max_length;
  if (field_max_length <= 1) {
    field_max_length = 2;
    common::Stats::get()->Incr(fieldMaxLenError);
    LOG_ERROR("invalid field_max_length: name=" << feature_info.field_name
      << " field_max_length=" << field_max_length);
  }

  for (const auto& creative : creatives) {
    decltype(field_seq_length) count = 0;
    auto it_feature = creative->sequence_features.find(feature_info.field_name);
    if (it_feature != creative->sequence_features.cend()) {
      const auto& values = it_feature->second;
      for (auto it_v = values.cbegin();
          it_v != values.cend() && count < field_seq_length;
          ++it_v, ++count) {
        // 取模，降低空间size
        tensor_proto.add_int64_val(Mod(*it_v, field_max_length));
      }
    }
    for (; count < field_seq_length; ++count) {
      tensor_proto.add_int64_val(0);
    }
  }
}


void FillIntFeature(
    const DnnFieldItem& feature_info,
    const std::vector<FeatureResultPtr>& creatives,
    tensorflow::TensorProto& tensor_proto
    ) {
  tensor_proto.set_dtype(tensorflow::DataType::DT_INT64);
  auto tensor_shape_proto = tensor_proto.mutable_tensor_shape();
  tensor_shape_proto->add_dim()->set_size(creatives.size());
  tensor_shape_proto->add_dim()->set_size(1);

  const auto& feature_name = feature_info.field_name;
  auto field_max_length = feature_info.field_max_length;
  if (field_max_length <= 1) {
    field_max_length = 2;
    common::Stats::get()->Incr(fieldMaxLenError);
    LOG_ERROR("invalid field_max_length: name=" << feature_info.field_name
      << " field_max_length=" << field_max_length);
  }
  for (const auto& creative : creatives) {
    auto it = creative->int_features.find(feature_name);
    if (it == creative->int_features.cend()) {
      tensor_proto.add_int64_val(0);
    } else {
      // 取模
      tensor_proto.add_int64_val(Mod(it->second, field_max_length));
    }
  }
}


void FillFloatFeature(
    const DnnFieldItem& feature_info,
    const std::vector<FeatureResultPtr>& creatives,
    tensorflow::TensorProto& tensor_proto
    ) {
  tensor_proto.set_dtype(tensorflow::DataType::DT_FLOAT);
  auto tensor_shape_proto = tensor_proto.mutable_tensor_shape();
  tensor_shape_proto->add_dim()->set_size(creatives.size());
  tensor_shape_proto->add_dim()->set_size(1);

  const auto& feature_name = feature_info.field_name;
  for (const auto& creative : creatives) {
    auto it_feature = creative->float_features.find(feature_name);
    auto value = (it_feature == creative->float_features.cend() ? 0.0 :
      it_feature->second);
    tensor_proto.add_float_val(value);
  }
}


bool FillTfFeatures(const std::map<std::string, DnnFieldItem>& model_dict,
    const std::vector<FeatureResultPtr>& features,
    google::protobuf::Map<std::string, tensorflow::TensorProto>& inputs) {
  using Func = std::add_pointer_t<void (
    const DnnFieldItem& feature_info,
    const std::vector<FeatureResultPtr>& creatives,
    tensorflow::TensorProto& tensor_proto
    )>;
  const static std::map<std::string, Func> functions {
    {"float", FillFloatFeature},
    {"int", FillIntFeature},
    {"sequence", FillSequenceFeature},
    };
  common::Timer timer(tfFeatureMs);
  for (const auto& [feature_name, feature_info] : model_dict) {
    auto it = functions.find(feature_info.field_type);
    if (it == functions.cend()) {
      common::Stats::get()->Incr(tfFeatureTypeError);
      LOG_ERROR("invalid feature type: name=" << feature_name << " type="
        << feature_info.field_type);
      return false;
    }
    it->second(feature_info, features, inputs[feature_name]);
  }
  return true;
}

/* ========================================================================== */

void DelExcessCapAd(std::vector<Feature> &fs) {
  std::vector<Feature> new_fs;
  for (const auto feature& : fs) {
    auto day_ainst = feature.ad_data().ad_counter().
        ad_id().count_features_bj_1d().attr_install();
    auto cap = feature.ad_data().ad_info().day_attr_install_cap();
    if (cap > 0 && day_ainsts > cap) {
      continue;
    }
    new_fs.push_back(feature);
  }
  fs.swap(fs);
}

double GetExploreScore(
    double ctr, double cvr, const Feature &feature,
    std::default_random_engine &random_gen, bool is_random = false) {
  if (is_random) {
    std::uniform_int_distribution<int> udist(1, 1000);
    auto rand_num = udist(random_gen);
    double up_num(0.005), low_num(0.0002);
    double score = (up_num - low_num) * rand_num / 1000.0 + low_num;
    return score;
  }

  double score = ctr * cvr;
  if (score > 0.999) {
    score = 0.01;
  }

  double imp(1000);
  double alpha = score * imp;
  double beta = imp - alpha;
  BetaDistribution dist(alpha, beta);
  double explore_score = dist(random_gen);
  return explore_score;
}


bool FillScore(
    const std::vector<Feature>& fs,
    const std::vector<double> &ctr_vec,
    const std::vector<double> &cvr_vec,
    std::vector<modelx::Model_result>& ads,
    const modelx::PredictionRequest& ad_request,
    bool is_explore_flow,
    metis::ReqAds& req_ads,
    std::map<std::string, metis::RecAdInfo>& rec_ads
    ) {
  const auto& creatives_list = ad_request.creatives();
  if (ctr_vec.size() != creatives_list.size()) {
    common::Stats::get()->Incr(creativesSizeError);
    LOG_ERROR("creatives size invalid: " << ctr_vec.size() <<
      " " << creatives_list.size());
    return false;
  }
  if (cvr_vec.size() != fs.size()) {
    common::Stats::get()->Incr(cvrSizeError);
    LOG_ERROR("cvr size invalid: " << cvr_vec.size() << " " << fs.size());
    return false;
  }
  ads.reserve(fs.size());
  double floor_price = ad_request.contexts().floor_price();
  std::default_random_engine random_gen(
      std::chrono::system_clock::now().time_since_epoch().count());
  for (int i = 0; i < ctr_vec.size(); ++i) {
    const auto& ad_info = fs[i].ad_data().ad_info();
    const auto& creatives = creatives_list[i];
    if (creatives.creative().empty()) {
      common::Stats::get()->Incr(creativesEmptyError);
      LOG_ERROR("creatives is empty" << ad_info.ad_id());
      continue;
    }
    modelx::Model_result result;
    double score = ctr_vec[i] * cvr_vec[i];
    if (is_explore_flow) {
      score = GetExploreScore(
          ctr_vec[i], cvr_vec[i], fs[i], random_gen, true);
    }
    double ecpm = std::max(floor_price,
                           score * 1000.0 * creatives_list[i].bid_price());
    const auto& cid = creatives.creative()[0].creative_id();
    result.set_creative_id(cid);
    result.set_camp_id(ad_info.ad_id());
    result.set_model_spec(score);
    result.set_ecpm(ecpm);
    result.set_app_id(ad_info.app_id());
    result.set_ext(1);
    result.set_samplerate(1.0);
    ads.emplace_back(std::move(result));
    // req_ads
    {
      auto req_ad = req_ads.mutable_req_ads()->Add();
      req_ad->set_request_id(ad_request.request_id());
      req_ad->set_user_id(ad_request.user_id());
      req_ad->set_pos_id(ad_request.pos_id());
      req_ad->set_nation(ad_request.nation());
      req_ad->set_package_name(ad_request.contexts().package_name());
      req_ad->set_floor_price(floor_price);
      req_ad->set_creative_id(cid);
      req_ad->set_camp_id(ad_info.ad_id());
      req_ad->set_app_id(ad_info.app_id());
      req_ad->set_req_time(fs[i].context().req_time());
      req_ad->set_bid_price(creatives.bid_price());
      req_ad->set_pctr(ctr_vec[i]);
      req_ad->set_pcvr(cvr_vec[i]);
      req_ad->set_explore_flow(is_explore_flow);
    }
    // rec_ads
    {
      auto rec_ad = &rec_ads[cid];
      rec_ad->set_request_id(ad_request.request_id());
      rec_ad->set_user_id(ad_request.user_id());
      rec_ad->set_pos_id(ad_request.pos_id());
      rec_ad->set_nation(ad_request.nation());
      rec_ad->set_package_name(ad_request.contexts().package_name());
      rec_ad->set_floor_price(floor_price);
      rec_ad->set_creative_id(cid);
      rec_ad->set_camp_id(ad_info.ad_id());
      rec_ad->set_app_id(ad_info.app_id());
      rec_ad->set_req_time(fs[i].context().req_time());
      rec_ad->set_bid_price(creatives.bid_price());
      rec_ad->set_pctr(ctr_vec[i]);
      rec_ad->set_pcvr(cvr_vec[i]);
      rec_ad->mutable_feature()->CopyFrom(fs[i]);
      rec_ad->set_explore_flow(is_explore_flow);
    }
  }
  return true;
}


std::optional<std::vector<double>>
GetStatsCtr(const std::vector<Feature> &features) {
  std::vector<double> ctr_vec;
  ctr_vec.reserve(features.size());
  for (uint32_t i = 0; i < features.size(); ++i) {
    const auto &feature = features[i];
    double ctr = 0.05;
    double click = 0.0;
    double cid_imp =feature.ad_data().ad_counter().
        c_id().count_features_7d().imp();
    double pkg_imp = feature.ad_data().ad_counter().
        ad_package_name().count_features_7d().imp();
    double cate_imp = feature.ad_data().ad_counter().
        ad_package_category().count_features_7d().imp();
    if (cid_imp > 500) {
      click = feature.ad_data().ad_counter().
          c_id().count_features_7d().click();
      ctr = click / cid_imp;
    } else if (pkg_imp > 500) {
      click = feature.ad_data().ad_counter().
          ad_package_name().count_features_7d().click();
      ctr = click / pkg_imp;
    } else if (cate_imp > 500) {
      click = feature.ad_data().ad_counter().
          ad_package_category().count_features_7d().click();
      ctr = click / cate_imp;
    }
    ctr_vec.push_back(ctr);
  }
  return std::make_optional(std::move(ctr_vec));
}


std::vector<double> GetCvr(const std::vector<Feature> &features) {
  common::Timer timer(cvrMs);
  std::vector<double> cvr_vec;
  cvr_vec.reserve(features.size());
  for (uint32_t i = 0; i < features.size(); ++i) {
    const auto &feature = features[i];
    double cvr = 0.003;
    double attr_install(0.0);
    double cid_click =feature.ad_data().ad_counter().
        c_id().count_features_7d().click();
    double pkg_click = feature.ad_data().ad_counter().
        ad_package_name().count_features_7d().click();
    double cate_click = feature.ad_data().ad_counter().
        ad_package_category().count_features_7d().click();
    if (cid_click > 300) {
      attr_install = feature.ad_data().ad_counter().
          c_id().count_features_7d().attr_install();
      cvr = attr_install / cid_click;
    } else if (pkg_click > 300) {
      attr_install = feature.ad_data().ad_counter().
          ad_package_name().count_features_7d().attr_install();
      cvr = attr_install / pkg_click;
    } else if (cate_click > 300) {
      attr_install = feature.ad_data().ad_counter().
          ad_package_category().count_features_7d().attr_install();
      cvr = attr_install / cate_click;
    }
    cvr_vec.push_back(cvr);
  }
  return cvr_vec;
}


/*
  优先返回新广告
  1 将新广告放到ads前面
  2 shuffle新广告
  3 截断ads为size_limit
*/
void NewAdBoost(
    const std::vector<Feature>& fs,
    std::vector<modelx::Model_result> ads,
    size_t size_limit,
    std::map<std::string, metis::RecAdInfo> &rec_ad_map) {
  std::vector<const modelx::Model_result*> new_ad, old_ad;
  auto now_time = time(NULL);
  auto time_delta = 3 * 24 * 3600;
  for (size_t i = 0; i < fs.size() && i < ads.size(); ++i) {
    const auto &ad = fs[i];
    auto time_diff = now_time - ad.ad_data().ad_info().creative_create_time();
    auto cid_imp = ad.ad_data().ad_counter().c_id().count_features_7d().imp();
    if (time_diff < time_delta && cid_imp < 10000) {  // 新广告
      const auto &cid = ad.ad_data().ad_info().creative_id();
      auto it = rec_ad_map.find(cid);
      if (it != rec_ad_map.end()) {
        it->second.set_new_ad_flow(true);
      }
      new_ad.push_back(&ads[i]);  // 不能提前终止，因为要shuffle
    } else {
      old_ad.push_back(&ads[i]);
    }
  }
  if (new_ad.empty()) {
    return;
  }
  std::random_shuffle(new_ad.begin(), new_ad.end());
  std::vector<modelx::Model_result> ad_result;
  for (auto p : new_ad) {
    ad_result.emplace_back(*p);
    if (ad_result.size() >= size_limit) {
      break;
    }
  }
  for (auto p : old_ad) {
    if (ad_result.size() >= size_limit) {
      break;
    }
    ad_result.emplace_back(*p);
  }
  ads.swap(ad_result);
}


std::vector<FeatureResultPtr> FeatureExtract(const std::vector<Feature> &fs) {
  common::Timer timer(featureExtractMs);
  std::vector<FeatureResultPtr> features;
  features.reserve(fs.size());
  ModelFeature mf;
  for (const auto& ad_feature : fs) {
    features.push_back(mf.extract_feature(ad_feature));
  }
  return features;
}


std::optional<std::vector<double>>
AdRec::GetModelCtr(const std::vector<Feature> &fs) {
  // TODO: get tf model name, output
  std::string model_name = "dnn_model_t1", tf_output = "predictions";
  auto model = GetTfModel(model_name);
  if (model == nullptr) {
    common::Stats::get()->Incr(tfModelNameError);
    LOG_ERROR("invalid tf model_name: " << model_name);
    return std::nullopt;
  }
  // tf request
  tensorflow::serving::PredictRequest request;
  request.mutable_model_spec()->set_name(model_name);
  if (!FillTfFeatures(model->dnn_dict, FeatureExtract(fs),
      *request.mutable_inputs())) {
    return std::nullopt;
  }
  // call tf-serving
  tensorflow::serving::PredictResponse response;
  if (!GetTfClient().Predict(request, response)) {
    return std::nullopt;
  }
  const auto& it_resp = response.outputs().find(tf_output);
  if (it_resp == response.outputs().end()) {
    common::Stats::get()->Incr(tfModelOutputError);
    LOG_ERROR("tf output not found: model=" << model_name
      << " output=" << tf_output);
    return std::nullopt;
  }

  const auto &tensor_proto = it_resp->second;
  if (tensor_proto.dtype() != tensorflow::DataType::DT_FLOAT) {
    common::Stats::get()->Incr(tfDataTypeError);
    LOG_ERROR("tf response data_type is not float: " << tensor_proto.dtype());
    return std::nullopt;
  }
  if (tensor_proto.float_val_size() != fs.size()) {
    common::Stats::get()->Incr(tfTensorSizeError);
    LOG_ERROR("tf response size invalid: " << tensor_proto.float_val_size() <<
      " " << fs.size());
    return std::nullopt;
  }

  // set score
  std::vector<double> ctr_vec;
  ctr_vec.reserve(fs.size());
  for (int i = 0; i < tensor_proto.float_val_size(); ++i) {
    ctr_vec.push_back(tensor_proto.float_val(i));
  }
  return std::make_optional(std::move(ctr_vec));
}

/* ========================================================================== */

std::optional<std::vector<double>> AdRec::GetCtr(
    const std::vector<Feature>& fs) {
  auto model_exp_config_ite =
      request_->exp_params().exp_params().find("stats_ctr");
  if (model_exp_config_ite != request_->exp_params().exp_params().end() &&
      model_exp_config_ite->second == 1) {
    return GetStatsCtr(fs);
  }
  return GetModelCtr(fs);
}

/* ========================================================================== */

std::tuple<bool, bool> GetEEConfig() {
  bool is_explore_flow(false), is_new_ad_sup(false);
  std::uniform_int_distribution<int> udist(1, 100);
  std::default_random_engine random_gen(
      std::chrono::system_clock::now().time_since_epoch().count());
  auto rand_num = udist(random_gen);
  if (rand_num <= 15) {
    is_explore_flow = true;
  } else if (rand_num <= 20) {
    is_new_ad_sup = true;
  }
  return std::make_tuple(is_explore_flow, is_new_ad_sup);
}


void SendMetisLog(
    const std::vector<modelx::Model_result>& ads,
    const metis::ReqAds& req_ads,
    std::map<std::string, metis::RecAdInfo>& rec_ad_map) {
  // prepare rec ads for metis log
  metis::RecAds rec_ads;
  auto rec_ads_list = rec_ads.mutable_rec_ads();
  for (const auto& ad : ads) {
    const auto& cid = ad.creative_id();
    auto it = rec_ad_map.find(cid);
    if (it == rec_ad_map.end()) {
      continue;
    }
    rec_ads_list->Add()->Swap(&it->second);
  }
  // send kafka
  SendRecAds(rec_ads);
  SendReqAds(req_ads);
}


inline bool cmp (const modelx::Model_result& a, const modelx::Model_result& b) {
  return a.ecpm() > b.ecpm();
}


bool AdRec::Recommend(std::vector<modelx::Model_result>& ads) {
  InitShareStoreData();
  // convert raw data to feature_input
  std::vector<Feature> fs;  // feature_input. 每个元素对应一个creative
  if (!DataToFeatureInput(*request_, store_user_counter_, store_user_profile_,
        *GetStoreAdInfo(), *GetStoreAdCounter(), fs)) {
    common::Stats::get()->Incr(data2FeatureInputError);
    LOG_ERROR("convert raw data to feature_input failed");
    return false;
  }

  DelExcessBudgetAd(fs);

  auto cvr_vec = GetCvr(fs);
  auto ctr_vec = GetCtr(fs);
  if (!ctr_vec.has_value()) {
    return false;
  }

  bool is_explore_flow(false), is_new_ad_sup(false);
  std::tie (is_explore_flow, is_new_ad_sup) = GetEEConfig();

  metis::ReqAds req_ads;  // metis logging for all ads in request
  std::map<std::string, metis::RecAdInfo> rec_ad_map;
  if (!FillScore(fs, ctr_vec.value(), cvr_vec, ads, request_->request(),
      is_explore_flow, req_ads, rec_ad_map)) {
    return false;
  }

  auto size = std::min(ads.size(),
    static_cast<size_t>(request_->request().contexts().ad_count()));
  std::partial_sort(ads.begin(), ads.begin() + size, ads.end(), cmp);
  if (is_new_ad_sup) {
    NewAdBoost(fs, ads, size, rec_ad_map);
  }
  ads.resize(size);

  SendMetisLog(ads, req_ads, rec_ad_map);
  return true;
}

}  // end of namespace
