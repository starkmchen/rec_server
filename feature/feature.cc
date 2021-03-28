#include <sys/time.h>

#include <vector>
#include <string>
#include "feature/feature.h"
#include "metrics/metrics.h"

namespace ad {

bool DataToFeatureInput(
    const ad_model::AdRequest &ad_request,
    const StoreUserCounter &user_counter,
    const StoreUserProfile &user_profile,
    const StoreAdInfo &store_ad_info,
    const StoreAdCounter &ad_counter,
    std::vector<Feature> &feature) {
  common::Timer timer(data2FeatureInputMs);

  struct timeval tv;
  gettimeofday(&tv, NULL);
  int64_t req_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;

  const auto &model_request = ad_request.request();

  UserProfile feature_user_profile;
  feature_user_profile.mutable_user_base()->
      CopyFrom(user_profile.user_base());
  feature_user_profile.mutable_user_behavior()->
      CopyFrom(user_profile.user_behavior());
  feature_user_profile.set_user_id(model_request.user_id());

  UserCount feature_user_counter;
  std::string key = "user_id#" + model_request.user_id();
  auto ite = user_counter.store_user_counter().find(key);
  if (ite != user_counter.store_user_counter().end()) {
    feature_user_counter.mutable_user_id()->CopyFrom(ite->second);
  }
  feature_user_profile.mutable_user_counter()->CopyFrom(feature_user_counter);

  Context context;
  context.set_pos_id(model_request.pos_id());
  context.set_network_type(model_request.contexts().network_type());
  context.set_os_version(model_request.contexts().os_version());
  context.set_brand(model_request.contexts().brand());
  context.set_model(model_request.contexts().model());
  context.set_language(model_request.contexts().language());
  context.set_app_version_code(model_request.contexts().app_version_code());
  context.set_app_name(model_request.contexts().package_name());
  context.set_client_ip(model_request.user_ip());
  context.set_req_time(req_time);

  for (int32_t i = 0; i < model_request.creatives_size(); ++i) {
    AdInfo ad_info;
    ad_info.set_ad_id(model_request.creatives(i).camp_id());
    const std::string &pkg_name = model_request.creatives(i).app_id();
    ad_info.set_app_id(pkg_name);
    auto ite_info = store_ad_info.ad_infos().find(pkg_name);
    if (ite_info != store_ad_info.ad_infos().end()) {
      ad_info.set_category(ite_info->second.category());
    }
    auto adinfo_key = "ad_id#" + std::to_string(ad_info.ad_id());
    auto adinfo_ite = store_ad_info.ad_infos().find(adinfo_key);
    if (adinfo_ite != store_ad_info.ad_infos().end()) {
      ad_info.set_day_attr_install_cap(adinfo_ite->second.day_attr_install_cap());
    }
    ad_info.set_attr_platform(model_request.creatives(i).attr_platform());
    ad_info.set_is_auto_download(model_request.creatives(i).is_auto_download());
    ad_info.set_bid_price(model_request.creatives(i).bid_price());

    UserAdFeature user_ad_feature;
    std::string key = "user_id#ad_id#" +
        model_request.user_id() + "#" + std::to_string(ad_info.ad_id());
    auto ite = user_counter.store_user_counter().find(key);
    if (ite != user_counter.store_user_counter().end()) {
      user_ad_feature.mutable_user_ad_count()->
          mutable_user_id_ad_id()->CopyFrom(ite->second);
    }

    key = "user_id#ad_package_name#" +
        model_request.user_id() + "#" + ad_info.app_id();
    ite = user_counter.store_user_counter().find(key);
    if (ite != user_counter.store_user_counter().end()) {
      user_ad_feature.mutable_user_ad_count()->
          mutable_user_id_ad_package_name()->CopyFrom(ite->second);
    }

    key = "user_id#ad_package_category#" +
        model_request.user_id() + "#" + ad_info.category();
    ite = user_counter.store_user_counter().find(key);
    if (ite != user_counter.store_user_counter().end()) {
      user_ad_feature.mutable_user_ad_count()->
          mutable_user_id_ad_package_category()->CopyFrom(ite->second);
    }

    key = "user_id#pos_id#ad_id#" + model_request.user_id() +
        "#" + context.pos_id() + "#" + std::to_string(ad_info.ad_id());
    ite = user_counter.store_user_counter().find(key);
    if (ite != user_counter.store_user_counter().end()) {
      user_ad_feature.mutable_user_ad_count()->
          mutable_user_id_pos_id_ad_id()->CopyFrom(ite->second);
    }

    key = "user_id#pos_id#ad_package_name#" + model_request.user_id() +
        "#" + context.pos_id() + "#" + ad_info.app_id();
    ite = user_counter.store_user_counter().find(key);
    if (ite != user_counter.store_user_counter().end()) {
      user_ad_feature.mutable_user_ad_count()->
          mutable_user_id_pos_id_ad_package_name()->CopyFrom(ite->second);
    }

    key = "user_id#pos_id#ad_package_category#" + model_request.user_id() +
        "#" + context.pos_id() + "#" + ad_info.category();
    ite = user_counter.store_user_counter().find(key);
    if (ite != user_counter.store_user_counter().end()) {
      user_ad_feature.mutable_user_ad_count()->
          mutable_user_id_pos_id_ad_package_category()->CopyFrom(ite->second);
    }


    AdCount feature_ad_counter;
    key = "ad_id#" + std::to_string(ad_info.ad_id());
    auto ite_ad = ad_counter.store_ad_counter().find(key);
    if (ite_ad != ad_counter.store_ad_counter().end()) {
      feature_ad_counter.mutable_ad_id()->CopyFrom(ite_ad->second);
    }

    key = "package_name#ad_package_name#" + context.package_name() +
        "#" + ad_info.app_id();
    ite_ad = ad_counter.store_ad_counter().find(key);
    if (ite_ad != ad_counter.store_ad_counter().end()) {
      feature_ad_counter.mutable_ad_package_name()->CopyFrom(ite_ad->second);
    }

    key = "package_name#ad_package_category#" + context.package_name() +
        "#" + ad_info.category();
    ite_ad = ad_counter.store_ad_counter().find(key);
    if (ite_ad != ad_counter.store_ad_counter().end()) {
      feature_ad_counter.mutable_ad_package_category()->
          CopyFrom(ite_ad->second);
    }

    key = "pos_id#ad_id#" + context.pos_id() + "#" +
        std::to_string(ad_info.ad_id());
    ite_ad = ad_counter.store_ad_counter().find(key);
    if (ite_ad != ad_counter.store_ad_counter().end()) {
      feature_ad_counter.mutable_pos_id_ad_id()->CopyFrom(ite_ad->second);
    }

    key = "pos_id#ad_package_name#" + context.pos_id() + "#" + ad_info.app_id();
    ite_ad = ad_counter.store_ad_counter().find(key);
    if (ite_ad != ad_counter.store_ad_counter().end()) {
      feature_ad_counter.mutable_pos_id_ad_package_name()->
          CopyFrom(ite_ad->second);
    }

    key = "pos_id#ad_package_category#" + context.pos_id() +
        "#" + ad_info.category();
    ite_ad = ad_counter.store_ad_counter().find(key);
    if (ite_ad != ad_counter.store_ad_counter().end()) {
      feature_ad_counter.mutable_pos_id_ad_package_category()->
          CopyFrom(ite_ad->second);
    }

    for (int32_t j = 0; j < model_request.creatives(i).creative_size(); ++j) {
      ad_info.set_creative_id(
          model_request.creatives(i).creative(j).creative_id());
      ad_info.set_cp_id(model_request.creatives(i).creative(j).cp_id());

      key = "c_id#" + ad_info.creative_id();
      auto ite_info = store_ad_info.ad_infos().find(key);
      if (ite_info != store_ad_info.ad_infos().end()) {
        ad_info.set_creative_create_time(
            ite_info->second.creative_create_time());
      }

      Feature one_feature;
      one_feature.mutable_context()->CopyFrom(context);
      one_feature.mutable_user_profile()->CopyFrom(feature_user_profile);
      one_feature.mutable_ad_data()->mutable_ad_info()->CopyFrom(ad_info);
      one_feature.mutable_ad_data()->
          mutable_ad_counter()->CopyFrom(feature_ad_counter);
      one_feature.mutable_user_ad_feature()->CopyFrom(user_ad_feature);

      key = "user_id#c_id#" +
          model_request.user_id() + "#" + ad_info.creative_id();
      auto ite = user_counter.store_user_counter().find(key);
      if (ite != user_counter.store_user_counter().end()) {
        one_feature.mutable_user_ad_feature()->mutable_user_ad_count()->
            mutable_user_id_c_id()->CopyFrom(ite->second);
      }

      key = "user_id#pos_id#c_id#" + model_request.user_id() +
          "#" + context.pos_id() + "#" + ad_info.creative_id();
      ite = user_counter.store_user_counter().find(key);
      if (ite != user_counter.store_user_counter().end()) {
        one_feature.mutable_user_ad_feature()->mutable_user_ad_count()->
            mutable_user_id_pos_id_c_id()->CopyFrom(ite->second);
      }

      key = "package_name#c_id#" + context.package_name() +
          "#" + ad_info.creative_id();
      auto ite_ad = ad_counter.store_ad_counter().find(key);
      if (ite_ad != ad_counter.store_ad_counter().end()) {
        one_feature.mutable_ad_data()->mutable_ad_counter()->
            mutable_c_id()->CopyFrom(ite_ad->second);
      }

      key = "pos_id#c_id#" + context.pos_id() + "#" + ad_info.creative_id();
      ite_ad = ad_counter.store_ad_counter().find(key);
      if (ite_ad != ad_counter.store_ad_counter().end()) {
        one_feature.mutable_ad_data()->mutable_ad_counter()->
            mutable_pos_id_c_id()->CopyFrom(ite_ad->second);
      }

      feature.push_back(one_feature);
    }
  }
  return true;
}

}  // namespace ad

