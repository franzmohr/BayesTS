// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "models.h"
#include <algorithm>
#include <unordered_map>
#include <functional>

namespace {

const std::unordered_map<std::string, std::function<std::unique_ptr<BaseModel>()>> &factory_map() {
	static const std::unordered_map<std::string, std::function<std::unique_ptr<BaseModel>()>> factory = {
		{"DfmNormalGamma", []() { return std::make_unique<DfmNormalGamma>(); }},
		{"DfmNormalStochvol", []() { return std::make_unique<DfmNormalStochvol>(); }},
		{"DfmTvpGamma", []() { return std::make_unique<DfmTvpGamma>(); }},
		{"DfmTvpStochvol", []() { return std::make_unique<DfmTvpStochvol>(); }},
		{"FavarNormalWishart", []() { return std::make_unique<FavarNormalWishart>(); }},
		{"VarNormalGamma", []() { return std::make_unique<VarNormalGamma>(); }},
		{"VarNormalAld", []() { return std::make_unique<VarNormalAld>(); }},
{"VarNormalStochvol", []() { return std::make_unique<VarNormalStochvol>(); }},
		{"VarNormalWishart", []() { return std::make_unique<VarNormalWishart>(); }},
		{"VarTvpDiscount", []() { return std::make_unique<VarTvpDiscount>(); }},
		{"VarTvpGamma", []() { return std::make_unique<VarTvpGamma>(); }},
		{"VarTvpWishart", []() { return std::make_unique<VarTvpWishart>(); }},
		{"VarTvpAld", []() { return std::make_unique<VarTvpAld>(); }},
{"VarTvpStochvol", []() { return std::make_unique<VarTvpStochvol>(); }},
		{"VecKlgs2010", []() { return std::make_unique<VecKlgs2010>(); }},
		{"VecNormalGamma", []() { return std::make_unique<VecNormalGamma>(); }},
		{"VecNormalStochvol", []() { return std::make_unique<VecNormalStochvol>(); }},
		{"VecNormalWishart", []() { return std::make_unique<VecNormalWishart>(); }},
		{"VecTvpDiscount", []() { return std::make_unique<VecTvpDiscount>(); }},
		{"VecTvpGamma", []() { return std::make_unique<VecTvpGamma>(); }},
		{"VecTvpStochvol", []() { return std::make_unique<VecTvpStochvol>(); }},
		{"VecTvpWishart", []() { return std::make_unique<VecTvpWishart>(); }}
	};
	return factory;
}

} // namespace

std::unique_ptr<BaseModel> create_model(const std::string& model_type) {
	const auto &factory = factory_map();

	auto it = factory.find(model_type);
	if (it != factory.end()) {
		return it->second();
	}

	throw std::runtime_error("Unknown model: " + model_type);
}

std::vector<std::string> registered_models() {
	std::vector<std::string> names;
	for (const auto &entry : factory_map()) {
		names.push_back(entry.first);
	}
	std::sort(names.begin(), names.end());
	return names;
}
