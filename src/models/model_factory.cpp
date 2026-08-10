// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "models.h"
#include <unordered_map>
#include <functional>

std::unique_ptr<BaseModel> create_model(const std::string& model_type) {
	static const std::unordered_map<std::string, std::function<std::unique_ptr<BaseModel>()>> factory = {
		{"VarNormalGamma", []() { return std::make_unique<VarNormalGamma>(); }},
		{"VarNormalStochvol", []() { return std::make_unique<VarNormalStochvol>(); }},
		{"VarNormalWishart", []() { return std::make_unique<VarNormalWishart>(); }},
		{"VarTvpGamma", []() { return std::make_unique<VarTvpGamma>(); }},
		{"VarTvpWishart", []() { return std::make_unique<VarTvpWishart>(); }},
		{"VarTvpStochvol", []() { return std::make_unique<VarTvpStochvol>(); }}
	};

	auto it = factory.find(model_type);
	if (it != factory.end()) {
		return it->second();
	}

	throw std::runtime_error("Unknown model: " + model_type);
}
