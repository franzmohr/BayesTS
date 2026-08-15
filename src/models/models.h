// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef MODELS_H
#define MODELS_H

#include <hdf5.h>
#include <filesystem>
#include <memory>
#include <string>

#include "bayests/arma.h"

// Base class for all models
class BaseModel
{
public:
    virtual ~BaseModel() = default;
    virtual void draw_coefficients(const std::filesystem::path &filepath_arg) = 0;
    virtual void forecast(const std::filesystem::path& filepath) = 0;
    virtual void log_likelihood(const std::filesystem::path& filepath) = 0;
};

class VarNormalGamma : public BaseModel
{
private:
    // File path
    std::filesystem::path filepath;
public:
    VarNormalGamma();
    ~VarNormalGamma() override;
    void draw_coefficients(const std::filesystem::path &filepath_arg) override;
    void forecast(const std::filesystem::path& filepath) override;
    void log_likelihood(const std::filesystem::path& filepath) override;
};


class VarNormalStochvol : public BaseModel
{
private:
    // File path
    std::filesystem::path filepath;
public:
    VarNormalStochvol();
    ~VarNormalStochvol() override;
    void draw_coefficients(const std::filesystem::path &filepath_arg) override;
    void forecast(const std::filesystem::path& filepath) override;
    void log_likelihood(const std::filesystem::path& filepath) override;
};


class VarNormalWishart : public BaseModel
{
private:
    // File path
    std::filesystem::path filepath;
public:
    VarNormalWishart();
    ~VarNormalWishart() override;
    void draw_coefficients(const std::filesystem::path &filepath_arg) override;
    void forecast(const std::filesystem::path& filepath) override;
    void log_likelihood(const std::filesystem::path& filepath) override;
};


class VarTvpGamma : public BaseModel
{
private:
    // File path
    std::filesystem::path filepath;
public:
    VarTvpGamma();
    ~VarTvpGamma() override;
    void draw_coefficients(const std::filesystem::path &filepath_arg) override;
    void forecast(const std::filesystem::path& filepath) override;
    void log_likelihood(const std::filesystem::path& filepath) override;
};

class VarTvpWishart : public BaseModel
{
private:
    // File path
    std::filesystem::path filepath;
public:
    VarTvpWishart();
    ~VarTvpWishart() override;
    void draw_coefficients(const std::filesystem::path &filepath_arg) override;
    void forecast(const std::filesystem::path& filepath) override;
    void log_likelihood(const std::filesystem::path& filepath) override;
};

class VarTvpStochvol : public BaseModel
{
private:
    // File path
    std::filesystem::path filepath;
public:
    VarTvpStochvol();
    ~VarTvpStochvol() override;
    void draw_coefficients(const std::filesystem::path &filepath_arg) override;
    void forecast(const std::filesystem::path& filepath) override;
    void log_likelihood(const std::filesystem::path& filepath) override;
};

class VecNormalWishart : public BaseModel
{
private:
    // File path
    std::filesystem::path filepath;
public:
    VecNormalWishart();
    ~VecNormalWishart() override;
    void draw_coefficients(const std::filesystem::path &filepath_arg) override;
    void forecast(const std::filesystem::path& filepath) override;
    void log_likelihood(const std::filesystem::path& filepath) override;
};

// Factory function to create models based on type string
std::unique_ptr<BaseModel> create_model(const std::string& model_type);

#endif // MODELS_H
