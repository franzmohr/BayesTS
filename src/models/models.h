// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef MODELS_H
#define MODELS_H

#include <hdf5.h>
#include <filesystem>
#include <memory>
#include <string>

#include "bayests/arma.h"

/// Where a model is: the file, and the group inside it that the model's tree
/// hangs under.
///
/// An empty group is the root of the file, which is where a file holding one
/// model puts it and what every command line without --group means. A file that
/// holds several models keeps each under a group of its own, and then the group
/// is the other half of the address -- neither the file nor the group names a
/// model on its own.
struct ModelLocation
{
    std::filesystem::path file;
    std::string group;

    /// For messages: "model.h5", or "model.h5:/models/3" for a model that does
    /// not sit at the root. A bare filename would not say which of several
    /// models in one file failed.
    std::string describe() const
    {
        return group.empty() ? file.string() : file.string() + ":" + group;
    }
};

// Base class for all models
class BaseModel
{
public:
    virtual ~BaseModel() = default;
    virtual void draw_coefficients(const ModelLocation &location_arg) = 0;
    virtual void forecast(const ModelLocation &location_arg) = 0;
    virtual void log_likelihood(const ModelLocation &location_arg) = 0;
};

class VarNormalGamma : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VarNormalGamma();
    ~VarNormalGamma() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};


class VarNormalAld : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VarNormalAld();
    ~VarNormalAld() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VarNormalStochvol : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VarNormalStochvol();
    ~VarNormalStochvol() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};


class VarNormalWishart : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VarNormalWishart();
    ~VarNormalWishart() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};


class VarTvpGamma : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VarTvpGamma();
    ~VarTvpGamma() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VarTvpWishart : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VarTvpWishart();
    ~VarTvpWishart() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VarTvpAld : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VarTvpAld();
    ~VarTvpAld() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VarTvpStochvol : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VarTvpStochvol();
    ~VarTvpStochvol() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class DfmNormalGamma : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    DfmNormalGamma();
    ~DfmNormalGamma() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class DfmNormalStochvol : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    DfmNormalStochvol();
    ~DfmNormalStochvol() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class DfmTvpGamma : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    DfmTvpGamma();
    ~DfmTvpGamma() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class DfmTvpStochvol : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    DfmTvpStochvol();
    ~DfmTvpStochvol() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VecNormalWishart : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VecNormalWishart();
    ~VecNormalWishart() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VecKlgs2010 : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VecKlgs2010();
    ~VecKlgs2010() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VecNormalGamma : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VecNormalGamma();
    ~VecNormalGamma() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VecNormalStochvol : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VecNormalStochvol();
    ~VecNormalStochvol() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VecTvpGamma : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VecTvpGamma();
    ~VecTvpGamma() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VecTvpWishart : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VecTvpWishart();
    ~VecTvpWishart() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class VecTvpStochvol : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    VecTvpStochvol();
    ~VecTvpStochvol() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

class FavarNormalWishart : public BaseModel
{
private:
    // Where the model is: file plus the group its tree hangs under
    ModelLocation location;
public:
    FavarNormalWishart();
    ~FavarNormalWishart() override;
    void draw_coefficients(const ModelLocation &location_arg) override;
    void forecast(const ModelLocation &location_arg) override;
    void log_likelihood(const ModelLocation &location_arg) override;
};

// Factory function to create models based on type string
std::unique_ptr<BaseModel> create_model(const std::string& model_type);

#endif // MODELS_H
