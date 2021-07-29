//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomBackend.hpp"

#include <armnn/BackendRegistry.hpp>

namespace
{

using namespace armnn;

static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    CustomBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new CustomBackend());
    }
};


} // Anonymous namespace
