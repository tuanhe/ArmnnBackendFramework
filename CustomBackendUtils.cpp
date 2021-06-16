//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomBackendUtils.hpp"

using namespace armnn;

const TensorInfo& GetTensorInfo(const ITensorHandle* tensorHandle)
{
    // We know that this workloads use CpuTensorHandles only, so this cast is legitimate.
    const ConstTensorHandle* cpuTensorHandle =
        boost::polymorphic_downcast<const ConstTensorHandle*>(tensorHandle);

    return cpuTensorHandle->GetTensorInfo();
}
