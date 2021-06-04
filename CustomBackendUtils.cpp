//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomBackendUtils.hpp"

using namespace armnn;

const TensorInfo& GetTensorInfo(const ITensorHandle* tensorHandle)
{
    // We know that this workloads use CpuTensorHandles only, so this cast is legitimate.
    const ConstCpuTensorHandle* cpuTensorHandle =
        boost::polymorphic_downcast<const ConstCpuTensorHandle*>(tensorHandle);

    return cpuTensorHandle->GetTensorInfo();
}
