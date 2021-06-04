//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

class CustomAdditionWorkload : public BaseWorkload<AdditionQueueDescriptor>
{
public:
    CustomAdditionWorkload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;
};

} // namespace armnn
