//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <custom/CustomPreCompiledObject.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

class CustomPreCompiledWorkload : public BaseWorkload<PreCompiledQueueDescriptor>
{
public:
    CustomPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    const CustomPreCompiledObject* m_PreCompiledObject;
};

} // namespace armnn
