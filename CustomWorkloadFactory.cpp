//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomWorkloadFactory.hpp"

#include "workloads/CustomAdditionWorkload.hpp"
#include "workloads/CustomPreCompiledWorkload.hpp"

#include <backendsCommon/TensorHandle.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>

#include <Layer.hpp>

#include <boost/log/trivial.hpp>

namespace armnn
{

namespace
{

static const BackendId s_Id{"Custom"};

} // Anonymous namespace

CustomWorkloadFactory::CustomWorkloadFactory()
{
}

const BackendId& CustomWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

bool CustomWorkloadFactory::IsLayerSupported(const Layer& layer,
                                             Optional<DataType> dataType,
                                             std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

std::unique_ptr<ITensorHandle> CustomWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo, bool) const
{
    return std::make_unique<ScopedTensorHandle>(tensorInfo);
}

std::unique_ptr<ITensorHandle> CustomWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                         DataLayout dataLayout,
						                                                 bool isMemoryManaged) const
{
    IgnoreUnused(dataLayout);
    IgnoreUnused(isMemoryManaged);
    return std::make_unique<ScopedTensorHandle>(tensorInfo);
}

std::unique_ptr<IWorkload> CustomWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos.empty() )
    {
        throw InvalidArgumentException("CustomWorkloadFactory::CreateInput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty())
    {
        throw InvalidArgumentException("CustomWorkloadFactory::CreateInput: Output cannot be zero length");
    }

    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
    {
        throw InvalidArgumentException("CustomWorkloadFactory::CreateInput: "
                                       "data input and output differ in byte count.");
    }

    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> CustomWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos.empty() )
    {
        throw InvalidArgumentException("CustomWorkloadFactory::CreateOutput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty())
    {
        throw InvalidArgumentException("CustomWorkloadFactory::CreateOutput: Output cannot be zero length");
    }
    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
    {
        throw InvalidArgumentException("CustomWorkloadFactory::CreateOutput: "
                                       "data input and output differ in byte count.");
    }

    //return MakeWorkloadHelper<CopyMemGenericWorkload>(descriptor, info);
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> CustomWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                        const WorkloadInfo& info) const
{
    return std::make_unique<CustomAdditionWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> CustomWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return std::make_unique<CustomPreCompiledWorkload>(descriptor, info);
}

} // namespace armnn
