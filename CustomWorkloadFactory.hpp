//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Optional.hpp>

#include <backendsCommon/WorkloadFactory.hpp>

#include <boost/core/ignore_unused.hpp>

namespace armnn
{

class CustomWorkloadFactory : public IWorkloadFactory
{
public:
    explicit CustomWorkloadFactory();
    ~CustomWorkloadFactory() {}

    const BackendId& GetBackendId() const override;

    static bool IsLayerSupported(const Layer& layer,
                                 Optional<DataType> dataType,
                                 std::string& outReasonIfUnsupported);

    bool SupportsSubTensors() const override { return false; }

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override
    {
        boost::ignore_unused(parent, subTensorShape, subTensorOrigin);
        return nullptr;
    }

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
		                                      bool isMemoryManaged) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
						      bool isMemoryManaged) const override;

    std::unique_ptr<IWorkload> CreateInput(const InputQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    std::unique_ptr<IWorkload> CreateOutput(const OutputQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const override;

    std::unique_ptr<IWorkload> CreateAddition(const AdditionQueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const override;

    std::unique_ptr<IWorkload> CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info) const override;
};

} // namespace armnn
