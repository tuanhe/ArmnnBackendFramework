//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/LayerSupportBase.hpp>

namespace armnn
{

class CustomLayerSupport : public LayerSupportBase
{
public:
    bool IsAdditionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsInputSupported(const TensorInfo& input,
                          Optional<std::string&> reasonIfUnsupported) const override;

    bool IsOutputSupported(const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported) const override;

    bool IsMemCopySupported(const armnn::TensorInfo &input,
                            const armnn::TensorInfo &output,
                            armnn::Optional<std::string &> reasonIfUnsupported) const override;

    bool IsPreCompiledSupported(const armnn::TensorInfo &input,
                                const armnn::PreCompiledDescriptor &descriptor,
                                armnn::Optional<std::string &> reasonIfUnsupported) const override;
};

} // namespace armnn
