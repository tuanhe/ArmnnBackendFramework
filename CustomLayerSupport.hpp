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

    bool IsMemCopySupported(const armnn::TensorInfo &input,
                            const armnn::TensorInfo &output,
                            armnn::Optional<std::string &> reasonIfUnsupported) const override;
   
    bool IsConvolution2dSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Convolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const Optional<TensorInfo>& biases,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                         const TensorInfo& weights,
                                         const Optional<TensorInfo>& biases,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsTransposeConvolution2dSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const TransposeConvolution2dDescriptor& descriptor,
                                           const TensorInfo& weights,
                                           const Optional<TensorInfo>& biases,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsFullyConnectedSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& weights,
                                   const TensorInfo& biases,
                                   const FullyConnectedDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsOutputSupported(const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsPooling2dSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling2dDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsPreCompiledSupported(const TensorInfo& input,
                                const PreCompiledDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;
    
    bool IsReshapeSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const ReshapeDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsSoftmaxSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const SoftmaxDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

};

} // namespace armnn
