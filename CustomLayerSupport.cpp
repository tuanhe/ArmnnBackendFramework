//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CustomLayerSupport.hpp"

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/Types.hpp>

#include <boost/core/ignore_unused.hpp>

using namespace boost;

namespace armnn
{

bool IsDataTypeSupported(DataType dataType, Optional<std::string&> reasonIfUnsupported)
{
    if (dataType == DataType::QAsymmU8)
    {
        return true;
    }
    else
    {
        reasonIfUnsupported.value() = "Data type not supported.";
        return false;
    }
}

bool CustomLayerSupport::IsActivationSupported(const TensorInfo& input,
                           const TensorInfo& output,
                           const ActivationDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsDataTypeSupported(input0.GetDataType(), reasonIfUnsupported);
}

bool CustomLayerSupport::IsArgMinMaxSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          const ArgMinMaxDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& mean,
                                   const TensorInfo& var,
                                   const TensorInfo& beta,
                                   const TensorInfo& gamma,
                                   const BatchNormalizationDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(mean);
    ignore_unused(var);
    ignore_unused(beta);
    ignore_unused(gamma);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsBatchToSpaceNdSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const BatchToSpaceNdDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsCastSupported(const TensorInfo& input,
                     const TensorInfo& output,
                     Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsComparisonSupported(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           const ComparisonDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                       const TensorInfo& output,
                       const OriginsDescriptor& descriptor,
                       Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(inputs);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsConstantSupported(const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsConvertBf16ToFp32Supported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsConvertFp32ToBf16Supported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsConvertFp32ToFp16Supported(
        const TensorInfo& input,
        const TensorInfo& output,
        Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const Convolution2dDescriptor& descriptor,
                              const TensorInfo& weights,
                              const Optional<TensorInfo>& biases,
                              Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(weights);
    ignore_unused(biases);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsDebugSupported(const TensorInfo& input,
                      const TensorInfo& output,
                      Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsDepthToSpaceSupported(const TensorInfo& input,
                             const TensorInfo& output,
                             const DepthToSpaceDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                     const TensorInfo& weights,
                                     const Optional<TensorInfo>& biases,
                                     Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(weights);
    ignore_unused(biases);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsDequantizeSupported(const TensorInfo& input,
                           const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsDetectionPostProcessSupported(const TensorInfo& boxEncodings,
                                     const TensorInfo& scores,
                                     const TensorInfo& anchors,
                                     const TensorInfo& detectionBoxes,
                                     const TensorInfo& detectionClasses,
                                     const TensorInfo& detectionScores,
                                     const TensorInfo& numDetections,
                                     const DetectionPostProcessDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(boxEncodings);
    ignore_unused(scores);
    ignore_unused(anchors);
    ignore_unused(detectionBoxes);
    ignore_unused(detectionClasses);
    ignore_unused(detectionScores);
    ignore_unused(numDetections);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsDilatedDepthwiseConvolutionSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const DepthwiseConvolution2dDescriptor& descriptor,
                                            const TensorInfo& weights,
                                            const Optional<TensorInfo>& biases,
                                            Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(weights);
    ignore_unused(biases);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsElementwiseUnarySupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const ElementwiseUnaryDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                 const FakeQuantizationDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsFillSupported(const TensorInfo& input,
                             const TensorInfo& output,
                             const FillDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsFloorSupported(const TensorInfo& input,
                      const TensorInfo& output,
                      Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const TensorInfo& weights,
                               const TensorInfo& biases,
                               const FullyConnectedDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(weights);
    ignore_unused(descriptor);
    ignore_unused(biases);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsGatherSupported(const TensorInfo& input0,
                       const TensorInfo& input1,
                       const TensorInfo& output,
                       const GatherDescriptor& descriptor,
                       Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsInputSupported(const TensorInfo& input,
                      Optional<std::string&> reasonIfUnsupported) const 
{
    return IsDataTypeSupported(input.GetDataType(), reasonIfUnsupported);
}

bool CustomLayerSupport::IsInstanceNormalizationSupported(
    const TensorInfo& input,
    const TensorInfo& output,
    const InstanceNormalizationDescriptor& descriptor,
    Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                const TensorInfo& output,
                                const L2NormalizationDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsLogicalBinarySupported(const TensorInfo& input0,
                              const TensorInfo& input1,
                              const TensorInfo& output,
                              const LogicalBinaryDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport:: IsLogicalUnarySupported(const TensorInfo& input,
                             const TensorInfo& output,
                             const ElementwiseUnaryDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsLogSoftmaxSupported(const TensorInfo& input,
                           const TensorInfo& output,
                           const LogSoftmaxDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsLstmSupported(const TensorInfo& input,
                     const TensorInfo& outputStateIn,
                     const TensorInfo& cellStateIn,
                     const TensorInfo& scratchBuffer,
                     const TensorInfo& outputStateOut,
                     const TensorInfo& cellStateOut,
                     const TensorInfo& output,
                     const LstmDescriptor& descriptor,
                     const LstmInputParamsInfo& paramsInfo,
                     Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(outputStateIn);
    ignore_unused(cellStateIn);
    ignore_unused(scratchBuffer);
    ignore_unused(outputStateOut);
    ignore_unused(cellStateOut);
    ignore_unused(paramsInfo);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsMaximumSupported(const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsMeanSupported(const TensorInfo& input,
                     const TensorInfo& output,
                     const MeanDescriptor& descriptor,
                     Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsMemCopySupported(const TensorInfo& input,
                        const TensorInfo& output,
                        Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(output);
    return IsDataTypeSupported(input.GetDataType(), reasonIfUnsupported);
}

bool CustomLayerSupport::IsMemImportSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsMergeSupported(const TensorInfo& input0,
                      const TensorInfo& input1,
                      const TensorInfo& output,
                      Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsMinimumSupported(const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const NormalizationDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsOutputSupported(const TensorInfo& output,
                       Optional<std::string&> reasonIfUnsupported) const 

{
    return IsDataTypeSupported(output.GetDataType(), reasonIfUnsupported);
}
bool CustomLayerSupport::IsPadSupported(const TensorInfo& input,
                    const TensorInfo& output,
                    const PadDescriptor& descriptor,
                    Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsPermuteSupported(const TensorInfo& input,
                        const TensorInfo& output,
                        const PermuteDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          const Pooling2dDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsPreCompiledSupported(const TensorInfo& input,
                            const PreCompiledDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(descriptor);
    return IsDataTypeSupported(input.GetDataType(), reasonIfUnsupported);
}

bool CustomLayerSupport::IsPreluSupported(const TensorInfo& input,
                      const TensorInfo& alpha,
                      const TensorInfo& output,
                      Optional<std::string &> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(alpha);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsQuantizeSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported) const 
{

    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsQLstmSupported(const TensorInfo& input,
                      const TensorInfo& previousOutputIn,
                      const TensorInfo& previousCellStateIn,
                      const TensorInfo& outputStateOut,
                      const TensorInfo& cellStateOut,
                      const TensorInfo& output,
                      const QLstmDescriptor& descriptor,
                      const LstmInputParamsInfo& paramsInfo,
                      Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(previousOutputIn);
    ignore_unused(previousCellStateIn);
    ignore_unused(outputStateOut);
    ignore_unused(cellStateOut);
    ignore_unused(paramsInfo);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsQuantizedLstmSupported(const TensorInfo& input,
                              const TensorInfo& previousCellStateIn,
                              const TensorInfo& previousOutputIn,
                              const TensorInfo& cellStateOut,
                              const TensorInfo& output,
                              const QuantizedLstmInputParamsInfo& paramsInfo,
                              Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(previousOutputIn);
    ignore_unused(previousCellStateIn);
    ignore_unused(cellStateOut);
    ignore_unused(paramsInfo);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);    
    return true;
}

bool CustomLayerSupport::IsRankSupported(const TensorInfo& input,
                     const TensorInfo& output,
                     Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsReduceSupported(const TensorInfo& input,
                       const TensorInfo& output,
                       const ReduceDescriptor& descriptor,
                       Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsReshapeSupported(const TensorInfo& input,
                        const TensorInfo& output,
                        const ReshapeDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsResizeSupported(const TensorInfo& input,
                       const TensorInfo& output,
                       const ResizeDescriptor& descriptor,
                       Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsSliceSupported(const TensorInfo& input,
                      const TensorInfo& output,
                      const SliceDescriptor& descriptor,
                      Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                        const TensorInfo& output,
                        const SoftmaxDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsSpaceToBatchNdSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const SpaceToBatchNdDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsSpaceToDepthSupported(const TensorInfo& input,
                             const TensorInfo& output,
                             const SpaceToDepthDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsSplitterSupported(const TensorInfo& input,
                         const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                         const ViewsDescriptor& descriptor,
                         Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(outputs);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                      const TensorInfo& output,
                      const StackDescriptor& descriptor,
                      Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(inputs);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsStandInSupported(const std::vector<const TensorInfo*>& inputs,
                        const std::vector<const TensorInfo*>& outputs,
                        const StandInDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(inputs);
    ignore_unused(outputs);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsStridedSliceSupported(const TensorInfo& input,
                             const TensorInfo& output,
                             const StridedSliceDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsSwitchSupported(const TensorInfo& input0,
                       const TensorInfo& input1,
                       const TensorInfo& output0,
                       const TensorInfo& output1,
                       Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output0);
    ignore_unused(output1);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsTransposeConvolution2dSupported(
    const TensorInfo& input,
    const TensorInfo& output,
    const TransposeConvolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const Optional<TensorInfo>& biases,
    Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(weights);
    ignore_unused(biases);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool CustomLayerSupport::IsTransposeSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          const TransposeDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported) const 
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return true;
}


} // namespace armnn
