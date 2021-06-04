//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <test/CreateWorkload.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>

#include <custom/CustomWorkloadFactory.hpp>
#include <custom/workloads/CustomAdditionWorkload.hpp>
#include <custom/workloads/CustomPreCompiledWorkload.hpp>

#include <boost/cast.hpp>

namespace
{

template <typename Workload>
void CheckInputsOutput(std::unique_ptr<Workload> workload,
                       const TensorInfo&         inputInfo0,
                       const TensorInfo&         inputInfo1,
                       const TensorInfo&         outputInfo)
{
    auto queueDescriptor = workload->GetData();
    auto inputHandle0    = boost::polymorphic_downcast<ConstCpuTensorHandle*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle1    = boost::polymorphic_downcast<ConstCpuTensorHandle*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle    = boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST((inputHandle0->GetTensorInfo() == inputInfo0));
    BOOST_TEST((inputHandle1->GetTensorInfo() == inputInfo1));
    BOOST_TEST((outputHandle->GetTensorInfo() == outputInfo));
}

} // Anonymous namespace

void CustomCreateAdditionWorkloadTest()
{
    Graph graph;
    CustomWorkloadFactory factory;

    // Create the addition layer
    Layer* const layer = graph.AddLayer<AdditionLayer>("addition");

    // Create the extra layers (inputs and output)
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const input2 = graph.AddLayer<InputLayer>(2, "input2");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect everything up
    armnn::TensorInfo tensorInfo({2, 3}, DataType::Float32);
    Connect(input1, layer, tensorInfo, 0, 0);
    Connect(input2, layer, tensorInfo, 0, 1);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it
    auto workload = MakeAndCheckWorkload<CustomAdditionWorkload>(*layer, factory);

    AdditionQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 2);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    CheckInputsOutput(std::move(workload),
                      TensorInfo({ 2, 3 }, DataType::Float32),
                      TensorInfo({ 2, 3 }, DataType::Float32),
                      TensorInfo({ 2, 3 }, DataType::Float32));
}

void CustomCreatePreCompiledWorkloadTest()
{
    Graph graph;
    CustomWorkloadFactory factory;

    // Create the pre-compiled layer
    PreCompiledDescriptor descriptor(2, 1);
    Layer* const layer = graph.AddLayer<PreCompiledLayer>(descriptor, "pre-compiled");

    // Defining a simple deleter for the mock pre-compiled object
    PreCompiledObjectDeleter customPrecompiledObjectDeleter = [](const void* data)
    {
        if (!data)
        {
            return;
        }

        const CustomPreCompiledObject* p = static_cast<const CustomPreCompiledObject*>(data);
        delete p;
    };

    // Mocking a pre-compiled object (the result of the optimization process done by the backend)
    PreCompiledObjectPtr customPrecompiledObject(new CustomPreCompiledObject(), customPrecompiledObjectDeleter);

    PreCompiledLayer* preCompiledLayer = boost::polymorphic_downcast<PreCompiledLayer*>(layer);
    preCompiledLayer->SetPreCompiledObject(std::move(customPrecompiledObject));

    // Create the extra layers (inputs and output)
    Layer* const input1 = graph.AddLayer<InputLayer>(1, "input1");
    Layer* const input2 = graph.AddLayer<InputLayer>(2, "input2");
    Layer* const output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect everything up
    armnn::TensorInfo tensorInfo({2, 3}, DataType::Float32);
    Connect(input1, layer, tensorInfo, 0, 0);
    Connect(input2, layer, tensorInfo, 0, 1);
    Connect(layer, output, tensorInfo);
    CreateTensorHandles(graph, factory);

    // Makes the workload and checks it
    auto workload = MakeAndCheckWorkload<CustomPreCompiledWorkload>(*layer, graph, factory);

    PreCompiledQueueDescriptor queueDescriptor = workload->GetData();
    BOOST_TEST(queueDescriptor.m_Inputs.size() == 2);
    BOOST_TEST(queueDescriptor.m_Outputs.size() == 1);

    CheckInputsOutput(std::move(workload),
                      TensorInfo({ 2, 3 }, DataType::Float32),
                      TensorInfo({ 2, 3 }, DataType::Float32),
                      TensorInfo({ 2, 3 }, DataType::Float32));
}

BOOST_AUTO_TEST_SUITE(CreateWorkloadCustom)

BOOST_AUTO_TEST_CASE(CreateAdditionWorkload)
{
    CustomCreateAdditionWorkloadTest();
}

BOOST_AUTO_TEST_CASE(CreatePreCompiledWorkload)
{
    CustomCreatePreCompiledWorkloadTest();
}

BOOST_AUTO_TEST_SUITE_END()
