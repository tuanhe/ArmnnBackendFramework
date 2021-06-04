//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ArmNN.hpp>

#include <Graph.hpp>
#include <Network.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>

std::vector<armnn::BackendId> defaultBackends = { "Custom" };

void AdditionToPreCompiledTest()
{
    // This test is designed to match the "AddTwo" test in android nn/runtime/test/TestTrivialModel.cpp

    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input1 = net->AddInputLayer(0);
    IConnectableLayer* input2 = net->AddInputLayer(1);
    IConnectableLayer* add    = net->AddAdditionLayer();
    IConnectableLayer* output = net->AddOutputLayer(0);

    input1->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input2->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Set the tensors in the network
    TensorInfo tensorInfo(TensorShape({3, 4}), DataType::Float32);
    input1->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    input2->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    add->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // Optimize the network
    IOptimizedNetworkPtr optimizedNet = Optimize(*net, defaultBackends, runtime->GetDeviceSpec());
    BOOST_CHECK(optimizedNet != nullptr);

    // Check that the addition layer has been substituted with a pre-compiled layer in the optimized graph
    Graph& optimizedGraph = static_cast<OptimizedNetwork*>(optimizedNet.get())->GetGraph();
    Layer* additionLayer = nullptr;
    Layer* preCompiledLayer = nullptr;
    for (auto& layer : optimizedGraph)
    {
        if (layer->GetType() == LayerType::Addition)
        {
            additionLayer = layer;
        }
        if (layer->GetType() == LayerType::PreCompiled)
        {
            preCompiledLayer = layer;
        }
    }
    BOOST_CHECK(additionLayer == nullptr);
    BOOST_CHECK(preCompiledLayer != nullptr);

    // Load it into the runtime
    NetworkId networkId;
    runtime->LoadNetwork(networkId, std::move(optimizedNet));

    // Create structures for input & output (matching the android NN test)
    std::vector<float> input1Data
    {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };
    std::vector<float> input2Data
    {
        100.f, 200.f, 300.f, 400.f, 500.f, 600.f, 700.f, 800.f, 900.f, 1000.f, 1100.f, 1200.f
    };
    std::vector<float> outputData(12);

    InputTensors inputTensors
    {
        { 0, ConstTensor(runtime->GetInputTensorInfo(networkId, 0), input1Data.data()) },
        { 1, ConstTensor(runtime->GetInputTensorInfo(networkId, 0), input2Data.data()) }
    };
    OutputTensors outputTensors
    {
        { 0, Tensor(runtime->GetOutputTensorInfo(networkId, 0), outputData.data()) }
    };

    // Do the inference
    runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);

    // Check the results
    BOOST_TEST(outputData[0] == 101);
    BOOST_TEST(outputData[1] == 202);
    BOOST_TEST(outputData[2] == 303);
    BOOST_TEST(outputData[3] == 404);
    BOOST_TEST(outputData[4] == 505);
    BOOST_TEST(outputData[5] == 606);
    BOOST_TEST(outputData[6] == 707);
    BOOST_TEST(outputData[7] == 808);
    BOOST_TEST(outputData[8] == 909);
    BOOST_TEST(outputData[9] == 1010);
    BOOST_TEST(outputData[10] == 1111);
    BOOST_TEST(outputData[11] == 1212);
}

BOOST_AUTO_TEST_SUITE(CustomEndToEnd)

BOOST_AUTO_TEST_CASE(AdditionToPreCompiled)
{
    AdditionToPreCompiledTest();
}

BOOST_AUTO_TEST_SUITE_END()
