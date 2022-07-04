/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Beginner example for the C++ interface.
//
// This program do the following:
//   - Scan the dataset columns to create a dataspec.
//   - Print a human readable report of the dataspec.
//   - Train a Random Forest model.
//   - Export the model to disk.
//   - Print and export a description of the model (meta-data and structure).
//   - Evaluate the model on a test dataset.
//   - Instantiate a serving engine with the model.
//   - Run a couple of predictions with the serving engine.
//   - Starts a http server
//   - Runs a prediction when a http request is received
//   - Returns the prediction result in the http response
//
// Most of the sections are equivalent as calling one of the CLI command. This
// is indicated in the comments. For example, the comment "Same as
// :infer_dataspec" indicates that the following section is equivalent as
// running the "infer_dataspec" CLI command.
//
// When converting a CLI pipeline in C++, it is also interesting to look at the
// implementation of each CLI command. Generally, one CLI command contains one
// or a small number of C++ calls.
//
// Usage example:
//  ./compile_and_run.sh
//

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <boost/json/src.hpp>

namespace json = boost::json;     // from <boost/json/src.hpp>
namespace beast = boost::beast;   // from <boost/beast.hpp>
namespace http = beast::http;     // from <boost/beast/http.hpp>
namespace net = boost::asio;      // from <boost/asio.hpp>
namespace ygg = yggdrasil_decision_forests;

using tcp = boost::asio::ip::tcp; // from <boost/asio/ip/tcp.hpp>

const std::string SERVER_ADDRESS = "0.0.0.0";
const unsigned short SERVER_PORT = 8081;

ABSL_FLAG(std::string, dataset_dir,
          "yggdrasil_decision_forests/test_data/dataset",
          "Directory containing the \"adult_train.csv\" and \"adult_test.csv\" "
          "datasets.");

ABSL_FLAG(std::string, output_dir, "/tmp/yggdrasil_decision_forest",
          "Output directory for the model and evaluation");

struct serving_data
{
    std::unique_ptr<ygg::serving::FastEngine> serving_engine_;
    std::unique_ptr<ygg::serving::FeaturesDefinitionNumericalOrCategoricalFlat> features_;
    std::unique_ptr<ygg::serving::FeaturesDefinitionNumericalOrCategoricalFlat::NumericalFeatureId> age_feature_;
    std::unique_ptr<ygg::serving::FeaturesDefinitionNumericalOrCategoricalFlat::CategoricalFeatureId> education_feature_;

    serving_data(std::unique_ptr<ygg::serving::FastEngine> serving_engine,
                 std::unique_ptr<ygg::serving::FeaturesDefinitionNumericalOrCategoricalFlat> features,
                 std::unique_ptr<ygg::serving::FeaturesDefinitionNumericalOrCategoricalFlat::NumericalFeatureId> age_feature,
                 std::unique_ptr<ygg::serving::FeaturesDefinitionNumericalOrCategoricalFlat::CategoricalFeatureId> education_feature)
        : serving_engine_(std::move(serving_engine)),
          features_(std::move(features)),
          age_feature_(std::move(age_feature)),
          education_feature_(std::move(education_feature)) {}
};

static bool endsWith(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

class http_connection : public std::enable_shared_from_this<http_connection>
{
public:
    http_connection(tcp::socket socket, std::unordered_map<std::string, serving_data> &model_store)
        : socket_(std::move(socket)), model_store_(&model_store) {}

    // Initiate the asynchronous operations associated with the connection.
    void
    start()
    {
        read_request();
        check_deadline();
    }

private:
    std::unordered_map<std::string, serving_data> *model_store_;

    // The socket for the currently connected client.
    tcp::socket socket_;

    // The buffer for performing reads.
    beast::flat_buffer buffer_{8192};

    // The request message.
    http::request<http::dynamic_body> request_;

    // The response message.
    http::response<http::dynamic_body> response_;

    // The timer for putting a deadline on connection processing.
    net::steady_timer deadline_{
        socket_.get_executor(), std::chrono::seconds(60)};

    // Asynchronously receive a complete request message.
    void
    read_request()
    {
        auto self = shared_from_this();

        http::async_read(
            socket_,
            buffer_,
            request_,
            [self](beast::error_code ec,
                   std::size_t bytes_transferred)
            {
                boost::ignore_unused(bytes_transferred);
                if (!ec)
                    self->process_request();
            });
    }

    // Determine what needs to be done with the request message.
    void
    process_request()
    {
        response_.version(request_.version());
        response_.keep_alive(false);

        switch (request_.method())
        {
        case http::verb::post:
            response_.result(http::status::ok);
            response_.set(http::field::server, "Beast");
            create_response(beast::buffers_to_string(request_.body().data()));
            break;
        default:
            // We return responses indicating an error if
            // we do not recognize the request method.
            response_.result(http::status::bad_request);
            response_.set(http::field::content_type, "application/json");
            json::object obj;
            obj["error"] = std::string("Invalid request-method '") + std::string(request_.method_string()) + std::string("'");
            beast::ostream(response_.body()) << json::serialize(obj);
            break;
        }

        write_response();
    }

    // Construct a response message based on the program state.
    void
    create_response(std::string req)
    {
        if (request_.target() == "/predict")
        {
            const auto model_path = file::JoinPath(absl::GetFlag(FLAGS_output_dir), "model");
            auto parsed_data = json::parse(req);
            std::string model_id = json::value_to<std::string>(parsed_data.at("model_id"));

            if (!endsWith(model_path, model_id))
            {
                response_.set(http::field::content_type, "application/json");
                json::object obj;
                obj["error"] = std::string("Model not found: ") + model_id;
                beast::ostream(response_.body()) << json::serialize(obj);
                return;
            }

            double age = json::value_to<double>(parsed_data.at("age"));
            std::string education = json::value_to<std::string>(parsed_data.at("education"));
            serving_data *serving_data_;

            auto it = model_store_->find(model_id);
            if (it == model_store_->end())
            {
                std::unique_ptr<ygg::model::AbstractModel> model;
                QCHECK_OK(ygg::model::LoadModel(model_path, &model));

                auto serving_engine = model->BuildFastEngine().value();
                auto features = std::make_unique<ygg::serving::FeaturesDefinitionNumericalOrCategoricalFlat>(serving_engine->features());

                // Handle to two features.
                auto age_feature = std::make_unique<ygg::serving::FeaturesDefinitionNumericalOrCategoricalFlat::NumericalFeatureId>(features.get()->GetNumericalFeatureId("age").value());
                auto education_feature = std::make_unique<ygg::serving::FeaturesDefinitionNumericalOrCategoricalFlat::CategoricalFeatureId>(features.get()->GetCategoricalFeatureId("education").value());

                serving_data_ = &model_store_->insert(std::pair<std::string, serving_data>(model_id, serving_data(std::move(serving_engine),
                                                                                                                  std::move(features),
                                                                                                                  std::move(age_feature),
                                                                                                                  std::move(education_feature))))
                                     .first->second;
            }
            else
            {
                serving_data_ = &it->second;
            }

            // Allocate a batch of 1 example.
            std::unique_ptr<ygg::serving::AbstractExampleSet> examples =
                serving_data_->serving_engine_->AllocateExamples(1);

            // Set all the values as missing. This is only necessary if you don't set all
            // the feature values manually e.g. SetNumerical.
            examples->FillMissing(*serving_data_->features_.get());

            // Set the value of "age" and "eduction" for the first example.
            examples->SetNumerical(/*example_idx=*/0, *serving_data_->age_feature_.get(), /*35.f*/ age, *serving_data_->features_.get());
            examples->SetCategorical(/*example_idx=*/0, *serving_data_->education_feature_.get(), /*"HS-grad"*/ education,
                                     *serving_data_->features_.get());

            // Run the predictions on the first two examples.
            std::vector<float> batch_of_predictions;
            serving_data_->serving_engine_->Predict(*examples, 1, &batch_of_predictions);

            response_.set(http::field::content_type, "application/json");

            std::stringstream ss;
            ss << batch_of_predictions[0];

            json::object obj;
            obj["res"] = ss.str();
            beast::ostream(response_.body()) << json::serialize(obj);
        }
        else
        {
            response_.result(http::status::not_found);
            response_.set(http::field::content_type, "application/json");
            json::object obj;
            obj["error"] = std::string("Target not found: ") + request_.target().to_string();
            beast::ostream(response_.body()) << json::serialize(obj);
        }
    }

    // Asynchronously transmit the response message.
    void
    write_response()
    {
        auto self = shared_from_this();

        response_.content_length(response_.body().size());

        http::async_write(
            socket_,
            response_,
            [self](beast::error_code ec, std::size_t)
            {
                self->socket_.shutdown(tcp::socket::shutdown_send, ec);
                self->deadline_.cancel();
            });
    }

    // Check whether we have spent enough time on this connection.
    void
    check_deadline()
    {
        auto self = shared_from_this();

        deadline_.async_wait(
            [self](beast::error_code ec)
            {
                if (!ec)
                {
                    // Close socket to cancel any outstanding operation.
                    self->socket_.close(ec);
                }
            });
    }
};

// "Loop" forever accepting new connections.
void http_server(tcp::acceptor &acceptor, tcp::socket &socket, std::unordered_map<std::string, serving_data> &ht)
{
    acceptor.async_accept(socket,
                          [&](beast::error_code ec)
                          {
                              if (!ec)
                                  std::make_shared<http_connection>(std::move(socket), ht)->start();
                              http_server(acceptor, socket, ht);
                          });
}

void train_and_export_model()
{
    // Path to the training and testing dataset.
    const auto train_dataset_path = absl::StrCat(
        "csv:",
        file::JoinPath(absl::GetFlag(FLAGS_dataset_dir), "adult_train.csv"));

    const auto test_dataset_path = absl::StrCat(
        "csv:",
        file::JoinPath(absl::GetFlag(FLAGS_dataset_dir), "adult_test.csv"));

    // Create the output directory
    QCHECK_OK(file::RecursivelyCreateDir(absl::GetFlag(FLAGS_output_dir),
                                         file::Defaults()));

    // Scan the columns of the dataset to create a dataspec.
    // Same as :infer_dataspec
    LOG(INFO) << "Create dataspec";
    const auto dataspec_path =
        file::JoinPath(absl::GetFlag(FLAGS_output_dir), "dataspec.pbtxt");
    ygg::dataset::proto::DataSpecification dataspec;
    ygg::dataset::CreateDataSpec(train_dataset_path, false, /*guide=*/{},
                                 &dataspec);
    QCHECK_OK(file::SetTextProto(dataspec_path, dataspec, file::Defaults()));

    // Display the dataspec in a human readable form.
    // Same as :show_dataspec
    LOG(INFO) << "Nice print of the dataspec";
    const auto dataspec_report =
        ygg::dataset::PrintHumanReadable(dataspec, false);
    QCHECK_OK(
        file::SetContent(absl::StrCat(dataspec_path, ".txt"), dataspec_report));
    LOG(INFO) << "Dataspec:\n"
              << dataspec_report;

    // Train the model.
    // Same as :train
    LOG(INFO) << "Train model";

    // Configure the learner.
    ygg::model::proto::TrainingConfig train_config;
    train_config.set_learner("RANDOM_FOREST");
    train_config.set_task(ygg::model::proto::Task::CLASSIFICATION);
    train_config.set_label("income");
    std::unique_ptr<ygg::model::AbstractLearner> learner;
    QCHECK_OK(GetLearner(train_config, &learner));

    // Set to export the training logs.
    learner->set_log_directory(absl::GetFlag(FLAGS_output_dir));

    // Effectively train the model.
    auto model = learner->TrainWithStatus(train_dataset_path, dataspec).value();

    // Save the model.
    LOG(INFO) << "Export the model";
    const auto model_path =
        file::JoinPath(absl::GetFlag(FLAGS_output_dir), "model");
    QCHECK_OK(ygg::model::SaveModel(model_path, model.get()));

    // Show information about the model.
    // Like :show_model, but without the list of compatible engines.
    std::string model_description;
    model->AppendDescriptionAndStatistics(/*full_definition=*/false,
                                          &model_description);
    QCHECK_OK(
        file::SetContent(absl::StrCat(model_path, ".txt"), model_description));
    LOG(INFO) << "Model:\n"
              << model_description;

    // Evaluate the model
    // Same as :evaluate
    ygg::dataset::VerticalDataset test_dataset;
    QCHECK_OK(ygg::dataset::LoadVerticalDataset(
        test_dataset_path, model->data_spec(), &test_dataset));

    ygg::utils::RandomEngine rnd;
    ygg::metric::proto::EvaluationOptions evaluation_options;
    evaluation_options.set_task(model->task());

    // The effective evaluation.
    const ygg::metric::proto::EvaluationResults evaluation =
        model->Evaluate(test_dataset, evaluation_options, &rnd);

    // Export the raw evaluation.
    const auto evaluation_path =
        file::JoinPath(absl::GetFlag(FLAGS_output_dir), "evaluation.pbtxt");
    QCHECK_OK(file::SetTextProto(evaluation_path, evaluation, file::Defaults()));

    // Export the evaluation to a nice text.
    std::string evaluation_report;
    QCHECK_OK(
        ygg::metric::AppendTextReportWithStatus(evaluation, &evaluation_report));
    QCHECK_OK(file::SetContent(absl::StrCat(evaluation_path, ".txt"),
                               evaluation_report));
    LOG(INFO) << "Evaluation:\n"
              << evaluation_report;

    // Compile the model for fast inference.
    const std::unique_ptr<ygg::serving::FastEngine> serving_engine =
        model->BuildFastEngine().value();
    const auto &features2 = serving_engine->features();

    // Handle to two features.
    const auto age_feature2 = features2.GetNumericalFeatureId("age").value();
    const auto education_feature2 = features2.GetCategoricalFeatureId("education").value();

    // Allocate a batch of 5 examples.
    std::unique_ptr<ygg::serving::AbstractExampleSet> examples2 =
        serving_engine->AllocateExamples(5);

    // Set all the values as missing. This is only necessary if you don't set all
    // the feature values manually e.g. SetNumerical.
    examples2->FillMissing(features2);

    // Set the value of "age" and "eduction" for the first example.
    examples2->SetNumerical(/*example_idx=*/0, age_feature2, 35.f, features2);
    examples2->SetCategorical(/*example_idx=*/0, education_feature2, "HS-grad",
                              features2);

    // Run the predictions on the first two examples.
    std::vector<float> batch_of_predictions;
    serving_engine->Predict(*examples2, 2, &batch_of_predictions);

    LOG(INFO) << "Predictions:";
    for (const float prediction : batch_of_predictions)
    {
        LOG(INFO) << "\t" << prediction;
    }
}

int main(int argc, char **argv)
{
    // Enable the logging. Optional in most cases.
    InitLogging(argv[0], &argc, &argv, true);

    try
    {
        train_and_export_model();

        std::unordered_map<std::string, serving_data> model_store;

        net::io_context ioc{1};

        tcp::acceptor acceptor{ioc, {net::ip::make_address(SERVER_ADDRESS), SERVER_PORT}};
        tcp::socket socket{ioc};
        http_server(acceptor, socket, model_store);

        ioc.run();
    }
    catch (std::exception const &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}