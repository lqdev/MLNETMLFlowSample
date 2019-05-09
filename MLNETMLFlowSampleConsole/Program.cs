using System;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.Extensions.DependencyInjection;
using MLNETMLFlowSampleConsole.Domain;
using MLFlow.NET.Lib;
using MLFlow.NET.Lib.Contract;
using MLFlow.NET.Lib.Model;
using MLFlow.NET.Lib.Model.Responses.Experiment;
using MLFlow.NET.Lib.Model.Responses.Run;

namespace MLNETMLFlowSampleConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            // Initialize Configurations
            var appConfig = new Startup();
            appConfig.ConfigureServices();

            // Initialize MLFlow Service
            IMLFlowService mlFlowService = appConfig.Services.GetRequiredService<IMLFlowService>();

            // Run Experiment
            RunExperiment(mlFlowService);
        }

        public static async void RunExperiment(IMLFlowService mlFlowService)
        {
            // 1. Create MLContext
            MLContext ctx = new MLContext();

            // 2. Load Data
            IDataView data = ctx.Data.LoadFromTextFile<IrisData>("Data/iris.data", separatorChar: ',');

            // 3. Define Experiment Settings
            var experimentSettings = new MulticlassExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 120;
            experimentSettings.OptimizingMetric = MulticlassClassificationMetric.LogLoss;

            // 4. Create Experiment
            var experiment = ctx.Auto().CreateMulticlassClassificationExperiment(experimentSettings);

            // 5. Create Experiment in MLFlow
            var experimentRequest = await mlFlowService.GetOrCreateExperiment(Guid.NewGuid().ToString());

            // 6. Run Experiment
            experiment.Execute(data, progressHandler: new ProgressHandler(mlFlowService, experimentRequest.ExperimentId));
        }
    }
}
