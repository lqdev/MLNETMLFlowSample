using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
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
        private readonly static IMLFlowService _mlFlowService;
        static Program()
        {
            // Initialize app configuration
            var appConfig = new Startup();
            appConfig.ConfigureServices();

            // Initialize MLFlow service
            _mlFlowService = appConfig.Services.GetService<IMLFlowService>();
        }

        static async Task Main(string[] args)
        {
            // Run experiment
            await RunExperiment();
        }

        public static async Task RunExperiment()
        {
            // 1. Create MLContext
            MLContext ctx = new MLContext();

            // 2. Load data
            IDataView data = ctx.Data.LoadFromTextFile<IrisData>("Data/iris.data", separatorChar: ',');

            // 3. Define Automated ML.NET experiment settings
            var experimentSettings = new MulticlassExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 30;
            experimentSettings.OptimizingMetric = MulticlassClassificationMetric.LogLoss;
            
            // 4. Create Automated ML.NET
            var experiment = ctx.Auto().CreateMulticlassClassificationExperiment(experimentSettings);

            // 5. Create experiment in MLFlow
            var experimentName = Guid.NewGuid().ToString();
            var experimentRequest = await _mlFlowService.GetOrCreateExperiment(experimentName);

            // 6. Run Automated ML.NET experiment
            var experimentResults = experiment.Execute(data, progressHandler: new ProgressHandler());
            
            // 7. Log Best Run
            LogRun(experimentRequest.ExperimentId,experimentResults);
            
            string savePath = Path.Join("MLModels", $"{experimentName}");
            string modelPath = Path.Join(savePath, "model.zip");

            if (!Directory.Exists(savePath))
            {
                Directory.CreateDirectory(savePath);
            }

            // 8. Save Best Trained Model
            ctx.Model.Save(experimentResults.BestRun.Model, data.Schema, modelPath);
        }

        static async void LogRun(int experimentId, ExperimentResult<MulticlassClassificationMetrics> experimentResults)
        {
            // Define run
            var runObject = new CreateRunRequest();
            runObject.ExperimentId = experimentId;
            runObject.StartTime = ((DateTimeOffset)DateTime.UtcNow).ToUnixTimeMilliseconds();
            runObject.UserId = Environment.UserName;
            runObject.SourceType = SourceType.LOCAL;

            // Create new run in MLFlow
            var runRequest = await _mlFlowService.CreateRun(runObject);

            // Get information for best run
            var runDetails = experimentResults.BestRun;

            // Log trainer name
            await _mlFlowService.LogParameter(runRequest.Run.Info.RunUuid, nameof(runDetails.TrainerName), runDetails.TrainerName);

            // Log metrics
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.RuntimeInSeconds), (float)runDetails.RuntimeInSeconds);
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.ValidationMetrics.LogLoss), (float)runDetails.ValidationMetrics.LogLoss);
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.ValidationMetrics.MacroAccuracy), (float)runDetails.ValidationMetrics.MacroAccuracy);
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.ValidationMetrics.MicroAccuracy), (float)runDetails.ValidationMetrics.MicroAccuracy);
        }
    }
}
