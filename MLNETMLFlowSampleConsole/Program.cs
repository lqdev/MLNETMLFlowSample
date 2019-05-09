using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using Microsoft.Extensions.DependencyInjection;
using MLNETMLFlowSampleConsole.Domain;
using MLFlow.NET.Lib.Contract;

namespace MLNETMLFlowSampleConsole
{
    class Program
    {
        private static readonly IMLFlowService _mlFlowService;
        static Program()
        {
            // Initialize app configuration
            var appConfig = new Startup();
            appConfig.ConfigureServices();

            // Initialize MLFlow service
            _mlFlowService = appConfig.Services.GetRequiredService<IMLFlowService>();
        }

        static void Main(string[] args)
        {
            // Run experiment
            RunExperiment();
        }

        public static async void RunExperiment()
        {
            // 1. Create MLContext
            MLContext ctx = new MLContext();

            // 2. Load data
            IDataView data = ctx.Data.LoadFromTextFile<IrisData>("Data/iris.data", separatorChar: ',');

            // 3. Define Automated ML.NET experiment settings
            var experimentSettings = new MulticlassExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 120;
            experimentSettings.OptimizingMetric = MulticlassClassificationMetric.LogLoss;

            // 4. Create Automated ML.NET
            var experiment = ctx.Auto().CreateMulticlassClassificationExperiment(experimentSettings);

            // 5. Create experiment in MLFlow
            var experimentName = Guid.NewGuid().ToString();
            var experimentRequest = await _mlFlowService.GetOrCreateExperiment(experimentName);

            // 6. Run Automated ML.NET experiment
            var experimentResults = experiment.Execute(data, progressHandler: new ProgressHandler(_mlFlowService, experimentRequest.ExperimentId));

            // 7. Save Best Trained Model
            string savePath = Path.Join("MLModels", $"{experimentName}");
            string modelPath = Path.Join(savePath, "model.zip");

            if (!Directory.Exists(savePath))
            {
                Directory.CreateDirectory(savePath);
            }

            ctx.Model.Save(experimentResults.BestRun.Model, data.Schema, modelPath);
        }
    }
}
