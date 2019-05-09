using System;
using System.Linq;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLFlow.NET.Lib;
using MLFlow.NET.Lib.Contract;
using MLFlow.NET.Lib.Model;
using MLFlow.NET.Lib.Model.Responses.Experiment;
using MLFlow.NET.Lib.Model.Responses.Run;

namespace MLNETMLFlowSampleConsole
{
    public class ProgressHandler : IProgress<RunDetail<MulticlassClassificationMetrics>>
    {
        private readonly IMLFlowService _mlFlowService;
        private readonly int _experimentId;

        public ProgressHandler(IMLFlowService mlFlowService, int experimentId)
        {
            _mlFlowService = mlFlowService;
            _experimentId = experimentId;
        }

        public async void Report(RunDetail<MulticlassClassificationMetrics> runDetails)
        {
            // Define run
            var runObject = new CreateRunRequest();
            runObject.ExperimentId = _experimentId;
            runObject.SourceType = SourceType.LOCAL;

            // Create new run in MLFlow
            var runRequest = await _mlFlowService.CreateRun(runObject);

            // Log trainer name
            await _mlFlowService.LogParameter(runRequest.Run.Info.RunUuid, nameof(runDetails.TrainerName), runDetails.TrainerName);

            // Extract current run's model parameters.
            var modelParameters = runDetails.Model as LinearModelParameters;

            // Log weights
            var weights =
                modelParameters
                .Weights
                .Select((weight, idx) => new { idx, weight });

            foreach (var item in weights)
            {
                await _mlFlowService.LogParameter(runRequest.Run.Info.RunUuid, $"weight_{item.idx}", item.weight.ToString());
            }

            // Log bias
            await _mlFlowService.LogParameter(runRequest.Run.Info.RunUuid, nameof(modelParameters.Bias), modelParameters.Bias.ToString());

            // Log performance metrics
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.RuntimeInSeconds), (float)runDetails.RuntimeInSeconds);
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.ValidationMetrics.LogLoss), (float)runDetails.ValidationMetrics.LogLoss);
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.ValidationMetrics.MacroAccuracy), (float)runDetails.ValidationMetrics.MacroAccuracy);
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.ValidationMetrics.MicroAccuracy), (float)runDetails.ValidationMetrics.MicroAccuracy);
        }
    }
}