using System;
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

            var runObject = new CreateRunRequest();
            runObject.ExperimentId = _experimentId;
            runObject.SourceType = SourceType.LOCAL;

            var runRequest = await _mlFlowService.CreateRun(runObject);

            await _mlFlowService.LogParameter(runRequest.Run.Info.RunUuid, nameof(runDetails.TrainerName), runDetails.TrainerName);
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.RuntimeInSeconds), (float)runDetails.RuntimeInSeconds);
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.ValidationMetrics.MacroAccuracy), (float)runDetails.ValidationMetrics.MacroAccuracy);
            await _mlFlowService.LogMetric(runRequest.Run.Info.RunUuid, nameof(runDetails.ValidationMetrics.MicroAccuracy), (float)runDetails.ValidationMetrics.MicroAccuracy);

        }
    }
}