using System;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MLNETMLFlowSampleConsole
{
    public class ProgressHandler : IProgress<RunDetail<MulticlassClassificationMetrics>>
    {
        public void Report(RunDetail<MulticlassClassificationMetrics> runDetails)
        {
            Console.WriteLine($"Ran {runDetails.TrainerName} in {runDetails.RuntimeInSeconds:0.##} with Log Loss {runDetails.ValidationMetrics.LogLoss:0.####}");
        }
    }
}