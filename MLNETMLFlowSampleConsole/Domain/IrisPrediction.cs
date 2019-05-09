using Microsoft.ML.Data;

namespace MLNETMLFlowSampleConsole.Domain
{
    public class IrisPrediction : IrisData
    {
        [ColumnName("PredictedLabel")]
        public string PredictedFlower { get; set; }
    }
}