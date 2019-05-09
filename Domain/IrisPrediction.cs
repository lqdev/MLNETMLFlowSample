using Microsoft.ML.Data;

namespace MLNETMLFlowSample.Domain
{
    public class IrisPrediction : IrisData
    {
        [ColumnName("PredictedLabel")]
        public string PredictedFlower { get; set; }
    }
}