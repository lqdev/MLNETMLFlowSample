using Microsoft.ML.Data;

namespace MLNETMLFlowSampleConsole.Domain
{
    public class IrisData
    {
        [LoadColumn(0, 3),
        VectorType(4),
        ColumnName("Features")]
        public float Features { get; set; }

        [LoadColumn(4),
        ColumnName("Label")]
        public string IrisType { get; set; }
    }
}