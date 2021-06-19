using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Prediction
{
	public enum ExpressionType
	{
		ANGRY = 0,
		DISGUST = 1,
		FEAR = 2,
		HAPPY = 3,
		SAD = 4,
		SURPRISED = 5
	}

	class ExpressionPrediction
	{
		[ColumnName("PredictedLabel")]
		public string Label { get; set; }

		[ColumnName("Score")]
		public float[] Scores { get; set; }
	}
}
