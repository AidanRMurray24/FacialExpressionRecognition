using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Prediction
{
	class FaceData
	{
		[LoadColumn(0)]
		public string Label { get; set; }

		[LoadColumn(1)]
		public float LeftEyebrow { get; set; }
	
		[LoadColumn(2)]
		public float RightEyebrow { get; set; }

		[LoadColumn(3)]
		public float LeftLip { get; set; }

		[LoadColumn(4)]
		public float RightLip { get; set; }

		[LoadColumn(5)]
		public float LipHeight { get; set; }

		[LoadColumn(6)]
		public float LipWidth { get; set; }

		[LoadColumn(7)]
		public float LeftEyeHeight { get; set; }

		[LoadColumn(8)]
		public float LeftEyeWidth { get; set; }

		[LoadColumn(9)]
		public float RightEyeHeight { get; set; }

		[LoadColumn(10)]
		public float RightEyeWidth { get; set; }

		[LoadColumn(11)]
		public float LipsToNose { get; set; }

		[LoadColumn(12)]
		public float NoseHeight { get; set; }

		[LoadColumn(13)]
		public float NoseWidth { get; set; }

		[LoadColumn(14)]
		public float LeftEyeToLeftLip { get; set; }

		[LoadColumn(15)]
		public float RightEyeToRightLip { get; set; }

	}
}
