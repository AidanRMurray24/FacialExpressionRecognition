using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using FeatureExtractionLib;


namespace Prediction
{
	class PredictionModel
	{
		private ITransformer model = null;
		private MLContext mlContext = new MLContext();

		public void Train(string directory)
		{
			// Let the user know that the training process has begun
			Console.WriteLine($"Training Model...");

			// Load data
			IDataView dataView = mlContext.Data.LoadFromTextFile<FaceData>(directory, hasHeader: true, separatorChar: ',');

			// Define the feature vector name and the label column name
			var featureVectorName = "Features";
			var labelColumnName = "Label";

			// Define data preparation estimator
			var pipeline = mlContext.Transforms.Conversion
				.MapValueToKey(inputColumnName: "Label", outputColumnName: labelColumnName)
				.Append(mlContext.Transforms.Concatenate(
					featureVectorName,
					"LeftEyebrow",
					"RightEyebrow",
					"LeftLip",
					"RightLip",
					"LipHeight",
					"LipWidth",
					"LeftEyeHeight",
					"LeftEyeWidth",
					"RightEyeHeight",
					"RightEyeWidth",
					"LipsToNose",
					"NoseHeight",
					"NoseWidth",
					"LeftEyeToLeftLip",
					"RightEyeToRightLip"))
				.AppendCacheCheckpoint(mlContext)
				.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName, featureVectorName))
				.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

			// Train model
			model = pipeline.Fit(dataView);

			// Save model
			using (var fileStream = new FileStream("model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
			{
				mlContext.Model.Save(model, dataView.Schema, fileStream);
			}
		}

		public void Predict(string imageDirectory)
		{
			// Load in the model
			LoadModel();

			// Setup the predictor using the model from the training
			var predictor = mlContext.Model.CreatePredictionEngine<FaceData, ExpressionPrediction>(model);

			// Extract the features from the image and make a prediction
			FeatureExtraction featureExtraction = new FeatureExtraction();
			FaceData faceData = featureExtraction.ExtractImageFeatures(imageDirectory);
			var prediction = predictor.Predict(faceData);

			// Show the predicted results
			Console.WriteLine($"*** Prediction: {prediction.Label } ***");
			Console.WriteLine($"*** Scores: {string.Join(" ", prediction.Scores)} ***");
		}

		public void EvaluateModel(string testData)
		{
			// Load in the model
			LoadModel();

			// Evaluate the model
			IDataView testDataView = mlContext.Data.LoadFromTextFile<FaceData>(testData, hasHeader: true, separatorChar: ',');
			var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testDataView));

			// Write out to the console
			Console.WriteLine($"* Metrics for Multi-class Classification model - Test Data");
			Console.WriteLine($"* MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
			Console.WriteLine($"* MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
			Console.WriteLine($"* LogLoss:          {testMetrics.LogLoss:#.###}");
			Console.WriteLine($"* LogLossReduction: {testMetrics.LogLossReduction:#.###}");

			// LogLossPerClass (print the list of doubles):
			//Console.WriteLine($"* Log Loss Per Class:");
			//System.Collections.Generic.IReadOnlyList<double> logList = testMetrics.PerClassLogLoss;
			//for (int i = 0; i < logList.Count; i++)
			//{
			//	Console.WriteLine($"*    - { (ExpressionType)i } : {logList[i]:#.###}");
			//}

			// Precision per class
			Console.WriteLine($"* ConfusionMatrixPrecision:");
			System.Collections.Generic.IReadOnlyList<double> precisionList = testMetrics.ConfusionMatrix.PerClassPrecision;
			for (int i = 0; i < precisionList.Count; i++)
			{
				Console.WriteLine($"*    - {(ExpressionType)i} : {precisionList[i]:#.###}");
			}

			// Recall per class
			Console.WriteLine($"* ConfusionMatrixRecall:");
			System.Collections.Generic.IReadOnlyList<double> recallList = testMetrics.ConfusionMatrix.PerClassRecall;
			for (int i = 0; i < recallList.Count; i++)
			{
				Console.WriteLine($"*    - {(ExpressionType)i} : {recallList[i]:#.###}");
			}
		}

		private void LoadModel()
		{
			DataViewSchema dataViewSchema = null;
			using (var fileStream = new FileStream("model.zip", FileMode.Open, FileAccess.Read))
			{
				model = mlContext.Model.Load(fileStream, out dataViewSchema);
			}
		}
	}
}
