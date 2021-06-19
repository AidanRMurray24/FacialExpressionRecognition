using System;
using System.Collections.Generic;
using System.Text;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;
using System.IO;
using Prediction;

namespace FeatureExtractionLib
{
	struct Vector2
	{
		public Vector2(double x, double y)
		{
			X = x;
			Y = y;
		}

		public double X { get; }
		public double Y { get; }

		// Distance function
		public static double Distance(Vector2 a, Vector2 b)
		{
			return (Math.Sqrt(Math.Pow(a.X - b.X, 2) + Math.Pow(a.Y - b.Y, 2)));
		}

		// Magnitude
		public static double Magnitude(Vector2 a)
		{
			return (Math.Sqrt(Math.Pow(a.X, 2) + Math.Pow(a.Y, 2)));
		}

		// Normalise
		public static Vector2 Normalise(Vector2 a)
		{
			return (new Vector2(a.X / Magnitude(a), a.Y / Magnitude(a)));
		}
	}

	class FeatureExtraction
	{
		public void CreateNewCSVFileToExtractTo(string fileName)
		{
			// Header definition of the CSV file
			string header = "Label," +
				"LeftEyebrow," +
				"RightEyebrow," +
				"LeftLip,RightLip," +
				"LipHeight," +
				"LipWidth," +
				"LeftEyeHeight," +
				"LeftEyeWidth," +
				"RightEyeHeight," +
				"RightEyeWidth," +
				"LipsToNose," +
				"NoseHeight," +
				"NoseWidth," +
				"LeftEyeToLeftLip," +
				"RightEyeToRightLip\n";

			// Create the CSV file and fill in the first line with the header
			System.IO.File.WriteAllText(fileName, header);
		}

		private void ExtractExpresionDirectory(string directory, string extractTo, string expression = "Default")
		{
			Console.WriteLine($"Extracting Features...");
			string[] inputImages = Directory.GetFiles(directory, "*");

			for (int i = 0; i < inputImages.Length; i++)
			{
				Console.WriteLine($"Extracting from image: {inputImages[i]}");
				ExtractImageFeatures(inputImages[i], extractTo, expression);
			}
		}

		public void ExtractData(string directory,string extractTo)
		{
			// Extract the neutral folder
			string surpriseDir = directory + "/Surprise";
			this.ExtractExpresionDirectory(surpriseDir, extractTo, "Surprise");

			// Extract the neutral folder
			string sadnessDir = directory + "/Sadness";
			this.ExtractExpresionDirectory(sadnessDir, extractTo, "Sadness");

			// Extract the neutral folder
			string fearDir = directory + "/Fear";
			this.ExtractExpresionDirectory(fearDir, extractTo, "Fear");

			// Extract the neutral folder
			string angerDir = directory + "/Anger";
			this.ExtractExpresionDirectory(angerDir, extractTo, "Anger");

			// Extract the neutral folder
			string disgustDir = directory + "/Disgust";
			this.ExtractExpresionDirectory(disgustDir, extractTo, "Disgust");

			// Extract the neutral folder
			string joyDir = directory + "/Joy";
			this.ExtractExpresionDirectory(joyDir, extractTo, "Joy");
		}

		public FaceData ExtractImageFeatures(string imageFile, string extractTo = "DontSave", string expression = "Default")
		{
			// File paths
			string inputFilePath = imageFile;

			// Set the label name
			string label = expression;

			// Facial features
			float leftEyebrow = 0f,
				rightEyebrow = 0f,
				leftLip = 0f,
				rightLip = 0f,
				lipHeight = 0f,
				lipWidth = 0f,
				leftEyeHeight = 0f,
				leftEyeWidth = 0f,
				rightEyeHeight = 0f,
				rightEyeWidth = 0f,
				lipsToNose = 0f,
				noseHeight = 0f,
				noseWidth = 0f,
				leftEyeToLeftLip = 0f,
				rightEyeToRightLip = 0f;

			// Set up Dlib Face Detector
			using (var fd = Dlib.GetFrontalFaceDetector())

			// ... and Dlib Shape Detector
			using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
			{
				// load input image
				var img = Dlib.LoadImage<RgbPixel>(inputFilePath);

				// find all faces in the image
				var faces = fd.Operator(img);
				// for each face draw over the facial landmarks
				foreach (var face in faces)
				{
					// find the landmark points for this face
					var shape = sp.Detect(img, face);

					// Eyebrows
					leftEyebrow = CalculateFeature(shape, 39, 21, 18, 21);
					rightEyebrow = CalculateFeature(shape, 42, 22, 22, 25);

					// Lips
					leftLip = CalculateFeature(shape, 33, 51, 48, 50);
					rightLip = CalculateFeature(shape, 33, 51, 52, 54);

					// Lip width and height
					lipWidth = NormalisedDistBetween2Points(shape, 33, 51, 48, 54);
					lipHeight = NormalisedDistBetween2Points(shape, 33, 51, 51, 57);

					// Left eye
					leftEyeHeight = NormalisedDistBetween2Points(shape, 39, 38, 40, 37);
					leftEyeWidth = NormalisedDistBetween2Points(shape, 39, 38, 39, 36);

					// Right eye
					rightEyeHeight = NormalisedDistBetween2Points(shape, 42, 43, 47, 44);
					rightEyeWidth = NormalisedDistBetween2Points(shape, 42, 43, 43, 46);

					// Lips to nose
					lipsToNose = NormalisedDistBetween2Points(shape, 33, 51, 33, 57);

					// Nose
					noseHeight = CalculateFeature(shape, 33, 30, 27, 33);
					noseWidth = CalculateFeature(shape, 51, 33, 31, 35);

					// Lips to eyes
					leftEyeToLeftLip = NormalisedDistBetween2Points(shape, 27, 39, 41, 48);
					rightEyeToRightLip = NormalisedDistBetween2Points(shape, 27, 42, 46, 54);

					//Then write a new line with the calculated feature vector values separated by commas. Check that this works:
					if (extractTo != "DontSave")
					{
						using (System.IO.StreamWriter file = new System.IO.StreamWriter(extractTo, true))
						{
							file.WriteLine(
								label + "," + 
								leftEyebrow + "," + 
								rightEyebrow + "," + 
								leftLip + "," + 
								rightLip + "," + 
								lipHeight + "," + 
								lipWidth + "," +
								leftEyeHeight + "," +
								leftEyeWidth + "," +
								rightEyeHeight + "," +
								rightEyeWidth + "," +
								lipsToNose + "," +
								noseHeight + "," +
								noseWidth + "," +
								leftEyeToLeftLip + "," +
								rightEyeToRightLip
								);
						}
					}
					else
					{
						// Draw the landmark points on the image
						for (var i = 0; i < shape.Parts; i++)
						{
							var point = shape.GetPart((uint)i);
							var rect = new Rectangle(point);

							Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
						}

						// Output the image with the features marked
						Dlib.SaveJpeg(img, "output.jpg");
					}
				}

				return new FaceData()
				{
					LeftEyebrow = leftEyebrow,
					RightEyebrow = rightEyebrow,
					LeftLip = leftLip,
					RightLip = rightLip,
					LipHeight = lipHeight,
					LipWidth = lipWidth,
					LeftEyeHeight = leftEyeHeight,
					LeftEyeWidth = leftEyeWidth,
					RightEyeHeight = rightEyeHeight,
					RightEyeWidth = rightEyeWidth,
					LipsToNose = lipsToNose,
					NoseHeight = noseHeight,
					NoseWidth = noseWidth,
					LeftEyeToLeftLip = leftEyeToLeftLip,
					RightEyeToRightLip = rightEyeToRightLip
				};
			}
		}

		private float CalculateFeature(FullObjectDetection shape, int innerPoint, int normalisePoint, int leftMostPoint, int rightMostPoint)
		{
			float feature = 0f;

			// Loop through all the points on the feature adding on the normalised distance between that point and innerPoint
			for (var i = leftMostPoint; i <= rightMostPoint; i++)
			{
				feature += NormalisedDistBetween2Points(shape, innerPoint, normalisePoint, i, innerPoint);
			}

			return feature;
		}

		private float NormalisedDistBetween2Points(FullObjectDetection shape, int innerPoint, int normalisePoint, int firstPoint, int secondPoint)
		{
			// Get the positions of the innerpoint and the normalise point and use them to calculate the distance normaliser
			Vector2 innerPointPos = new Vector2(shape.GetPart((uint)innerPoint).X, shape.GetPart((uint)innerPoint).Y);
			Vector2 normalisePos = new Vector2(shape.GetPart((uint)normalisePoint).X, shape.GetPart((uint)normalisePoint).Y);
			double distNormaliser = Vector2.Distance(innerPointPos, normalisePos);

			// Get the position of both points and calculate the distance between them
			Vector2 firstPointPos = new Vector2(shape.GetPart((uint)firstPoint).X, shape.GetPart((uint)firstPoint).Y);
			Vector2 secondPointPos = new Vector2(shape.GetPart((uint)secondPoint).X, shape.GetPart((uint)secondPoint).Y);
			double distance = Vector2.Distance(firstPointPos, secondPointPos);

			// calculate the normalised distance between the points and return the value
			float normalisedDistance = (float)(distance / distNormaliser);
			return normalisedDistance;
		}
	}
}
