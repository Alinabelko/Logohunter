using Alturos.Yolo;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http.Headers;
using TensorFlow;

namespace Logohunter_cshap
{
    class Logohunter
    {
        private YoloWrapper _yolowrapper;

        private TFGraph _graph;

        private Dictionary<string, double> _cutoffs;

        private const string CFG = @"C:\Users\Alina\source\repos\new\Logohunter_cshap\yolo_logohunter.cfg";
        private const string WEIGHTS = @"C:\Users\Alina\source\repos\new\Logohunter_cshap\yolo_logohunter.weights";
        private const string NAMES = @"C:\Users\Alina\source\repos\new\Logohunter_cshap\yolo_logohunter.names";
        private const string INCEPTION = @"C:\Users\Alina\source\repos\new\Logohunter_cshap\inception_logohunter.pb";
        private const string FEATURES = @"C:\Users\Alina\source\repos\new\Logohunter_cshap\features.csv";
        public Logohunter(List<string> brandsPaths)
        {
            Stopwatch sw;
            sw = Stopwatch.StartNew();
            _yolowrapper = new YoloWrapper(CFG, WEIGHTS, NAMES);
            Console.WriteLine($"Yolo initialization: {sw.ElapsedMilliseconds}");
            sw = Stopwatch.StartNew();
            _graph = new TFGraph();
            _graph.Import(File.ReadAllBytes(INCEPTION));
            Console.WriteLine($"Inception initialization: {sw.ElapsedMilliseconds}");
            sw = Stopwatch.StartNew();
            _cutoffs = LoadBrandsComputeCutoffs(brandsPaths, LoadFeatures());
            Console.WriteLine($"ComputeCutoffs: {sw.ElapsedMilliseconds} for single brand");
        }

        public Logohunter(string brandsFolder)
        {
            _yolowrapper = new YoloWrapper(CFG, WEIGHTS, NAMES);
            _graph = new TFGraph();
            _graph.Import(File.ReadAllBytes(INCEPTION));
            List<string> brandsPaths = Directory.GetFiles(brandsFolder).ToList();
            _cutoffs = LoadBrandsComputeCutoffs(brandsPaths, LoadFeatures());
        }

        public void RunDetection(List<string> imagePaths)
        {
            foreach (string imagePath in imagePaths)
            {
                var candidates = _yolowrapper.Detect(imagePath);
                foreach (var candidate in candidates)
                {
                    Bitmap bmp = ImageUtil.CropImage(new Bitmap(imagePath), candidate.X, candidate.Y, candidate.Width, candidate.Height);
                    float[] candidateFeatures = ExtractFeatures(bmp);
                    foreach(var brand in _cutoffs)
                    {
                        var similarity = ComputeCosineSimilatity(candidateFeatures, ExtractFeatures(new Bitmap(brand.Key)));
                        if(similarity > brand.Value)
                        {
                            Random random = new Random();
                            bmp.Save($@"C:\Users\Alina\source\repos\new\Logohunter_cshap\data\results\{Path.GetFileNameWithoutExtension(brand.Key)}_{random.Next()}.jpg");
                        }
                    }                  
                }
            }
        }

        public void RunDetection(string imagesFolder)
        {
            List<string> imagePaths = Directory.GetFiles(imagesFolder).ToList();
            foreach (string imagePath in imagePaths)
            {
                var candidates = _yolowrapper.Detect(imagePath);
                foreach (var candidate in candidates)
                {
                    Bitmap bmp = ImageUtil.CropImage(new Bitmap(imagePath), candidate.X, candidate.Y, candidate.Width, candidate.Height);
                    float[] candidateFeatures = ExtractFeatures(bmp);
                    foreach (var brand in _cutoffs)
                    {
                        var similarity = ComputeCosineSimilatity(candidateFeatures, ExtractFeatures(new Bitmap(brand.Key)));
                        if (similarity > brand.Value)
                        {
                            Random random = new Random();
                            bmp.Save($@"C:\Users\Alina\source\repos\new\Logohunter_cshap\data\results\{Path.GetFileNameWithoutExtension(brand.Key)}_{random.Next()}.jpg");
                        }
                    }
                }
            }
        }

        private float[] ExtractFeatures(Bitmap candidate)
        {
            var tensor = ImageUtil.CreateTensorFromBitmap(candidate);
            var session = new TFSession(_graph);
            var runner = session.GetRunner();
            runner.AddInput(_graph["input_2"][0], tensor);
            runner.Fetch(_graph["mixed8/concat"][0]);

            var output = runner.Run();

            TFTensor result = output[0];

            var features = (float[,,,])result.GetValue(jagged: false);
            //features shape: [1,4,4,1280]
            float[] flattenFeatures = new float[20480];
            int n = 0;
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < 1280; k++)
                    {
                        flattenFeatures[n] = features[0, i, j, k];
                        n++;
                    }
                }
            }
            return flattenFeatures;
        }

        private float[] ExtractFeatures(string imagePath)
        {
            var tensor = ImageUtil.CreateTensorFromImageFile(imagePath);
            var session = new TFSession(_graph);
            var runner = session.GetRunner();
            runner.AddInput(_graph["input_2"][0], tensor);
            runner.Fetch(_graph["mixed8/concat"][0]);

            var output = runner.Run();

            TFTensor result = output[0];

            var features = (float[,,,])result.GetValue(jagged: false);
            float[] flattenFeatures = new float[20480];
            int n = 0;
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < 1280; k++)
                    {
                        flattenFeatures[n] = features[0, i, j, k];
                        n++;
                    }
                }
            }
            return flattenFeatures;
        }

        private List<float[]> LoadFeatures()
        {
            Stopwatch sw;
            sw = Stopwatch.StartNew();
            List<float[]> features = new List<float[]>();
            using (var reader = new StreamReader(FEATURES))
            {               
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    float[] vector = new float[values.Length];                   
                    for (int i = 0; i < vector.Length; i++)
                    {
                        vector[i] = float.Parse(values[i], NumberStyles.Number | NumberStyles.AllowExponent, new CultureInfo("en-US"));
                    }
                    features.Add(vector);
                }
            }
            Console.WriteLine($"Features loaded in {sw.ElapsedMilliseconds}");
            return features;
        }

        private Dictionary<string, double> LoadBrandsComputeCutoffs(List<string> brands, List<float[]> featuresList) 
        {
            Dictionary<string, double> cutoffs = new Dictionary<string, double>();
            for (int i = 0; i < brands.Count; i++)
            {
                var brandFeatures = ExtractFeatures(brands[i]);
                //top compute 95% cutoff of similarity distibution we save only top 5% values of similarity and choose min of them.
                double[] topFivePercentSimilarity = new double[1024];
                foreach (float[] features in featuresList)
                {
                    double similarity = ComputeCosineSimilatity(brandFeatures, features);
                    if (similarity > topFivePercentSimilarity[0])
                    {
                        topFivePercentSimilarity[0] = similarity;
                        topFivePercentSimilarity = topFivePercentSimilarity.OrderBy(x => x).ToArray();
                    }
                }
                cutoffs.Add(brands[i], topFivePercentSimilarity[0]);
            }
            return cutoffs;
        }

        private double ComputeCosineSimilatity(float[] brandFeatures, float[] features)
        {
            double sum = 0;
            double squareBrand = 0;
            double squareFeatures = 0;
            for(int i = 0; i < 20480; i++)
            {
                sum += brandFeatures[i] * features[i];
                squareBrand += brandFeatures[i] * brandFeatures[i];
                squareFeatures += features[i] * features[i];
            }
            return sum / (Math.Sqrt(squareBrand) * Math.Sqrt(squareFeatures));
        }
    }
}