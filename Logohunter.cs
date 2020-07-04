using Alturos.Yolo;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows.Markup;
using TensorFlow;

namespace Logohunter_cshap
{
    class Logohunter
    {
        private YoloWrapper _yolowrapper;

        private TFGraph _graph;

        private Dictionary<string, double> _cutoffs;

        private const string CFG = "yolo_logohunter.cfg";
        private const string WEIGHTS = "yolo_logohunter.weights";
        private const string NAMES = "yolo_logohunter.names";
        private const string INCEPTION = "inception_logohunter.pb";
        //possible values flickr27_features.csv or logosinthewild_features.csv
        private const string FEATURES = "flickr27_features.csv";
        public Logohunter(List<string> brandsPaths)
        {
            _yolowrapper = new YoloWrapper(CFG, WEIGHTS, NAMES);
            _graph = new TFGraph();
            _graph.Import(File.ReadAllBytes(INCEPTION));
            _cutoffs = LoadBrandsComputeCutoffs(brandsPaths, LoadFeatures());
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
                            bmp.Save($@"data\results\{Path.GetFileNameWithoutExtension(brand.Key)}_{random.Next()}.jpg");
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
                    double maxSimilarity = 0;
                    string maxSimilarBrand = "";
                    foreach (var brand in _cutoffs)
                    {
                        var similarity = ComputeCosineSimilatity(candidateFeatures, ExtractFeatures(new Bitmap(brand.Key)));
                        if (similarity > brand.Value && (similarity - brand.Value) > maxSimilarity)
                        {
                            maxSimilarity = similarity;
                            maxSimilarBrand = brand.Key;
                        }
                    }
                    if (maxSimilarity != 0)
                    {
                        Random random = new Random();
                        bmp.Save($@"data\results\{Path.GetFileNameWithoutExtension(maxSimilarBrand)}_{random.Next()}.jpg");
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
            for (int k = 0; k < 1280; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        flattenFeatures[n] = features[0, j, i, k];
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
            for (int k = 0; k < 1280; k++)            
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        flattenFeatures[n] = features[0, j, i, k];
                        n++;
                    }
                }
            }
            return flattenFeatures;
        }

        private List<float[]> LoadFeatures()
        {
            List<float[]> features = new List<float[]>();
            using (var reader = new StreamReader(FEATURES))
            {               
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split('\t');
                    if(values.Length != 20480)
                    {
                        values = line.Split(',');
                    }
                    float[] vector = new float[values.Length];                   
                    for (int i = 0; i < vector.Length; i++)
                    {
                        if (string.IsNullOrEmpty(values[i])){
                            vector[i] = 0;
                        }
                        else {
                            if(FEATURES == "logosinthewild_features.csv")
                            {
                                vector[i] = float.Parse(values[i], NumberStyles.Number | NumberStyles.AllowExponent, new CultureInfo("en-US"));
                            }
                            else
                            {
                                vector[i] = float.Parse(values[i], CultureInfo.InvariantCulture);
                            }                         
                        }
                    
                    }
                    
                    features.Add(vector);
                    Console.WriteLine($"Feature readed: {features.Count}");
                }
            }
            return features;
        }

        private Dictionary<string, double> LoadBrandsComputeCutoffs(List<string> brands, List<float[]> featuresList) 
        {
            Dictionary<string, double> cutoffs = new Dictionary<string, double>();
            for (int i = 0; i < brands.Count; i++)
            {
                var brandFeatures = ExtractFeatures(brands[i]);
                //to compute 95% cutoff of similarity distibution we save only top 5% values of similarity and choose min of them.
                double[] topFivePercentSimilarity = new double[(int)(featuresList.Count*0.05)];
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

        /// <summary>
        /// Computes logo features of LogosInTheWildv2 dataset for extractor model
        /// </summary>
        /// <param name="datasetFolder">Folder with all logos images.</param>
        /// <param name="featuresFile">Output features file name.</param>
        public void CreateFeatures(string datasetFolder, string Rois, string featuresFile)
        {
            var file = File.CreateText(featuresFile);
            List<float[]> features = new List<float[]>();
            System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
            customCulture.NumberFormat.NumberDecimalSeparator = ".";

            System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;
            using (var reader = new StreamReader(Rois))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(' ');
                    var logo = ImageUtil.CropImage(
                        Image.FromFile(@"flickr_logos_27_dataset_images\" + values[0]), 
                        int.Parse(values[3]),
                        int.Parse(values[4]),
                        int.Parse(values[5]) - int.Parse(values[3]),
                        int.Parse(values[6]) - int.Parse(values[4]));
                  
                    var feature = ExtractFeatures(logo);
                    features.Add(feature);
                    file.WriteLine(string.Join("\t", feature));
                    Console.WriteLine($"Feature created: {features.Count}");
                }
            }
            file.Close();
        }
    }
}