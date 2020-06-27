using Alturos.Yolo;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace Logohunter_cshap
{
    class Logohunter
    {
        private YoloWrapper _yolowrapper;

        private TFGraph _graph;

        public Logohunter()
        {
            _yolowrapper = new YoloWrapper("yolo_logohunter.cfg",
                "yolo_logohunter.weights",
                "yolo_logohunter.names");
            _graph = new TFGraph();
            _graph.Import(File.ReadAllBytes("inception_logohunter.pb"));
        }
        public void RunDetection(List<string> imagePaths)
        {
            foreach (string imagePath in imagePaths)
            {
                var logos = _yolowrapper.Detect(imagePath);
                //int i = 0;
                foreach (var logo in logos)
                {
                    Bitmap bmp = ImageUtil.CropImage(new Bitmap(imagePath), logo.X, logo.Y, logo.Width, logo.Height);
                    //save for testing
                    //bmp.Save($"logo{i}.jpg");
                    //i++;
                }
            }
        }

        public void ExtractFeatures(string imagePath)
        {
            var tensor = ImageUtil.CreateTensorFromImageFile(imagePath);
            var session = new TFSession(_graph);
            var runner = session.GetRunner();
            runner.AddInput(_graph["input_2"][0], tensor);
            runner.Fetch(_graph["mixed8/concat"][0]);

            var output = runner.Run();

            TFTensor result = output[0];

            var val = (float[,,,])result.GetValue(jagged: false);
            //TODO: flatten result
        }

        public void LoadFeatures()
        {
            using (var reader = new StreamReader("features.csv"))
            {
                List<float[]> features = new List<float[]>();
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
        }
        public void CalculateCosineSimilarity() { }


    }
}
