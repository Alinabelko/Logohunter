using Alturos.Yolo;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Logohunter_cshap
{
    class Logohunter
    {
        private YoloWrapper _yolowrapper;

        public Logohunter()
        {
            _yolowrapper = new YoloWrapper("yolo_logohunter.cfg",
                "yolo_logohunter.weights",
                "yolo_logohunter.names");
        }
        public void RunDetection(List<string> imagePaths, List<string> brands = null)
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
    }
}
