using Logohunter_cshap;
using System;
using System.Collections.Generic;

namespace Logohunter_charp
{
    class Program
    {
        static void Main(string[] args)
        {
            List<string> images = new List<string> { @"C:\Users\Alina\source\repos\new\Logohunter_cshap\lexus.jpg" };
            List<string> brands = new List<string> { @"C:\Users\Alina\source\repos\new\Logohunter_cshap\test_lexus.png" };

            //Logohunter logohunter = new Logohunter(brands);
            //logohunter.RunDetection(images);

            Logohunter logohunter = new Logohunter(@"C:\Users\Alina\source\repos\new\Logohunter_cshap\data\brands");
            logohunter.RunDetection(@"C:\Users\Alina\source\repos\new\Logohunter_cshap\data\images");
        }
    }
}