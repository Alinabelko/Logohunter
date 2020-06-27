using Logohunter_cshap;
using System;
using System.Collections.Generic;

namespace Logohunter_charp
{
    class Program
    {
        static void Main(string[] args)
        {
            Logohunter logohunter = new Logohunter();

            List<string> images = new List<string> { "test.jpg" };

            logohunter.RunDetection(images);
        }
    }
}