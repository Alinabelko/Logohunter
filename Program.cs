using Logohunter_cshap;
using System;
using System.Collections.Generic;

namespace Logohunter_charp
{
    class Program
    {
        static void Main(string[] args)
        {

            Logohunter logohunter = new Logohunter(@"\data\brands");

            logohunter.RunDetection(@"data\images");
        }
    }
}