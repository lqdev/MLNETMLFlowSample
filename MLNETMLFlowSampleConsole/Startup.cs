using System;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using MLFlow.NET.Lib;
using MLFlow.NET.Lib.Contract;
using MLFlow.NET.Lib.Model;

namespace MLNETMLFlowSampleConsole
{
    public class Startup
    {
        private readonly IConfiguration Configuration;
        public IServiceProvider Services;

        public Startup()
        {
            Configuration =
                new ConfigurationBuilder()
                    .AddJsonFile("appsettings.json", false)
                    .Build();
        }

        public void ConfigureServices()
        {
            var services = new ServiceCollection();

            // Add and configure MLFlow Service
            services.AddMFlowNet();
            services.Configure<MLFlowConfiguration>(
                Configuration.GetSection(nameof(MLFlowConfiguration))
            );

            Services = services.BuildServiceProvider();
        }
    }
}