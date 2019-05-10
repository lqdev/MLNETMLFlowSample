# ML.NET MLFlow Sample Application

This application demos how to track model runs in MLFLow for models built using Automated ML in ML.NET

## Prerequisites

This project was built on an Ubuntu 18.04 PC but should work on Windows and Mac. Note that MLFlow does not natively run on Windows at the time of this writing. To run it on Windows use [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

- [Python 3.x](https://www.python.org/downloads/)
- [MLFlow](https://www.mlflow.org/docs/latest/quickstart.html)
- [.NET SDK 2.x](https://dotnet.microsoft.com/download)

### Get The Data

The data used in this dataset comes from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) and looks like the data below:

```text
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
```

First, create a directory for the data inside the console application directory:

```bash
mkdir Data
```

Then, download and save the file into the `Data` directory.

```bash
curl -o Data/iris.data https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

## Run The Application

### Start MLFlow Server

In the terminal, from the console application directory, enter the following command to start the MLFlow Server:

```bash
mlflow server
```

Navigate to `http://localhost:5000` in your browser. This will load the MLFLow UI.

### Train Model

Then, in another terminal, from the console application directory, enter the following command to run the experiment:

```bash
dotnet build
dotnet run
```