//Product Recommendation System Using Matrix Factorization in ML.NET(https://medium.com/codenx/product-recommendation-system-using-matrix-factorization-in-ml-net-003f4370e3ec)
using ConsoleApp;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

MLContext mlContext = new MLContext();

string FileName = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Amazon0302.txt");
var data = mlContext.Data.LoadFromTextFile(FileName,
    columns: new[]
    {
        new TextLoader.Column("Label", DataKind.Single, 0),
        new TextLoader.Column(name: nameof(PurchaseHistory.ProductId),
          dataKind: DataKind.UInt32,
          source: new[] { new TextLoader.Range(0) },
          keyCount: new KeyCount(262111)),
        new TextLoader.Column(name: nameof(PurchaseHistory.CoPurchaseProductId),
          dataKind: DataKind.UInt32,
          source: new[] { new TextLoader.Range(1) },
          keyCount: new KeyCount(262111))
    },
    hasHeader: true,
    separatorChar: '\t');

var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.4);

var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = nameof(PurchaseHistory.ProductId),
    MatrixRowIndexColumnName = nameof(PurchaseHistory.CoPurchaseProductId),
    LabelColumnName = "Label",
    LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
    Alpha = 0.01,
    Lambda = 0.025,
    C = 0.00001
};

var trainer = mlContext.Recommendation()
.Trainers
.MatrixFactorization(options);
var model = trainer.Fit(split.TrainSet);

var metrics = mlContext.Recommendation()
.Evaluate(model.Transform(split.TestSet));
Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");

//mlContext.Model.Save(model, data.Schema, "product recommender model.zip");

var predictionEngine = mlContext.Model
.CreatePredictionEngine<PurchaseHistory, PurchasePrediction>(model);

var lines = File.ReadAllLines(FileName);
var products = new List<PurchaseHistory>();
foreach (var line in lines)
{
    if (!string.IsNullOrWhiteSpace(line) && !line.StartsWith('#')) 
    {
        var parts = line.Split('\t');
        products.Add(new PurchaseHistory
        {
            ProductId = uint.Parse(parts[0]),
            CoPurchaseProductId = uint.Parse(parts[1])
        });
    }
}
// Loop through the products and make predictions
foreach (var product in products)
{
    var prediction = predictionEngine.Predict(product);
    if (prediction.Score > 0.75)
    {
        Console.WriteLine($"Product: {product.ProductId}, Co-purchased Product:{product.CoPurchaseProductId}, Score: {prediction.Score:P2}");
    }
}