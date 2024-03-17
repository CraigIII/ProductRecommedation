//Product Recommendation System Using Matrix Factorization in ML.NET(https://medium.com/codenx/product-recommendation-system-using-matrix-factorization-in-ml-net-003f4370e3ec)
using Microsoft.ML.Data;

public class PurchaseHistory
{
    [KeyType(count: 262111)]
    public uint ProductId { get; set; }

    [KeyType(count: 262111)]
    public uint CoPurchaseProductId { get; set; }
}