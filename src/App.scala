package example
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoder}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.log4j.Logger
import org.apache.log4j.Level
import java.io.{File, PrintWriter}

object App {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .appName("Linear Regression Example")
      .getOrCreate()

    val data = spark.read.option("header", "true").option("inferSchema", "true").csv("/user/jmestre/input/movie_try.csv")

    // Create indexers and encoders
    val genreIndexer = new StringIndexer().setInputCol("genre").setOutputCol("genreIndex").fit(data)
    val genreEncoder = new OneHotEncoder().setInputCol("genreIndex").setOutputCol("genreVec")
    val companyIndexer = new StringIndexer().setInputCol("first_company_name").setOutputCol("companyIndex").fit(data)
    val companyEncoder = new OneHotEncoder().setInputCol("companyIndex").setOutputCol("companyVec")

    // Transform and encode data
    val indexedData = genreIndexer.transform(data)
    val companyIndexedData = companyIndexer.transform(indexedData)
    val encodedData = genreEncoder.transform(companyIndexedData)
    val fullyEncodedData = companyEncoder.transform(encodedData)

    val assembler = new VectorAssembler()
      .setInputCols(Array("budget", "popularity", "vote_average", "vote_count", "genreVec", "companyVec"))
      .setOutputCol("features")

    val lr = new LinearRegression()
      .setLabelCol("revenue")
      .setFeaturesCol("features")

    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    val model = pipeline.fit(fullyEncodedData)
    val predictions = model.transform(fullyEncodedData)

    val lrModel = model.stages.last.asInstanceOf[LinearRegressionModel]
    val featureAttributes = AttributeGroup.fromStructField(predictions.schema("features")).attributes.get

    val output = new StringBuilder
    output.append(s"Intercept: ${lrModel.intercept}\n")
    lrModel.coefficients.toArray.zip(featureAttributes).foreach {
      case (coeff, attr) => output.append(s"${attr.name.get}: $coeff\n")
    }

    val evaluator = new RegressionEvaluator()
      .setLabelCol("revenue")
      .setPredictionCol("prediction")
      .setMetricName("mse")
    val mse = evaluator.evaluate(predictions)
    output.append(s"Mean Squared Error (MSE) = $mse\n")

    // Writing output to a file
    val writer = new PrintWriter(new File("./output.txt"))
    writer.write(output.toString)
    writer.close()

    spark.stop()
  }
}
