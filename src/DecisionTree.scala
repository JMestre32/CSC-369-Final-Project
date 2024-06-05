package example

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoder}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import java.io.{File, PrintWriter}

object App {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .appName("Decision Tree Example")
      .master("local[*]")
      .getOrCreate()

    val data = spark.read.option("header", "true").option("inferSchema", "true").csv("/home/bmontelo/CSC369/final-project/movie-data/movie_try.csv")

    // Create the target variable
    val dataWithLabel = data.withColumn("label", when(col("revenue") < col("budget"), 0).otherwise(1))

    // Create indexers and encoderes
    val genreIndexer = new StringIndexer().setInputCol("genre").setOutputCol("genreIndex").fit(dataWithLabel)
    val genreEncoder = new OneHotEncoder().setInputCol("genreIndex").setOutputCol("genreVec")
    val companyIndexer = new StringIndexer().setInputCol("first_company_name").setOutputCol("companyIndex").fit(dataWithLabel)
    val companyEncoder = new OneHotEncoder().setInputCol("companyIndex").setOutputCol("companyVec")

    // Transform and encode data
    val indexedData = genreIndexer.transform(dataWithLabel)
    val companyIndexedData = companyIndexer.transform(indexedData)
    val encodedData = genreEncoder.transform(companyIndexedData)
    val fullyEncodedData = companyEncoder.transform(encodedData)

    // Assemble features into a feature vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("popularity", "vote_average", "vote_count", "genreVec", "companyVec"))
      .setOutputCol("features")

    val dt = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipeline = new Pipeline().setStages(Array(assembler, dt))

    val model = pipeline.fit(fullyEncodedData)
    val predictions = model.transform(fullyEncodedData)

    // Evaluate the model
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy = $accuracy")

    val treeModel = model.stages.last.asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

    val featureAttrs = AttributeGroup.fromStructField(predictions.schema("features")).attributes.get
    val writer = new PrintWriter(new File("./feature_names.txt"))
    featureAttrs.zipWithIndex.foreach { case (attr, index) =>
      writer.write(s"Feature $index: ${attr.name.getOrElse("Unknown")}\n")
    }
    writer.close()


    spark.stop()
  }
}
