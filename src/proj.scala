import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.log4j.{Level, Logger}
import org.apache.spark

import scala.util.Try
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vector => SparkVector}
import org.apache.spark.ml.regression.LinearRegression

import java.io.{File, PrintWriter}


// Define case class for movie information
case class movieInfo1(budget: Int, title: String, popularity: Double, company: String, revenue: Int, vote_avg: Double, vote_cnt: Int, genre: String)

object proj {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "c:/winutils")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("try").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val movieFile = sc.textFile("c:/Users/Justin/Desktop/movies_try.txt")
    val movies = movieFile.flatMap { line =>
      val fields = line.split(",")
      if (fields.length == 8) {
        // despite my attempts to clean the data there is still some issues parsing
        //https://www.scala-lang.org/api/3.2.1/scala/util/Try.html
        Try {
          val budget = fields(0).trim.toInt
          val popularity = fields(2).trim.toDouble
          val revenue = fields(4).trim.toInt
          val vote_avg = fields(5).trim.toDouble
          val vote_cnt = fields(6).trim.toInt
          movieInfo1(budget, fields(1).trim, popularity, fields(3).trim, revenue, vote_avg, vote_cnt, fields(7).trim)
        }.toOption
      } else {
        List.empty[movieInfo1] // this creates an empty list effectively skipping this portion
      }
    }
    /*
    Finding the movies that are blockbusters
      - setting budget * 2.5
      - filtering movies greater than that
      - printing movie name, and number of movies that are blockbusters.
     */
    // movies.collect().foreach(println)
    // mapping budget and revenue, multiplying by 2.5 to fit blockbuster category
    val budgetRevenue = movies.map(movie => (movie.budget * 2.5, movie.revenue, movie.title))
    // filter movies with revenue greater than 2.5x the budget
    val blockBusters = budgetRevenue.filter {case (budgetX2_5, revenue, title) => revenue > budgetX2_5}
    blockBusters.collect().foreach{case(_,_,title) => println(title)}
    println("Total Number of Blockbusters", blockBusters.count())

    /*
    Find the avg popularity of each company
      - using aggregatebykey to do it by company
      - first acc: get value, and counter
      - second acc: add both of them together
      - sorted and wrote to a file to make plots
     */
    val companyPopularity = movies.map(movie => (movie.company, movie.popularity))
    val popularityTotalsAndCounts = companyPopularity
      .aggregateByKey((0.0, 0))(
        (acc, popularity) => (acc._1 + popularity, acc._2 + 1),
        (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2)
      )
    // Compute the average popularity for each company
    val averagePopularity = popularityTotalsAndCounts.mapValues {
      case (totalPopularity, count) => totalPopularity / count
    }
    averagePopularity.collect().foreach(println)
    // sorting and writing to a file for making plots
    val sortedAveragePopularity = averagePopularity.sortBy(_._2, ascending = false).take(10)
    val output = new StringBuilder
    sortedAveragePopularity.foreach { case (company, avgPopularity) =>
      output.append(s"Company: $company, Average Popularity: $avgPopularity\n")
    }
    val writer = new PrintWriter(new File("c:/Users/Justin/Desktop/soccer.txt"))
    writer.write(output.toString)
    writer.close()

    /*
      Find the average budget of each company
        - using aggregatebykey to do it by company
        - first acc: get value, and counter
        - second acc: add both of them together
        - sorted and wrote to a file to make plots
       */
    val companyBudget = movies.map(movie => (movie.company, movie.budget))
    val companyBudgetAvg = companyBudget
      .aggregateByKey((0.0, 0))(
        (acc, budget) => (acc._1 + budget, acc._2 + 1),
        (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2)
      )
    // calculating average budget per company
    val averageBudget = companyBudgetAvg.mapValues {
      case (totalBudget, count) => totalBudget / count
    }.sortBy(_._2, ascending = false).take(10)
   // averageBudget.foreach(println)
    val budgetCompanyoutput = new StringBuilder
    averageBudget.foreach{ case(company, avgBudget) =>
      budgetCompanyoutput.append(s"Company: $company, Average Popularity: $avgBudget\n")}

    val writer1 = new PrintWriter(new File("c:/Users/Justin/Desktop/budgetCompany.txt"))
    writer1.write(budgetCompanyoutput.toString)
    writer1.close()

  }
}

