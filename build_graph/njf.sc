import io.shiftleft.codepropertygraph.generated.EdgeTypes

import scala.collection.JavaConverters._

import scala.collection.mutable

import java.io.{PrintWriter, File => JFile}
import java.io.{File}

import overflowdb.traversal
import overflowdb.{Node, Edge}

type aEdgeEntry = (Long, Long, Int)
type aVertexEntry = (Long, String, String)
type cEdgeEntry = (Long, Long, Int)
type cVertexEntry = (Long, String, String)
type pEdgeEntry = (Long, Long, Int)
type pVertexEntry = (Long, String, String)

//type r = (Option[String], List[aEdgeEntry], List[aVertexEntry])
type r = (Option[String], List[aEdgeEntry], List[aVertexEntry], List[cEdgeEntry], List[cVertexEntry],List[pEdgeEntry], List[pVertexEntry])

//java.util.Iterator[overflowdb.Edge]

private def astFromEdges(edges: overflowdb.traversal.Traversal[overflowdb.Edge]): (List[aEdgeEntry], List[aVertexEntry]) = {
  val filteredEdges = edges.hasLabel(EdgeTypes.AST).dedup.l
  val (edgeResult, vertexResult) =
    filteredEdges.foldLeft((mutable.Set.empty[aEdgeEntry], mutable.Set.empty[aVertexEntry])) {
      case ((edgeList, vertexList), edge) =>
        val edgeEntry = (edge.outNode().id, edge.inNode()id,0)
        var outVertexEntry = (edge.outNode().id, edge.outNode().property("CODE").toString, "Empty")
        if (edge.outNode().property("LINE_NUMBER") != null) {
            outVertexEntry = (edge.outNode().id, edge.outNode().property("CODE").toString, edge.outNode().property("LINE_NUMBER").toString)
        }
        var inVertexEntry = (edge.inNode().id, edge.inNode().property("CODE").toString, "Empty")
        if (edge.outNode().property("LINE_NUMBER") != null) {
            inVertexEntry = (edge.inNode().id, edge.inNode().property("CODE").toString, edge.outNode().property("LINE_NUMBER").toString)
        }
        (edgeList += edgeEntry, vertexList ++= Set(outVertexEntry, inVertexEntry))
    }
  (edgeResult.toList, vertexResult.toList)
}

private def cfgFromEdges(edges: overflowdb.traversal.Traversal[overflowdb.Edge]): (List[aEdgeEntry], List[aVertexEntry]) = {
  val filteredEdges = edges.hasLabel(EdgeTypes.CFG, EdgeTypes.CDG).dedup.l
  val (edgeResult, vertexResult) =
    filteredEdges.foldLeft((mutable.Set.empty[aEdgeEntry], mutable.Set.empty[aVertexEntry])) {
      case ((edgeList, vertexList), edge) =>
        val edgeEntry = (edge.outNode().id, edge.inNode()id,1)
        var outVertexEntry = (edge.outNode().id, edge.outNode().property("CODE").toString, "Empty")
        if (edge.outNode().property("LINE_NUMBER") != null) {
            outVertexEntry = (edge.outNode().id, edge.outNode().property("CODE").toString, edge.outNode().property("LINE_NUMBER").toString)
        }
        var inVertexEntry = (edge.inNode().id, edge.inNode().property("CODE").toString, "Empty")
        if (edge.outNode().property("LINE_NUMBER") != null) {
            inVertexEntry = (edge.inNode().id, edge.inNode().property("CODE").toString, edge.outNode().property("LINE_NUMBER").toString)
        }
        (edgeList += edgeEntry, vertexList ++= Set(outVertexEntry, inVertexEntry))
    }
  (edgeResult.toList, vertexResult.toList)
}

private def pdgFromEdges(edges: overflowdb.traversal.Traversal[overflowdb.Edge]): (List[aEdgeEntry], List[aVertexEntry]) = {
  val filteredEdges = edges.hasLabel(EdgeTypes.CFG, EdgeTypes.CDG, EdgeTypes.REACHING_DEF).dedup.l
  //val filteredEdges = edges.hasLabel(EdgeTypes.CDG, EdgeTypes.REACHING_DEF).dedup.l
  val (edgeResult, vertexResult) =
    filteredEdges.foldLeft((mutable.Set.empty[aEdgeEntry], mutable.Set.empty[aVertexEntry])) {
      case ((edgeList, vertexList), edge) =>
        val edgeEntry = (edge.outNode().id, edge.inNode()id,2)
        var outVertexEntry = (edge.outNode().id, edge.outNode().property("CODE").toString, "Empty")
        if (edge.outNode().property("LINE_NUMBER") != null) {
            outVertexEntry = (edge.outNode().id, edge.outNode().property("CODE").toString, edge.outNode().property("LINE_NUMBER").toString)
        }
        var inVertexEntry = (edge.inNode().id, edge.inNode().property("CODE").toString, "Empty")
        if (edge.outNode().property("LINE_NUMBER") != null) {
            inVertexEntry = (edge.inNode().id, edge.inNode().property("CODE").toString, edge.outNode().property("LINE_NUMBER").toString)
        }
        (edgeList += edgeEntry, vertexList ++= Set(outVertexEntry, inVertexEntry))
    }
  (edgeResult.toList, vertexResult.toList)
}


def result(methodRegex: String = ""): List[r] = {
  if (methodRegex.isEmpty) {
    val (aedgeEntries, avertexEntries) = astFromEdges(cpg.graph.E())
    val (cedgeEntries, cvertexEntries) = cfgFromEdges(cpg.graph.E())
    val (pedgeEntries, pvertexEntries) = pdgFromEdges(cpg.graph.E())
    List((null, aedgeEntries, avertexEntries, cedgeEntries, cvertexEntries, pedgeEntries, pvertexEntries))
    //List((null, pedgeEntries, pvertexEntries))


  } else {
    cpg.method(methodRegex).l.map { method =>
      val methodFile = method.location.filename+"-"+method.name
      val (aedgeEntries, avertexEntries) = astFromEdges(method.out().flatMap(_.outE()))
      val (cedgeEntries, cvertexEntries) = cfgFromEdges(method.out().flatMap(_.outE()))
      val (pedgeEntries, pvertexEntries) = pdgFromEdges(method.out().flatMap(_.outE()))
      (Some(methodFile), aedgeEntries, avertexEntries, cedgeEntries, cvertexEntries, pedgeEntries, pvertexEntries)
      //(Some(methodFile), pedgeEntries, pvertexEntries)
    }
  }
}

@main def main(cpgFile: String, outDir: String)= {
  loadCpg(cpgFile)
  var item = 0
  val list = result(".*")
  println(list.length)
  //Please modify the path of the result
  //val dirPath = "/home/qiufangcheng/workspace/SGS/data/sard1/raw_result/Vul"
  val dirPath = outDir
  val resultPath = new File(dirPath)
  resultPath.mkdirs()

  for (item <- list){
        var filename=BigInt(100, scala.util.Random).toString(36)
	val writer = new PrintWriter(new JFile(dirPath+"/"+filename+".txt"))

	writer.println(item)
        writer.close()
  }
  
}
