/**
 * Testing BST - Binary Search Tree functions
 *
 * @author Yusuf Pisan
 * @date 19 Oct 2019
 */

#include "graph.h"
#include <cassert>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

using namespace std;

// global value for testing
// NOLINTNEXTLINE
stringstream globalSS;

// need to reset SS before calling this
void vertexPrinter(const string &s) { globalSS << s; }

// need to reset SS before calling this
void edgePrinter(const string &from, const string &to, int weight) {
  globalSS << "[" << from << to << " " << weight << "]";
}

// convert a map to a string so we can compare it
template <typename K, typename L>
static string map2string(const map<K, L> &mp) {
  stringstream out;
  for (auto &p : mp) {
    out << "[" << p.first << ":" << p.second << "]";
  }
  return out.str();
}

void testGraphBasic() {
  cout << "testGraphBasic" << endl;
  Graph g;
  assert(g.add("a") && "add vertex a");
  assert(g.add("b") && "add vertex b");
  assert(g.add("c") && "add vertex c");
  assert(g.add("d") && "add vertex d");
  assert(g.add("e") && "add vertex e");
  assert(!g.add("b") && "b added twice");
  assert(g.connect("a", "b", 10) && "connect a b");
  assert(!g.connect("a", "b", 50) && "duplicate connect a b");
  assert(!g.connect("a", "a", 1) && "connect a to itself");
  g.connect("a", "d", 40);
  g.connect("a", "c", 20);
  assert((g.verticesSize() == 5) && "graph number of vertices");
  assert((g.edgesSize() == 3) && "graph number of edges");
  assert((g.vertexDegree("a") == 3) && "vertex number of edges");
  assert((g.vertexDegree("c") == 0) && "no outgoing edges c");
  assert((g.vertexDegree("xxx") == -1) && "no edges for xxx");
  assert(!g.contains("xxx") && "xxx not in graph");
  assert(g.contains("a") && "a in graph");

  //   // check that they are sorted based on edge end label
  assert(g.getEdgesAsString("a") == "b(10),c(20),d(40)");
  //   // disconnect non-existent edge/vertex
  assert(!g.disconnect("a", "e") && "disconnecting non-existent vertex");
  assert((g.edgesSize() == 3) && "disconnected nonexisting");
  assert(g.disconnect("a", "c") && "a-c disconnect");
  assert((g.edgesSize() == 2) && "number of edges after disconnect");
  assert((g.vertexDegree("a") == 2) && "a has 2 edges");
  assert(g.getEdgesAsString("a") == "b(10),d(40)" && "removing middleedge");
  assert(!g.disconnect("z", "y"));
}
void testNonDirectGraphBasic() {
  cout << "testNonDirectGraphBasic" << endl;
  Graph g(false);
  assert(g.add("a") && "add vertex a");
  assert(g.add("b") && "add vertex b");
  assert(g.add("c") && "add vertex c");
  assert(g.add("d") && "add vertex d");
  assert(g.add("e") && "add vertex e");
  assert(!g.add("b") && "b added twice");
  assert(g.connect("a", "b", 10) && "connect a b");
  assert(!g.connect("a", "b", 50) && "duplicate connect a b");
  assert(!g.connect("a", "a", 1) && "connect a to itself");
  g.connect("a", "d", 40);
  g.connect("a", "c", 20);
  assert((g.verticesSize() == 5) && "graph number of vertices");
  assert((g.edgesSize() == 6) && "graph number of edges");
  assert((g.vertexDegree("a") == 3) && "vertex number of edges");
  assert((g.vertexDegree("c") == 1) && "1 outgoing edges c");
  assert((g.vertexDegree("xxx") == -1) && "no edges for xxx");
  assert(!g.contains("xxx") && "xxx not in graph");
  assert(g.contains("a") && "a in graph");

  //   // check that they are sorted based on edge end label
  assert(g.getEdgesAsString("a") == "b(10),c(20),d(40)");
  //   // disconnect non-existent edge/vertex
  assert(!g.disconnect("a", "e") && "disconnecting non-existent vertex");
  assert((g.edgesSize() == 6) && "disconnected nonexisting");
  assert(g.disconnect("a", "c") && "a-c disconnect");
  assert((g.edgesSize() == 4) && "number of edges after disconnect");
  assert((g.vertexDegree("a") == 2) && "a has 2 edges");
  assert(g.getEdgesAsString("a") == "b(10),d(40)" && "removing middleedge");
}

void testGraph0DFS() {
  cout << "testGraph0DFS" << endl;
  Graph g;
  if (!g.readFile("graph0.txt")) {
    return;
  }
  assert(g.contains("A") && "a in graph");
  assert(g.contains("B") && "b in graph");
  assert(g.contains("C") && "c in graph");
  assert(g.getEdgesAsString("A") == "B(1),C(8)");
  assert(g.getEdgesAsString("B") == "C(3)");
  assert(g.getEdgesAsString("C").empty());

  globalSS.str("");
  g.dfs("X", vertexPrinter);
  assert(globalSS.str().empty() && "starting from X");

  g.dfs("A", vertexPrinter);
  assert(globalSS.str() == "ABC" && "starting from A");

  globalSS.str("");
  g.dfs("B", vertexPrinter);
  assert(globalSS.str() == "BC" && "starting from B");

  globalSS.str("");
  g.dfs("C", vertexPrinter);
  assert(globalSS.str() == "C" && "starting from C");

  globalSS.str("");
  g.dfs("X", vertexPrinter);
  assert(globalSS.str().empty() && "starting from X");
}

void testGraph0BFS() {
  cout << "testGraph0BFS" << endl;
  Graph g;
  if (!g.readFile("graph0.txt")) {
    return;
  }

  globalSS.str("");
  g.bfs("A", vertexPrinter);
  assert(globalSS.str() == "ABC" && "starting from A");

  globalSS.str("");
  g.dfs("B", vertexPrinter);
  assert(globalSS.str() == "BC" && "starting from B");

  globalSS.str("");
  g.dfs("C", vertexPrinter);
  assert(globalSS.str() == "C" && "starting from C");

  globalSS.str("");
  g.dfs("X", vertexPrinter);
  assert(globalSS.str().empty() && "starting from X");
}

void testGraph0Dijkstra() {
  cout << "testGraph0Dijkstra" << endl;
  Graph g;
  if (!g.readFile("graph0.txt")) {
    return;
  }
  map<string, int> weights;
  map<string, string> previous;
  tie(weights, previous) = g.dijkstra("A");
  // cout << "Dijkstra(A) weights is " << map2string(weights) << endl;
  assert(map2string(weights) == "[B:1][C:4]" && "Dijkstra(A) weights");
  // cout << "Dijkstra(A) previous is " << map2string(previous) << endl;
  assert(map2string(previous) == "[B:A][C:B]" && "Dijkstra(A) previous");

  tie(weights, previous) = g.dijkstra("B");
  // cout << "Dijkstra(B) weights is " << map2string(weights) << endl;
  assert(map2string(weights) == "[C:3]" && "Dijkstra(B) weights");
  assert(map2string(previous) == "[C:B]" && "Dijkstra(B) previous");

  tie(weights, previous) = g.dijkstra("X");
  assert(map2string(weights).empty() && "Dijkstra(C) weights");
  assert(map2string(previous).empty() && "Dijkstra(C) previous");
}

void testGraph0NotDirected() {
  cout << "testGraph0NotDirected" << endl;
  bool isDirectional = false;
  Graph g(isDirectional);
  if (!g.readFile("graph0.txt")) {
    return;
  }

  globalSS.str("");
  g.bfs("A", vertexPrinter);
  assert(globalSS.str() == "ABC" && "starting from A");

  globalSS.str("");
  g.dfs("B", vertexPrinter);
  assert(globalSS.str() == "BAC" && "starting from B");

  globalSS.str("");
  g.dfs("C", vertexPrinter);
  assert(globalSS.str() == "CAB" && "starting from C");

  globalSS.str("");
  g.dfs("X", vertexPrinter);
  assert(globalSS.str().empty() && "starting from X");

  map<string, int> weights;
  map<string, string> previous;
  tie(weights, previous) = g.dijkstra("A");
  // cout << "Dijkstra(A) weights is " << map2string(weights) << endl;
  assert(map2string(weights) == "[B:1][C:4]" && "Dijkstra(A) weights");
  // cout << "Dijkstra(A) previous is " << map2string(previous) << endl;
  assert(map2string(previous) == "[B:A][C:B]" && "Dijkstra(A) previous");

  tie(weights, previous) = g.dijkstra("B");
  assert(map2string(weights) == "[A:1][C:3]" && "Dijkstra(B) weights");
  assert(map2string(previous) == "[A:B][C:B]" && "Dijkstra(B) previous");

  tie(weights, previous) = g.dijkstra("X");
  assert(map2string(weights).empty() && "Dijkstra(C) weights");
  assert(map2string(previous).empty() && "Dijkstra(C) previous");

  globalSS.str("");
  int mstLength = g.mstPrim("A", edgePrinter);
  assert(mstLength == 4 && "mst A is 4");
  assert(globalSS.str() == "[AB 1][BC 3]" && "mst A is [AB 1][BC 3]");

  globalSS.str("");
  mstLength = g.mstPrim("B", edgePrinter);
  assert(mstLength == 4 && "mst 4 is 4");
  assert(globalSS.str() == "[BA 1][BC 3]");

  globalSS.str("");
  mstLength = g.mstPrim("C", edgePrinter);
  assert(mstLength == 4 && "mst C is 4");
  assert(globalSS.str() == "[CB 3][BA 1]");

  globalSS.str("");
  mstLength = g.mstPrim("X", edgePrinter);
  assert(mstLength == -1 && "mst X is -1");
  assert(globalSS.str().empty() && "mst for vertex not found");
}

void testGraph1() {
  cout << "testGraph1" << endl;
  Graph g;
  if (!g.readFile("graph1.txt")) {
    return;
  }
  globalSS.str("");
  g.dfs("A", vertexPrinter);
  assert(globalSS.str() == "ABCDEFGH" && "dfs starting from A");

  globalSS.str("");
  g.bfs("A", vertexPrinter);
  assert(globalSS.str() == "ABHCGDEF" && "bfs starting from A");

  globalSS.str("");
  g.dfs("B", vertexPrinter);
  assert(globalSS.str() == "BCDEFG" && "dfs starting from B");

  globalSS.str("");
  g.bfs("B", vertexPrinter);
  assert(globalSS.str() == "BCDEFG" && "bfs starting from B");

  map<string, int> weights;
  map<string, string> previous;
  auto p = g.dijkstra("A");
  weights = p.first;
  previous = p.second;
  assert(map2string(weights) == "[B:1][C:2][D:3][E:4][F:5][G:4][H:3]" &&
         "Dijkstra(B) weights");
  assert(map2string(previous) == "[B:A][C:B][D:C][E:D][F:E][G:H][H:A]" &&
         "Dijkstra(B) previous");
}

void DijkstraTest() {
  cout << "DijkstraTest" << endl;
  Graph g(false);
  if (!g.readFile("graph5.txt")) {
    return;
  }
  map<string, int> weights;
  map<string, string> previous;
  tie(weights, previous) = g.dijkstra("A");

  assert(map2string(weights) ==
             "[B:4][C:14][D:20][E:15][F:17][G:6][H:19][I:25][Z:28]" &&
         "Dijkstra(A) weights");
  assert(map2string(previous) ==
             "[B:A][C:B][D:F][E:A][F:E][G:A][H:G][I:H][Z:I]" &&
         "Dijkstra(A) previous");
  // cout << "Dijkstra(A) weights is " << map2string(weights) << endl;
  // cout << "Dijkstra(A) previous is " << map2string(previous) << endl;
}

void PrimTest() {
  cout << "PrimTest" << endl;
  Graph g(false);
  if (!g.readFile("graph6.txt")) {
    return;
  }

  globalSS.str("");
  int mstLength = g.mstPrim("A", edgePrinter);

  // cout << mstLength << endl;
  // cout << globalSS.str() << endl;
  assert(mstLength == 27 && "mst A is 27");
  assert(globalSS.str() == "[AI 2][AF 4][FG 2][GD 5][DH 1][DC 4][CE 3][AB 6]" &&
         "mst A is [AB 1][BC 3]");
}
void KruskalTest() {
  cout << "KruskalTest" << endl;
  Graph g(false);
  if (!g.readFile("graph7.txt")) {
    return;
  }
  globalSS.str("");
  int kruskalLength = g.mstKruskal("", edgePrinter);
  assert(kruskalLength == 39);
  assert(globalSS.str() == "[AD 5][EC 5][DF 6][BA 7][EB 7][EG 9]");
  // cout << "KruskalLength: " << kruskalLength << endl;
  // cout << globalSS.str() << endl;
}

void FailTest() {
  cout << "FailTest" << endl;
  Graph g;
  g.readFile("xxx.txt");
  if (!g.readFile("graph5.txt")) {
    return;
  }

  globalSS.str("");
  Graph g1(true);
  g.readFile("graph5.txt");
  g1.mstKruskal("", edgePrinter);
}
void midterm(){
  Graph g(false);
  g.readFile("midterm.txt");
  globalSS.str("");
  g.dfs("A", vertexPrinter);
  cout << "DFS: " << globalSS.str() << endl;

  globalSS.str("");
  g.bfs("A", vertexPrinter);
  cout << "BFS: " << globalSS.str() << endl;


  map<string, int> weights;
  map<string, string> previous;
  tie(weights, previous) = g.dijkstra("A");
  cout <<"dijkstra: " <<  map2string(previous) << endl;

  globalSS.str("");
  int mstLength = g.mstPrim("A", edgePrinter);
  cout << "mstPrim weights: " << mstLength << endl;
  cout << "Prim path: " << globalSS.str() << endl;

}

void testAll() {
  // testGraphBasic();
  // testNonDirectGraphBasic();
  // testGraph0DFS();
  // testGraph0BFS();
  // testGraph0Dijkstra();
  // testGraph0NotDirected();
  // testGraph1();
  // DijkstraTest();
  // PrimTest();
  // KruskalTest();
  // FailTest();
  midterm();
}
