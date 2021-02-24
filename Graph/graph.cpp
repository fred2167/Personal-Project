#include "graph.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

// constructor, empty graph
// directionalEdges defaults to true
Graph::Graph(bool directionalEdges) { directionalEdges_ = directionalEdges; }

// destructor
Graph::~Graph() {
  auto iter = vertexes_.begin();
  auto end = vertexes_.end();
  while (iter != end) {
    delete iter->second;
    iter->second = nullptr;
    iter++;
  }
}

// @return total number of vertices
int Graph::verticesSize() const { return vertexes_.size(); }

// @return total number of edges
int Graph::edgesSize() const {
  auto iter = vertexes_.cbegin();
  auto end = vertexes_.cend();
  size_t edgeSize = 0;

  while (iter != end) {
    edgeSize += iter->second->edges_.size();
    iter++;
  }
  return edgeSize;
}
Vertex *Graph::findVertex(const string &label) const {
  auto iter = vertexes_.find(label);

  if (iter == vertexes_.end()) {
    return nullptr;
  }
  return iter->second;
}

// @return number of edges from given vertex, -1 if vertex not found
int Graph::vertexDegree(const string &label) const {
  Vertex *vertexPtr = findVertex(label);

  if (vertexPtr == nullptr) {
    return -1;
  }
  return vertexPtr->edges_.size();
}

// find vertex, if not found create the vertex
Vertex *Graph::findMakeVertex(const string &label) {
  Vertex *vertexPtr = findVertex(label);

  // NOT found, then insert vertex
  if (vertexPtr == nullptr) {
    // make a copy of string since the constructor
    // of vertex cannot take a const copy
    string copy = label;
    auto *item = new Vertex(copy);
    vertexes_[label] = item;
    return item;
  }

  // vertex already exist
  return vertexPtr;
}

// @return true if vertex added, false if it already is in the graph
bool Graph::add(const string &label) {
  size_t sizeBefore = vertexes_.size();
  findMakeVertex(label);
  return sizeBefore != vertexes_.size();
}

/** return true if vertex already in graph */
bool Graph::contains(const string &label) const {
  return (findVertex(label) != nullptr);
}

// sort vector of edges alphabatically by end vertex
bool Graph::edgesSortByEndVertexAsn(Edge *e1, Edge *e2) {
  return *(e1->to_) < *(e2->to_);
}
bool Graph::edgesSortByEndVertexDes(Edge *e1, Edge *e2) {
  return *(e1->to_) > *(e2->to_);
}

// @return string representing edges and weights, "" if vertex not found
// A-3->B, A-5->C should return B(3),C(5)
string Graph::getEdgesAsString(const string &label) const {
  Vertex *vertexPtr = findVertex(label);

  if (vertexPtr == nullptr || vertexPtr->edges_.empty()) {
    return "";
  }

  sort(vertexPtr->edges_.begin(), vertexPtr->edges_.end(),
       edgesSortByEndVertexAsn);

  stringstream ss;
  ss << *(vertexPtr->edges_[0]);
  for (int i = 1; i < vertexPtr->edges_.size(); i++) {
    ss << ',' << *(vertexPtr->edges_[i]);
  }
  return ss.str();
}

bool Graph::connect(const string &from, const string &to, int weight) {
  return connect(from, to, weight, true);
}
// @return true if successfully connected
bool Graph::connect(const string &from, const string &to, int weight,
                    bool cycle) {

  if (from == to) {
    return false;
  }

  Vertex *fromVertex = findMakeVertex(from);
  Vertex *toVertex = findMakeVertex(to);

  Edge *edge = new Edge(fromVertex, toVertex, weight);

  for (Edge *existEdge : fromVertex->edges_) {
    if (*existEdge == *edge) {
      delete edge;
      return false;
    }
  }

  fromVertex->edges_.push_back(edge);

  if (!directionalEdges_ && cycle) {
    return connect(to, from, weight, false);
  }

  return true;
}

bool Graph::disconnect(const string &from, const string &to) {
  return disconnect(from, to, true);
}

bool Graph::disconnect(const string &from, const string &to, bool cycle) {
  Vertex *fromVertexPtr = findVertex(from);
  Vertex *toVertexPtr = findVertex(to);

  // return when either vertex was not found
  if (fromVertexPtr == nullptr || toVertexPtr == nullptr) {
    return false;
  }
  for (auto edgeIter = fromVertexPtr->edges_.begin();
       edgeIter != fromVertexPtr->edges_.end(); edgeIter++) {

    Edge *edgePtr = *edgeIter;
    if (*(edgePtr->from_) == *fromVertexPtr &&
        (*edgePtr->to_) == *toVertexPtr) {

      delete edgePtr;
      fromVertexPtr->edges_.erase(edgeIter);

      if (!directionalEdges_ && cycle) {
        return disconnect(to, from, false);
      }
      return true;
    }
  }

  return false;
}

// depth-first traversal starting from given startLabel
void Graph::dfs(const string &startLabel, void visit(const string &label)) {
  Vertex *startVertexPtr = findVertex(startLabel);
  if (startVertexPtr == nullptr) {
    return;
  }

  stack<Vertex *> stack;
  unordered_set<Vertex *> visited;
  stack.push(startVertexPtr);

  while (!stack.empty()) {
    Vertex *vertexPtr = stack.top();
    stack.pop();

    if (visited.count(vertexPtr) == 1) {
      continue;
    }
    // cerr << vertexPtr->name_ << endl;
    visit(vertexPtr->name_);
    visited.insert(vertexPtr);

    // edges are sorted in the *descending* order to put into *stack*
    sort(vertexPtr->edges_.begin(), vertexPtr->edges_.end(),
         edgesSortByEndVertexDes);
    for (Edge *edge : vertexPtr->edges_) {
      stack.push(edge->to_);
    }
  }
}

// breadth-first traversal starting from startLabel
void Graph::bfs(const string &startLabel, void visit(const string &label)) {
  Vertex *startVertexPtr = findVertex(startLabel);
  if (startVertexPtr == nullptr) {
    return;
  }

  queue<Vertex *> q;
  unordered_set<Vertex *> visited;
  q.push(startVertexPtr);

  while (!q.empty()) {
    Vertex *vertexPtr = q.front();
    q.pop();

    if (visited.count(vertexPtr) == 1) {
      continue;
    }
    // cerr << vertexPtr->name_ << endl;
    visit(vertexPtr->name_);
    visited.insert(vertexPtr);

    // edges are sorted in the *ascending* order to put into *queue*
    sort(vertexPtr->edges_.begin(), vertexPtr->edges_.end(),
         edgesSortByEndVertexAsn);
    for (Edge *edge : vertexPtr->edges_) {
      q.push(edge->to_);
    }
  }
}

// store the weights in a map
// store the previous label in a map
pair<map<string, int>, map<string, string>>
Graph::dijkstra(const string &startLabel) const {
  map<string, int> weights;
  map<string, string> previous;

  Vertex *startVertexPtr = findVertex(startLabel);

  if (startVertexPtr == nullptr) {
    return make_pair(weights, previous);
  }

  auto compare = [](pair<int, Vertex *> &p1, pair<int, Vertex *> &p2) {
    return p1.first > p2.first;
  };
  // min heap that is sort by weights
  priority_queue<pair<int, Vertex *>, vector<pair<int, Vertex *>>,
                 decltype(compare)>
      pq(compare);
  unordered_set<Vertex *> visited;

  pq.push(make_pair(0, startVertexPtr));

  // set start vertex distance to 0
  weights[startVertexPtr->name_] = 0;

  while (!pq.empty()) {

    // grab current values
    auto pair = pq.top();
    int curWeight = pair.first;
    Vertex *curVertexPtr = pair.second;
    pq.pop();

    // if current vertex has visisted, skipped
    if (visited.count(curVertexPtr) == 1) {
      continue;
    }

    visited.insert(curVertexPtr);

    for (Edge *edge : curVertexPtr->edges_) {
      Vertex *toVertex = edge->to_;

      // calculate the weigh from current vertex to the next vertex
      int toWeight = curWeight + edge->weight_;

      pq.push(make_pair(toWeight, toVertex));

      if (weights.find(toVertex->name_) == weights.end() ||
          toWeight < weights[toVertex->name_]) {
        weights[toVertex->name_] = toWeight;
        previous[toVertex->name_] = curVertexPtr->name_;
      }
    }
  }

  // delete start vertex's weight to conform with test
  weights.erase(weights.find(startLabel));

  return make_pair(weights, previous);
}

// minimum spanning tree using Prim's algorithm
int Graph::mstPrim(const string &startLabel,
                   void visit(const string &from, const string &to,
                              int weight)) const {
  Vertex *startVertexPtr = findVertex(startLabel);

  if (directionalEdges_ || startVertexPtr == nullptr) {
    return -1;
  }

  auto compare = [](Edge *e1, Edge *e2) { return e1->weight_ > e2->weight_; };

  // min heap sort by edge weights
  priority_queue<Edge *, vector<Edge *>, decltype(compare)> pq(compare);

  unordered_set<Vertex *> visited;
  visited.insert(startVertexPtr);
  for (Edge *edge : startVertexPtr->edges_) {
    pq.push(edge);
  }
  int totalWeight = 0;
  while (!pq.empty()) {
    Edge *edge = pq.top();
    int weight = edge->weight_;
    Vertex *toVertexPtr = edge->to_;
    pq.pop();

    if (visited.count(toVertexPtr) == 1) {
      continue;
    }

    visit(edge->from_->name_, edge->to_->name_, weight);
    visited.insert(toVertexPtr);
    totalWeight += weight;

    for (Edge *edge : toVertexPtr->edges_) {

      if (visited.count(edge->to_) == 1) {
        continue;
      }
      pq.push(edge);
    }
  }

  return totalWeight;
}
// recursivly find parents, given vertex
Vertex *Graph::find(map<Vertex *, Vertex *> &parents, Vertex *v) const {
  if (v == parents.at(v)) {
    return v;
  }
  return find(parents, parents.at(v));
}
bool Graph::isConnected(map<Vertex *, Vertex *> &parents, Vertex *v1,
                        Vertex *v2) const {
  return find(parents, v1) == find(parents, v2);
}

// minimum spanning tree using Kruskal's algorithm
int Graph::mstKruskal(const string &startLabel,
                      void visit(const string &from, const string &to,
                                 int weight)) const {

  if (directionalEdges_) {
    return -1;
  }

  auto pqCompare = [](Edge *e1, Edge *e2) { return e1->weight_ > e2->weight_; };

  // min heap sort by edge weights
  priority_queue<Edge *, vector<Edge *>, decltype(pqCompare)> pq(pqCompare);

  map<Vertex *, Vertex *> parents;
  for (auto &pair : vertexes_) {
    // initialize all parents as self
    parents[pair.second] = pair.second;
    for (Edge *edge : pair.second->edges_) {
      pq.push(edge);
    }
  }

  int totalWeight = 0;
  while (!pq.empty()) {
    Edge *edge = pq.top();
    int weight = edge->weight_;
    Vertex *toVertexPtr = edge->to_;
    Vertex *fromVertexPtr = edge->from_;
    pq.pop();

    if (isConnected(parents, fromVertexPtr, toVertexPtr)) {
      continue;
    }

    totalWeight += weight;
    visit(fromVertexPtr->name_, toVertexPtr->name_, weight);
    parents[toVertexPtr] = fromVertexPtr;
    // cerr << '[' << *edge->from_ << *edge->to_ << ' ' << weight << ']';
  }

  return totalWeight;
}

// read a text file and create the graph
bool Graph::readFile(const string &filename) {
  ifstream myfile(filename);
  if (!myfile.is_open()) {
    cerr << "Failed to open " << filename << endl;
    return false;
  }
  int edges = 0;
  int weight = 0;
  string fromVertex;
  string toVertex;
  myfile >> edges;
  for (int i = 0; i < edges; ++i) {
    myfile >> fromVertex >> toVertex >> weight;
    connect(fromVertex, toVertex, weight);
  }
  myfile.close();
  return true;
}
