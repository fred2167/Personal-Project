#ifndef EDGE_H
#define EDGE_H

#include "vertex.h"
#include <iostream>

using namespace std;

class Vertex;

class Edge {
  friend class Vertex;
  friend class Graph;
  friend ostream &operator<<(ostream &out, const Edge &rhs);

public:
  explicit Edge(Vertex *from, Vertex *to, int weight);

  bool operator==(const Edge &rhs);

private:
  Vertex *from_;
  Vertex *to_;
  int weight_;
};

#endif