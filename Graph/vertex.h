#ifndef VERTEX_H
#define VERTEX_H

#include "edge.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Edge;

class Vertex {
  friend class Edge;
  friend class Graph;
  friend ostream &operator<<(ostream &out, const Vertex &rhs);

public:
  explicit Vertex(string &name);

  // copy not allowed
  Vertex(const Vertex &other) = delete;

  // move not allowed
  Vertex(Vertex &&other) = delete;

  // assignment not allowed
  Vertex &operator=(const Vertex &other) = delete;

  // move assignment not allowed
  Vertex &operator=(Vertex &&other) = delete;

  ~Vertex();

  bool operator==(const Vertex &rhs);
  bool operator>(const Vertex &rhs);
  bool operator<(const Vertex &rhs);

private:
  vector<Edge *> edges_;
  string name_;
};

#endif