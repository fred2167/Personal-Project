#include "vertex.h"

Vertex::Vertex(string &name) { name_ = name; }

Vertex::~Vertex() {
  for (Edge *e : edges_) {
    delete e;
  }
}
bool Vertex::operator==(const Vertex &rhs) { return name_ == rhs.name_; }

bool Vertex::operator>(const Vertex &rhs) { return name_ > rhs.name_; }

bool Vertex::operator<(const Vertex &rhs) { return name_ < rhs.name_; }

ostream &operator<<(ostream &out, const Vertex &rhs) {
  out << rhs.name_;
  return out;
}
