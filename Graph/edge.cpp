#include "edge.h"

Edge::Edge(Vertex *from, Vertex *to, int weight) {
  from_ = from;
  to_ = to;
  weight_ = weight;
}

bool Edge::operator==(const Edge &rhs) {
  return (from_ == rhs.from_) && (to_ == rhs.to_);
}

ostream &operator<<(ostream &out, const Edge &rhs) {
  out << *rhs.to_ << '(' << rhs.weight_ << ')';
  return out;
}