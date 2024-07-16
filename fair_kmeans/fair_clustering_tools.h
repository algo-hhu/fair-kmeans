/* Sample code that reads comma separated integer values from a file
 *
 * Daniel Schmidt, 2018
 * */
/* Edited to become fair assignment algorithm. Now reads files that are
 * separated by semicolon (standard option). Different delimiter can
 * be given as second paramter.
 *
 * Main functionality of the program is now:
 * 1) Computation of fairlets.
 * 2) Performing Fair Lloyds.
 *
 * Melanie Schmidt, 2018
 * */

#pragma once

#include <math.h>

#include <lemon/static_graph.h>
#include <lemon/capacity_scaling.h>
#include <vector>

#include "point.h"

// Some typedefs to handle lemon data types and algorithms
typedef lemon::StaticDigraph Graph;
typedef lemon::CapacityScaling<Graph, int, double> CapacityScaling;
typedef std::pair<int, int> InputArc;
typedef std::vector<InputArc> InputArcList;

// Type for the return value of the fair assignment computation
typedef std::tuple<std::vector<Point>, double, bool> CentersValueChange;

// Computing fairlets. Points are paired in one point of color 0 and one of color 1, of the same weight.
// For each pair, we output the centroid (with twice the weight of one of the points).
// This operation results in a set of the same weight but half as many distinct points.
// Returns a new vector consisting of the centroids.
std::vector<Point> computeFairlets(const std::vector<std::vector<Point>> &coloredPoints, const std::vector<int> &number, const std::vector<int> &weightsum);

// Writes fairlets to a file named filename.
// Fairlets always get color 0.5 and weight 2.
void write_fairlets(const std::vector<Point> &fairlets, std::string &filename);

// Computes the fair assignment of points to a given set of centers. Returns a parent array.
// std::vector<uint> fairassignment(const std::vector<Point>& points, const std::vector<Point>& centers);
CentersValueChange compute_fair_assignment(std::vector<std::vector<Point>> &points, const std::vector<Point> &centers, const std::vector<int> &number, const std::vector<int> &weightsum, bool fillAllClusters, int seed);

struct Matching
{
    int source = 1;
    int sink = 1;
    int flow = 0;
    int l = -1;
};

struct find_id
{
    int id;
    find_id(int id) : id(id) {}
    bool operator()(Matching const &m) const
    {
        return m.source == id;
    }
};
