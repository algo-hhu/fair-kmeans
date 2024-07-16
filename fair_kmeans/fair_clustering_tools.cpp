#include <lemon/static_graph.h>
#include <lemon/capacity_scaling.h>

#include <math.h>
#include <vector>
#include <iostream>
#include <random>
#include <cstddef>

#include "fair_clustering_tools.h"

// Compute fairlets by using a min cost flow (network simplex)
std::vector<Point> computeFairlets(const std::vector<std::vector<Point>> &coloredPoints, const std::vector<int> &number, const std::vector<int> &weightsum)
{

  // std::vector<int> number(2);
  // number[0] = coloredPoints[0].size();
  // number[1] = coloredPoints[1].size();

  // construct the graph:
  // complete bipartite graph with one color on each side.

  // base graph is then equipped with source and sink that
  // are connected to the partitions in the natural way
  // capactities from source / to sink are the weights
  // of the points

  // color 0 has nodes 0,..,number[0]-1
  // color 1 has nodes number[0],...,number[0]+number[1]-1
  // source is number[0]+number[1], sink is source+1

  // TODO: Fix everything with number if you want this to ever work with more than 2 colors.
  int source = number[0] + number[1];
  int sink = source + 1;
  //  std::cout << "Source:" << source << ", sink:" << sink << std::endl;

  // List of input arcs
  //  std::cout << "Number of arcs in the min cost flow network is " << number[0]*number[1]+number[0]+number[1] << "." << std::endl;
  InputArcList arcs(number[0] * number[1] + number[0] + number[1]);

  // Arcs between the point nodes
  uint l = 0;

  for (int i = 0; i < number[0]; i++)
  {
    for (int j = 0; j < number[1]; j++)
    {
      arcs[l] = InputArc(i, number[0] + j);
      l++;
    }
  }

  // Adding edges from source to partition 0 and from partition 1 to sink
  for (int i = number[0]; i < number[0] + number[1]; i++)
  {
    arcs[l] = InputArc(i, sink);
    l++;
  }
  for (int i = 0; i < number[0]; i++)
  {
    arcs[l] = InputArc(source, i);
    l++;
  }

  // Building the graph structure
  Graph g;
  g.build(number[0] * number[1] + number[0] + number[1], arcs.begin(), arcs.end());

  // Defining costs and capacities
  Graph::ArcMap<double> cost(g);
  Graph::ArcMap<uint> cap(g);

  l = 0;

  // cost of edges between point nodes is the squared distance between the points
  // capacities could be infinity, they are here limited to the minimum
  // of the weight of the two points, since that's an upper bound on the
  // maximum flow that can travel through

  // the costs are divided by 2, since that is the actual k-means cost
  // for merging two points of weight 1: 1/2 times their squared distance
  for (int i = 0; i < number[0]; i++)
  {
    for (int j = 0; j < number[1]; j++)
    {
      cost.set(g.arc(l), (Point::sd(coloredPoints[0][i], coloredPoints[1][j]) / 2));
      cap.set(g.arc(l), std::min(coloredPoints[0][i].getWeight(), coloredPoints[1][j].getWeight()));
      l++;
    }
  }

  // capacities of edges from source and to sink are
  // the point weights. Cost of this edges is 0.
  for (int i = number[0]; i < number[0] + number[1]; i++)
  {
    cap.set(g.arc(l), coloredPoints[1][i - number[0]].getWeight());
    cost.set(g.arc(l), 0);
    l++;
  }
  for (int i = 0; i < number[0]; i++)
  {
    cap.set(g.arc(l), coloredPoints[0][i].getWeight());
    cost.set(g.arc(l), 0);
    l++;
  }

  // Capacity Scaling object

  CapacityScaling cap_scaling(g);
  cap_scaling.upperMap(cap);
  cap_scaling.costMap(cost);
  cap_scaling.stSupply(g.node(source), g.node(sink), weightsum[0]);

  CapacityScaling::ProblemType status = cap_scaling.run();

  switch (status)
  {
  case CapacityScaling::INFEASIBLE:

    std::cout << "Instanz ist unzulässig" << std::endl;
    break;
  case CapacityScaling::UNBOUNDED:
    std::cout << "Instanz ist unbeschränkt" << std::endl;
    break;
  default:
    break;
  }

  // Now we read the flow in order to compute the fairlets.
  // If u units of flow go from point i to j, then we create
  // a point at centroid(i,j) with weight u.
  // Notice that the capacities ensure that we create exactly
  // w(p) copies influenced by a point p with weight w(p);
  // the total weight of the fairlets is the same as the
  // total weight of the input point set.

  // We iterate through the edges in the same order as
  // during their creation to read out the correct flow values.

  std::vector<Point> fairlets(weightsum[0]);

  l = 0;
  uint p = 0;

  for (int i = 0; i < number[0]; i++)
  {
    for (int j = 0; j < number[1]; j++)
    {
      int flowij = cap_scaling.flow(g.arc(l));
      // std::cout << "Flow " << flowij << " between "<< i << " and " << j << std::endl;
      if (flowij > 0)
      {
        // std::cout << coloredPoints[0][i].print() << std::endl;
        // std::cout << coloredPoints[1][j].print() << std::endl;
        fairlets[p] = Point::pointAtUnweightedCentroid(coloredPoints[0][i], coloredPoints[1][j], flowij);
        fairlets[p].setWeight(2);
        // std::cout << fairlets[p].print() << std::endl;
        p++;
      }
      l++;
    }
  }

  fairlets.resize(p);

  std::cout << "Fairlet assignment cost is " << cap_scaling.totalCost() << "." << std::endl;

  //  std::vector<Point> fairlets;
  return fairlets;
}

// Computes the fair assignment of points to a given set of centers. Returns new centroids.
CentersValueChange compute_fair_assignment(
    std::vector<std::vector<Point>> &points,
    const std::vector<Point> &centers,
    const std::vector<int> &number,
    const std::vector<int> &weightsum,
    bool fillAllClusters,
    int seed)
{

  // compute the edge weights
  // cubic algorithm: for all pairs (x,y) of a 0-colored and a 1-colored point,
  // compute the center for which sd(x,c)+sd(y,c) is smallest. This will later be the edge weight.
  // We also store the centers for which the value is minimized,
  // so we can later construct an actual clustering from the min cost flow.

  std::vector<std::vector<double>> edgeweights(number[0], std::vector<double>(number[1]));
  std::vector<std::vector<int>> edgecenters(number[0], std::vector<int>(number[1]));

  for (int i = 0; i < number[0]; i++)
  {
    for (int j = 0; j < number[1]; j++)
    {
      double minij = std::numeric_limits<double>::infinity();
      edgecenters[i][j] = -1;
      for (uint l = 0; l < centers.size(); l++)
      {
        double distil = Point::sd(points[0][i], centers[l]);
        double distjl = Point::sd(points[1][j], centers[l]);
        double ndistij = distil + distjl;
        if (ndistij < minij)
        {
          edgecenters[i][j] = l;
          minij = ndistij;
        }
      }
      edgeweights[i][j] = minij;
    }
  }

  // base graph:
  // complete bipartite graph with one color on each side.
  // color 0 has nodes 0,..,cnum-1
  // color 1 has nodes cnum,..2cnum-1

  // base graph is then equipped with source and sink
  // and unit edges to model that there is one unit at each point

  // source and sink will be placed behind the two partitions
  int source = number[0] + number[1];
  int sink = source + 1;
  //  std::cout << "Source:" << source << ", sink:" << sink << std::endl;

  // build min cost graph for assignment computation

  // List of input arcs
  //  std::cout << "Number of arcs in the min cost flow network is " << number[0]*number[1]+number[0]+number[1] << "." << std::endl;
  InputArcList arcs(number[0] * number[1] + number[0] + number[1]);

  uint l = 0;

  for (int i = 0; i < number[0]; i++)
  {
    for (int j = 0; j < number[1]; j++)
    {
      arcs[l] = InputArc(i, number[0] + j);
      l++;
    }
  }

  // Adding edges from source to partition 0 and from partition 1 to sink
  for (int i = number[0]; i < number[0] + number[1]; i++)
  {
    arcs[l] = InputArc(i, sink);
    l++;
  }
  for (int i = 0; i < number[0]; i++)
  {
    arcs[l] = InputArc(source, i);
    l++;
  }

  // Building the actual graph
  Graph g;
  g.build(number[0] * number[1] + number[0] + number[1], arcs.begin(), arcs.end());
  Graph::ArcMap<double> cost(g);
  Graph::ArcMap<int> cap(g);

  // Setting the costs and capacities

  l = 0;
  for (int i = 0; i < number[0]; i++)
  {
    for (int j = 0; j < number[1]; j++)
    {
      cost.set(g.arc(l), edgeweights[i][j]);
      cap.set(g.arc(l), std::min(points[0][i].getWeight(), points[1][j].getWeight()));
      l++;
    }
  }

  for (int i = number[0]; i < number[0] + number[1]; i++)
  {
    cap.set(g.arc(l), points[1][i - number[0]].getWeight());
    cost.set(g.arc(l), 0);
    l++;
  }
  for (int i = 0; i < number[0]; i++)
  {
    cap.set(g.arc(l), points[0][i].getWeight());
    cost.set(g.arc(l), 0);
    l++;
  }

  // Capacity Scaling object

  CapacityScaling cap_scaling(g);
  cap_scaling.upperMap(cap);
  cap_scaling.costMap(cost);
  cap_scaling.stSupply(g.node(source), g.node(sink), weightsum[0]);

  CapacityScaling::ProblemType status = cap_scaling.run();

  CentersValueChange result;
  std::get<0>(result) = centers;
  std::get<2>(result) = false;

  if (status == CapacityScaling::INFEASIBLE)
  {
    std::get<1>(result) = -3.0;
    return result;
  }

  if (status == CapacityScaling::UNBOUNDED)
  {
    std::get<1>(result) = -4.0;
    return result;
  }

  // Now we read the flow in order to recover the assignment.
  // If u units of flow go from point i to j, then u of the weight
  // of i and j is assigned to the center associated with the
  // edge (i,j).
  //
  // We also include the recomputation step directly into
  // this recovery step: We compute the centroids of the
  // newly formed clusters. These centroids will then
  // be the centers for the next iteration, when
  // the assignment is called again by the framing LLoyd's algorithm.
  //
  // We iterate through the edges in the same order as
  // during their creation to read out the correct flow values.

  // An array to store the sum of all points assigned to each center
  // We will have to check that the doubles in the point coordinates
  // of this array do not overflow. (in safe mode)
  std::vector<Point> pointsums(centers.size());
  std::vector<int> ncluster(centers.size());
  for (size_t i = 0; i < centers.size(); ++i)
  {
    ncluster[i] = 0;
  }

  l = 0;
  int centerindex = 0;

  std::mt19937 rng(seed); // random-number engine used (Mersenne-Twister in this case)
  std::uniform_int_distribution<int> uni(0, points[0].size() - 1); // guaranteed unbiased

  // Choose k - 1 random integers between 0 and the size of the blue points
  // Such that we can use them in case our clusters are empty
  std::vector<Matching> matchingPairs(centers.size() - 1);

  for (size_t i = 0; i < centers.size() - 1; ++i)
  {
    auto randomInteger = uni(rng);
    while (std::find_if(matchingPairs.begin(), matchingPairs.end(), find_id(randomInteger)) != matchingPairs.end())
    {
      randomInteger = uni(rng);
    }
    matchingPairs[i].source = randomInteger;
  }

  for (int i = 0; i < number[0]; i++)
  {
    size_t iPosition = std::find_if(matchingPairs.begin(), matchingPairs.end(), find_id(i)) - matchingPairs.begin();
    for (int j = 0; j < number[1]; j++)
    {
      int flowij = cap_scaling.flow(g.arc(l));
      if (flowij > 0)
      {
        Point ijcentroid = Point::pointAtUnweightedCentroid(points[0][i], points[1][j], 2 * flowij);
        centerindex = edgecenters[i][j];
        points[0][i].setLabel(centerindex);
        points[1][j].setLabel(centerindex);
        if (iPosition < matchingPairs.size())
        {
          matchingPairs[iPosition].sink = j;
          matchingPairs[iPosition].flow = flowij;
          matchingPairs[iPosition].l = l;
        }

        if (ncluster[centerindex] == 0)
        {
          pointsums[centerindex] = ijcentroid;
        }
        else
        {
          pointsums[centerindex].addPoint(ijcentroid, false);
        }
        ncluster[centerindex] += flowij;
      }
      l++;
    }
  }

  // std::cout << std::endl;
  double totalCost = cap_scaling.totalCost();
  int replaceLocations = 0;
  for (size_t i = 0; i < centers.size(); ++i)
  {
    if (fillAllClusters && ncluster[i] == 0)
    {
      // std::cout << "Empty center: " << i << std::endl;

      // Old version: Only move one point to the empty cluster
      //
      // auto random_integer = uni(rng);
      // pointsums[i] = points[0][random_integer];
      // ncluster[i] = 1;
      // std::cout << "Moving points [" << random_integer << "] to " << i << std::endl;
      //

      int random_integer = matchingPairs[replaceLocations].source;
      int paired_integer = matchingPairs[replaceLocations].sink;

      // std::cout << "Moving points [" << random_integer << ", " << paired_integer << "] to " << i << std::endl;

      pointsums[i] = Point::pointAtUnweightedCentroid(
          points[0][random_integer],
          points[1][paired_integer],
          2 * matchingPairs[replaceLocations].flow);

      ncluster[i] = 2;

      points[0][random_integer].setLabel(i);
      points[1][paired_integer].setLabel(i);

      // We need to update the cost
      // First we remove the old cost of this pair
      totalCost -= cost[g.arc(matchingPairs[replaceLocations].l)];
      // Then we add the new cost, which is the distance between the two points and the new center
      totalCost += Point::sd(points[0][random_integer], pointsums[i]) + Point::sd(points[1][paired_integer], pointsums[i]);

      replaceLocations++;
    }
  }

  for (size_t i = 0; i < centers.size(); ++i)
  {
    pointsums[i].divideCoords(ncluster[i]);
  }

  std::get<0>(result) = pointsums;
  std::get<1>(result) = totalCost;
  std::get<2>(result) = true;

  return result;
}
