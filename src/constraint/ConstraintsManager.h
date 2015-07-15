/*
 * ConstraintsManager.h
 *
 *  Created on: Jun 12, 2015
 *      Author: vvminh
 *
 * Load pairwise constraints from files and create full list of ML and CL
 */

#ifndef PCKMEANS_CONSTRAINTSMANAGER_H_
#define PCKMEANS_CONSTRAINTSMANAGER_H_

#include <list>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "../utils/Eigen3.h"

namespace dml {

typedef std::map<int, std::list<int> > ConstraintMap;

class ConstraintsManager {
public:
	ConstraintsManager(std::string inputFileName);
	virtual ~ConstraintsManager();

	ConstraintMap ML;
	ConstraintMap CL;

	float mlConst = 0.05f;
	float clConst = 0.05f;
	int numML = 0;
	int numCL = 0;
	int nConstraintsOriginal = 0;
	int nConstraintsDeduced = 0;

	void readConstraintsFromFile();
	void refineConstraints();
	int refineAndCount(ConstraintMap& constraintsMap);
	void dumpConstraints();
	void dumpConstraints(const ConstraintMap& constraintsMap);

	std::vector<std::vector<int> > scc;//strongly connected component

	void readConnectedComponents();
	Eigen::MatrixXf genInitCentersFromML(const Eigen::Ref<const Eigen::MatrixXf>& X, int nClusters);
	Eigen::MatrixXf getComponentCenters(const Eigen::Ref<const Eigen::MatrixXf>& X);
	std::vector<float> getComponentWeights();

private:
	std::string fileName;
	static const int MUST_LINK = 1;
	static const int CANNOT_LINK = -1;
};

typedef std::shared_ptr<ConstraintsManager> ConstraintPtr;

} /* namespace dml */

#endif /* PCKMEANS_CONSTRAINTSMANAGER_H_ */
