/*
 * InitManager.h
 *
 *  Created on: Jul 10, 2015
 *      Author: vvminh
 *
 * Determine the way of init centers for a very first clustring
 */

#ifndef CONSTRAINT_INITMANAGER_H_
#define CONSTRAINT_INITMANAGER_H_

#include <vector>
#include <algorithm>
#include "../utils/Eigen3.h"
#include "../utils/functionUtils.h"

namespace dml {

using namespace Eigen;

class InitManager
{
public:
	InitManager(int dim, int numClts) : nDims(dim), nClusters(numClts) {}
	~InitManager() {}

	void fillWithTotalRandomInit(const Ref<const MatrixXf>& X, MatrixXf& initCenters) {
		int nData = X.cols();
		int nEstimate = nData / nClusters;
		for (int cltId = 0; cltId < nClusters; ++cltId) {
			int rndIdx = cltId * nEstimate + randomInRange(0, nEstimate);
			initCenters.col(cltId) = X.col(rndIdx);
		}
	}

	void fillWithFarthestFirstTraversal(const Ref<const MatrixXf>& compsCentroids,
		const std::vector<float>& compsWeights, MatrixXf& initCenters) {

		calculateDistance(compsCentroids, compsWeights);
		
		// select firstly the most crowded component
		// TODO find std::max(container)
		auto maxElemIter = std::max_element(compsWeights.begin(), compsWeights.end());
		int mostCrowdedComp = std::distance(compsWeights.begin(), maxElemIter);
		// std::cout << "1st comp: " << mostCrowdedComp << ", val = " << *maxElemIter << std::endl;

		// find nClusters that satisfy farthest first traversal
		int nClusters = initCenters.cols();
		std::vector<int> compSelected = doFFT(mostCrowdedComp, nClusters);

		// copy selecte component to init center
		for (int i = 0; i < nClusters; ++i) {
			int compId = compSelected.at(i);
			initCenters.col(i) = compsCentroids.col(compId);
		}
	}

private:
	void calculateDistance(const Ref<const MatrixXf>& compsCentroids, const std::vector<float>& compsWeights) {
		int nComps = compsCentroids.cols();
		cachedDist = MatrixXf::Zero(nComps, nComps);

		for (int compId1 = 0; compId1 < nComps-1; ++compId1) {
			for (int compId2 = compId1+1; compId2 < nComps; ++compId2) {
				float d = (compsCentroids.col(compId1) - compsCentroids.col(compId2)).norm() *
						  (compsWeights.at(compId1) * compsWeights.at(compId2));
				cachedDist(compId1, compId2) = d;
				cachedDist(compId2, compId1) = d;
			}
		}
	}	

	std::vector<int> doFFT(int mostCrowdedComp, int nClusters) {
		std::vector<int> selectedSet;
		selectedSet.push_back(mostCrowdedComp);

		std::vector<int> notSelectedSet;
		int nComps = cachedDist.cols();
		for (int compId =0; compId < nComps; ++compId) {
			if (mostCrowdedComp != compId) {
				notSelectedSet.push_back(compId);
			}
		}

		while ((int)selectedSet.size() < nClusters) {
			int farthestCompId = findFarthestComp(selectedSet, notSelectedSet);
			// std::cout << farthestCompId << " selected\n";
			selectedSet.push_back(farthestCompId);
			notSelectedSet.erase(
				std::remove(notSelectedSet.begin(), notSelectedSet.end(), farthestCompId), 
				notSelectedSet.end()
			);
		}
		return selectedSet;
	}

	int findFarthestComp(const std::vector<int>& selectedSet, const std::vector<int> notSelectedSet) {
		float farthestDist = 0.0f;
		float farthestCompId = 0;
		for (const int& notSelectedCompId : notSelectedSet) {
			float dist = distanceToSet(notSelectedCompId, selectedSet);
			if (dist > farthestDist) {
				farthestDist = dist;
				farthestCompId = notSelectedCompId;
			}
		}
		return farthestCompId;
	}

	float distanceToSet(int fromCompId, const std::vector<int>& toSet) {
		if (toSet.empty()) return 0.0f;

		float minDist = 9999.9f;
		for (const int& setElem : toSet) {
			minDist = std::min(minDist, cachedDist(fromCompId, setElem));
		}
		return minDist;
	}

private:
	int nDims = 0;
	int nClusters = 0;
	MatrixXf cachedDist;
};

} /* namespace dml */

#endif /* CONSTRAINT_INITMANAGER_H_ */
