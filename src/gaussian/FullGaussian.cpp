/*
 * FullGaussian.cpp
 *
 *  Created on: Jun 26, 2015
 *      Author: vvminh
 */

#ifndef GAUSSIAN_FULLGAUSSIAN_
#define GAUSSIAN_FULLGAUSSIAN_

#include "DiagGaussian.cpp"

#include <map>
#include <utility>
#include <vector>
#include <iostream>
 
namespace dml {

using namespace Eigen;

class FullGaussian : public DiagGaussian {
public:
	FullGaussian(int id, int size, int dimensions)
		: DiagGaussian(id, size, dimensions) {}

	virtual ~FullGaussian(){}

	virtual void updateConstraintImpact(const Ref<const MatrixXf>& X, 
		const std::vector<int>& vAssign, const ConstraintPtr constraints) {

		MatrixXf mlImpact = getMLImpact(X, vAssign, constraints->ML);
		MatrixXf clImpact = getCLImpact(X, vAssign, constraints->CL);
		MatrixXf covMat = updateCovMat(mlImpact, clImpact, constraints->mlConst, constraints->clConst);
		decomposeCovMat(covMat);
		calculateLogDet();

		std::cout << "\t@Cluster " << cltId << ": maxDist = " << maxDist
			<< "\t logDet = " << logDet << '\n';
	}		

protected:

	MatrixXf getMLImpact(const Ref<const MatrixXf>& X, 
		const std::vector<int>& vAssign, const ConstraintMap& ML) {
		
		MatrixXf impact = MatrixXf::Zero(nDims, nDims);
		for (const auto& it : ML) {
			const int idx1 = it.first;
			if ( (GLOBAL_GAUSSIAN_ID == this->cltId) ||
				(vAssign[idx1] == this->cltId) ){
				for (const int& idx2 : it.second) {
					if (vAssign[idx1] != vAssign[idx2]) {
						impact += (
							(X.col(idx1) - X.col(idx2)) * 
							(X.col(idx1) - X.col(idx2)).transpose()
						);
					}
				}
			}
		}
		return 0.5 * impact;
	}

	MatrixXf getCLImpact(const Ref<const MatrixXf>& X, 
		const std::vector<int>& vAssign, const ConstraintMap& CL) {

		MatrixXf impact = MatrixXf::Zero(nDims, nDims);
		int numViolation = 0;
		for (const auto& it : CL) {
			const int idx1 = it.first;
			if ( (GLOBAL_GAUSSIAN_ID == this->cltId) ||
				(vAssign[idx1] == this->cltId) ){
				for (const int& idx2 : it.second) {
					if (vAssign[idx1] == vAssign[idx2]) {
						numViolation ++;
						impact -= (
							(X.col(idx1) - X.col(idx2)) * 
							(X.col(idx1) - X.col(idx2)).transpose()
						);
					}
				}
			}
		}
		if (numViolation > 0) {
			impact += numViolation * (
				(X.col(farthest1) - X.col(farthest2)) *
				(X.col(farthest1) - X.col(farthest2)).transpose()
			);
		}
		return impact;
	}

	MatrixXf updateCovMat(const Ref<const MatrixXf>& mlImpact, const Ref<const MatrixXf>& clImpact,
		const float mlConst, const float clConst) {
		MatrixXf covMat(nDims, nDims);
		covMat = (data.colwise() - mean) * (data.colwise() - mean).transpose();
		covMat += mlConst * mlImpact;
		covMat += clConst * clImpact;
		covMat /= nSize;
		return covMat;
	}

	void decomposeCovMat(const Ref<const MatrixXf>& covMat) {
		JacobiSVD<MatrixXf> svd(covMat, ComputeThinU);
		covDiag = svd.singularValues();
		trans = svd.matrixU().transpose();

		//// check % explain:
		// float acc = 0.0f;
		// float ssum = svd.singularValues().array().sum();
		// for (int d = 0; d < nDims; ++d) {
		// 	acc += svd.singularValues()[d];
		// 	std::cout << "singular th " << d << ", val = " <<  svd.singularValues()[d]
		// 		<< ", acc = " << acc << ", percent = " << (acc / ssum) * 100 << "\n";
		// }
	}

	void calculateLogDet() {
		// logDet = 0.0f;
		logDet = -covDiag.array().log().sum() / nSize;
	}

	virtual void cacheDistPoint2Point(const ConstMatrixRef& X) {
		maxDist = 0.0f;
		MatrixXf Xp(X.rows(), X.cols());
		if (0 == distP2P.cols() && 0 == distP2P.rows()) {
			distP2P = MatrixXf(X.cols(), X.cols());
			Xp = X.colwise() - mean;
		} else {
			distP2P.setZero();
			Xp = trans.transpose()* trans * (X.colwise() - mean);//translate data to mean
		}

		for (int i = 0; i < Xp.cols() - 1; ++i) {
			for (int j = i; j < Xp.cols(); ++j) {
				const float dist = applyDistance(Xp.col(i), Xp.col(j));
				distP2P(i, j) = dist;
				distP2P(j, i) = dist;
				if (dist > maxDist) {
					maxDist = dist;
					farthest1 = i;
					farthest2 = j;
				}
			}
		}
	}

	virtual void cacheDistPoint2Mean(const ConstMatrixRef& X) {
		MatrixXf Xp(X.rows(), X.cols());
		VectorXf meanProjected(X.rows());
		if (0 == distP2M.size()) {
			distP2M = VectorXf(X.cols());
			Xp = X.colwise() - mean;
			meanProjected = mean;
		} else {
			distP2M.setZero();
			Xp = trans.transpose() * trans * (X.colwise() - mean);
		 	meanProjected = trans.transpose() * trans * mean;
		}

		for (int i = 0; i < Xp.cols(); ++i) {
			distP2M[i] = applyDistance(Xp.col(i), meanProjected);
		}
	}

protected:
	MatrixXf trans;
};

} /* namespace dml */

#endif /* GAUSSIAN_FULLGAUSSIAN_ */
