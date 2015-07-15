/*
 * MPCKMeans.h
 *
 *  Created on: Jun 24, 2015
 *      Author: vvminh
 */

#ifndef MPCKMEANS_PCKMEANS_H_
#define MPCKMEANS_PCKMEANS_H_

#include "../pckmeans/PCKMeans.h"
#include <memory>
#include <string>

namespace dml {

typedef std::shared_ptr<ConstraintsManager> ConstraintPtr;

class MPCKMeans : public PCKMeans {
public:
	MPCKMeans(const ConstMatrixRef& dataset, const int numClts,
		const std::string constraintFileName, const CovType type = COV_DIAG);

	virtual ~MPCKMeans();
	virtual void updateMixtures();
};

} /* namespace dml */

#endif /* MPCKMEANS_PCKMEANS_H_ */
