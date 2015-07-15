set(MPCKMEANS_SRC
	MPCKMeans.h
	MPCKMeans.cpp)

add_library (mpckmeans SHARED ${MPCKMEANS_SRC})
target_link_libraries(mpckmeans 
	pckmeans 
	diagonalGaussian
	fullGaussian)

cotire(mpckmeans)
