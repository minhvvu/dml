set(PCKMEANS_SRC
	PCKMeans.h
	PCKMeans.cpp)

add_library (pckmeans SHARED ${PCKMEANS_SRC})
target_link_libraries(pckmeans emkmeans simpleGaussian)
cotire(pckmeans)
