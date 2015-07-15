set(GLOBAL_METRIC_SRC
	GlobalMetricKMeans.h
	GlobalMetricKMeans.cpp)

add_library(globalMetricKMeans SHARED ${GLOBAL_METRIC_SRC})
target_link_libraries(globalMetricKMeans pckmeans)
cotire(globalMetricKMeans)
