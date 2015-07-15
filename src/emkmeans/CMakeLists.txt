set(EMKMEANS_SRC
	EMKMeans.h 
	EMKMeans.cpp
	EMResult.h)

add_library (emkmeans SHARED ${EMKMEANS_SRC})
target_link_libraries(emkmeans pcimpact)
cotire(emkmeans)

