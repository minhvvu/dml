set(GAUSSIAN_SRC Gaussian.h Gaussian.cpp)
add_library (gaussian SHARED ${GAUSSIAN_SRC})
target_link_libraries(gaussian pcimpact)
cotire(gaussian)

add_library (simpleGaussian SHARED SimpleGaussian.cpp)
target_link_libraries(simpleGaussian gaussian)
cotire(simpleGaussian)

add_library (diagonalGaussian SHARED DiagGaussian.cpp)
target_link_libraries(diagonalGaussian gaussian pcimpact)
cotire(diagonalGaussian)

add_library (fullGaussian SHARED FullGaussian.cpp)
target_link_libraries(fullGaussian diagonalGaussian pcimpact)
cotire(fullGaussian)
