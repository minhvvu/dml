set(CONSTRAINT_SRC
	ConstraintsManager.h
	ConstraintsManager.cpp
	InitManager.h)

add_library (pcimpact SHARED ${CONSTRAINT_SRC})
target_link_libraries(pcimpact)
cotire(pcimpact)
