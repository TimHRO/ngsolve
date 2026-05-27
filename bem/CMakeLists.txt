include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

add_library( ngsbem ${NGS_LIB_TYPE}
        ${ngsbem_object_libs}
        mptools.cpp potentialtools.cpp python_bem.cpp ngbem.cpp
        intrules_SauterSchwab.cpp analytic_integrals.cpp
        )

target_include_directories(ngsbem PRIVATE ${NETGEN_PYTHON_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../ngstd ${CMAKE_CURRENT_SOURCE_DIR}/../linalg)

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED OFF)

# Find Eigen properly
find_package(Eigen3 5.0.1 REQUIRED)

# IFGF FetchContent block
#FetchContent_Declare(
#    ifgf
#    GIT_REPOSITORY https://github.com/TimHRO/ifgf.git
#    GIT_TAG "main"
#)

set(FETCHCONTENT_SOURCE_DIR_IFGF
    /home/tim/Documents/ifgf/ifgf
)

FetchContent_Declare(
    ifgf
    SOURCE_DIR ${FETCHCONTENT_SOURCE_DIR_IFGF}
)

FetchContent_MakeAvailable(ifgf) # Use this instead of Populate for better target handling
target_include_directories(ngsbem PRIVATE
    ${ifgf_SOURCE_DIR}
)

# Link everything to your library
target_link_libraries(ngsbem PRIVATE 
    Eigen3::Eigen 
    tbb
)

target_compile_definitions(ngsbem PRIVATE EIGEN_INTERNAL_HEADER_CHECK)

target_link_libraries (ngsbem PUBLIC ngcomp ngfem ngla ngbla ngstd ${MPI_CXX_LIBRARIES} PRIVATE "$<BUILD_INTERFACE:netgen_python>" ${HYPRE_LIBRARIES})
target_link_libraries(ngsbem ${LAPACK_CMAKE_LINK_INTERFACE} "$<BUILD_INTERFACE:ngs_lapack>")
install( TARGETS ngsbem ${ngs_install_dir} )

install( FILES
        mptools.hpp potentialtools.hpp mp_coefficient.hpp
        kernels.hpp bem_diffops.hpp diffopwithfactor.hpp
        intrules_SauterSchwab.hpp analytic_integrals.hpp
        DESTINATION ${NGSOLVE_INSTALL_DIR_INCLUDE}
        COMPONENT ngsolve_devel
       )


