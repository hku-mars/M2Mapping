#pragma once

#include "Eigen/Dense"
#include "vector"

#define SIGN(x) ((x > 0) - (x < 0))

namespace type_utils {

/*
 * @\brief Rename the float type used in lib

    Default is set to be double, but user can change it to float.
*/
    typedef double decimal_t;

///Pre-allocated std::vector for Eigen using vec_E
    template<typename T>
    using vec_E = std::vector<T, Eigen::aligned_allocator<T>>;
///Eigen 1D float vector
    template<int N>
    using Vecf = Eigen::Matrix<decimal_t, N, 1>;
///Eigen 1D int vector
    template<int N>
    using Veci = Eigen::Matrix<int, N, 1>;
///MxN Eigen matrix
    template<int M, int N>
    using Matf = Eigen::Matrix<decimal_t, M, N>;
///MxN Eigen matrix with M unknown
    template<int N>
    using MatDNf = Eigen::Matrix<decimal_t, Eigen::Dynamic, N>;

///MxN Eigen matrix with N unknown
    template<int M>
    using MatMDf = Eigen::Matrix<decimal_t, M, Eigen::Dynamic>;

///Vector of Eigen 1D float vector
    template<int N>
    using vec_Vecf = vec_E<Vecf<N>>;
///Vector of Eigen 1D int vector
    template<int N>
    using vec_Veci = vec_E<Veci<N>>;

///Eigen 1D float vector of size 2
    typedef Vecf<2> Vec2f;
///Eigen 1D int vector of size 2
    typedef Veci<2> Vec2i;
///Eigen 1D float vector of size 3
    typedef Vecf<3> Vec3f;
///Eigen 1D int vector of size 3
    typedef Veci<3> Vec3i;
///Eigen 1D float vector of size 4
    typedef Vecf<4> Vec4f;
///Column vector in float of size 6
    typedef Vecf<6> Vec6f;

///Vector of type Vec2f.
    typedef vec_E<Vec2f> vec_Vec2f;
///Vector of type Vec2i.
    typedef vec_E<Vec2i> vec_Vec2i;
///Vector of type Vec3f.
    typedef vec_E<Vec3f> vec_Vec3f;
///Vector of type Vec3i.
    typedef vec_E<Vec3i> vec_Vec3i;

///2x2 Matrix in float
    typedef Matf<2, 2> Mat2f;
///3x3 Matrix in float
    typedef Matf<3, 3> Mat3f;
///4x4 Matrix in float
    typedef Matf<4, 4> Mat4f;
///6x6 Matrix in float
    typedef Matf<6, 6> Mat6f;

///Dynamic Nx1 Eigen float vector
    typedef Vecf<Eigen::Dynamic> VecDf;
///Dynamic Nx1 Eigen int vector
    typedef Veci<Eigen::Dynamic> VecDi;
///Nx2 Eigen float matrix
    typedef MatDNf<2> MatD2f;
///Nx3 Eigen float matrix
    typedef MatDNf<3> MatD3f;
///Nx4 Eigen float matrix
    typedef MatDNf<4> MatD4f;
///4xM Eigen float matrix
    typedef MatMDf<4> Mat4Df;
    typedef MatD4f MatPlanes;
///3xM Eigen float matrix
    typedef MatMDf<3> Mat3Df;
    typedef Mat3Df MatPoints;

    typedef Mat3Df PolyhedronV;
    typedef MatD4f PolyhedronH;
    typedef vec_E<PolyhedronV> PolyhedraV;
    typedef vec_E<PolyhedronH> PolyhedraH;

///Dynamic MxN Eigen float matrix
    typedef Matf<Eigen::Dynamic, Eigen::Dynamic> MatDf;

///Allias of Eigen::Affine2d
    typedef Eigen::Transform<decimal_t, 2, Eigen::Affine> Aff2f;
///Allias of Eigen::Affine3d
    typedef Eigen::Transform<decimal_t, 3, Eigen::Affine> Aff3f;


#ifndef EIGEN_QUAT
#define EIGEN_QUAT
///Allias of Eigen::Quaterniond
    typedef Eigen::Quaternion<decimal_t> Quatf;
#endif

#ifndef EIGEN_EPSILON
#define EIGEN_EPSILON
///Compensate for numerical error
    constexpr decimal_t
            epsilon_ = 1e-10; // numerical calculation error
#endif
// Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> way_pts_E(way_pts[0].data(), 3, way_pts.size());

    typedef vec_E<Vec3f> vec_Vec3f;

/// Function Retuen Code

    enum RET_CODE {
        /// FOR Planner
        FAILED = 0,
        NO_NEED = 1,
        SUCCESS = 2,
        FINISH = 3,
        NEW_TRAJ = 4,
        EMER = 5,
        OPT_FAILED,

        /// FOR Astar search
        INIT_ERROR,
        TIME_OUT,
        NO_PATH,
        REACH_GOAL,
        REACH_HORIZON,
    };

    static const std::vector<std::string> RET_CODE_STR{"FAILED", "NO_NEED", "SUCCESS",
                                                       "FINISH", "NEW_TRAJ", "EMER",
                                                       "OPT_FAILED", "INIT_ERROR", "TIME_OUT",
                                                       "NO_PATH", "REACH_GOAL", "REACH_HORIZON"};

    typedef Eigen::Matrix<double, 3, 3> StatePVA;
    typedef Eigen::Matrix<double, 3, 4> StatePVAJ;
    typedef std::pair<double, Vec3f> TimePosPair;
    typedef std::pair<Vec3f, Vec3f> Line;
    typedef std::pair<Vec3f, Quatf> Pose;

    struct RobotState {
        Vec3f p, v, a, j;
        double yaw;
        double rcv_time;
        bool rcv{false};
        Quatf q;
    };

    enum GridType {
        UNDEFINED = 0,
        UNKNOWN = 1,
        OUT_OF_MAP,
        OCCUPIED,
        KNOWN_FREE,
        FRONTIER, // The frontier is the unknown grid which is adjacent to the known free grid
    };

    const static std::vector<std::string> GridTypeStr{"UNDEFINED", "UNKNOWN", "OUT_OF_MAP", "OCCUPIED", "KNOWN_FREE"};


#ifndef LINCALIB_COLOR_H
#define LINCALIB_COLOR_H

#define RESET       "\033[0m"
#define BLACK       "\033[30m"             /* Black */
#define RED         "\033[31m"             /* Red */
#define GREEN       "\033[32m"             /* Green */
#define YELLOW      "\033[33m"             /* Yellow */
#define BLUE        "\033[34m"             /* Blue */
#define MAGENTA     "\033[35m"             /* Magenta */
#define CYAN        "\033[36m"             /* Cyan */
#define WHITE       "\033[37m"             /* White */
#define REDPURPLE   "\033[95m"             /* Red Purple */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
#define BOLDREDPURPLE   "\033[1m\033[95m"  /* Bold Red Purple */

#endif //LINCALIB_COLOR_H
}
