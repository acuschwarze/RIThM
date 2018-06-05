#include <iostream>
#include <fstream>
#include <string.h>
#include <chrono>
#include <numeric>
#include <vector>
#include <map>
#include <tr1/random>
//#include <boost/math/special_functions/gamma.hpp>
//#include <boost/math/special_functions/digamma.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>
#include <itertools.h>

using namespace Eigen;
using namespace std;

//#ifdef DYER
//constexpr bool DYER_CONST = true;
//#else
//constexpr bool DYER_CONST = false;
//#endif

//*** PROTOTYPES **************************************************************

//*****************************************************************************
//*****************************************************************************
//*** FUNCTIONS PART 1: *******************************************************
//*** Helpful for output and debugging. ***************************************
//*****************************************************************************

/** @brief Print like in python. Why? Because I am lazy.
 */
template <typename T>
void print(T message)
{
    cout<<message<<endl;
}


/** @brief Print an instance of std::vector.
 */
template<typename T>
void printVec(const std::vector<T>& path)
{
    for (auto i = path.begin(); i != path.end(); ++i)
    {
        std::cout << *i << ' ';
    }
    cout << endl;
}


/** @brief Print elapsed time between t0 and t1.
 */
template<typename T>
void printTime(T title, 
               std::chrono::time_point<std::chrono::_V2::high_resolution_clock> t0, 
               std::chrono::time_point<std::chrono::_V2::high_resolution_clock> t1)
{  
    std::chrono::duration<double, std::milli> fp_ms = t1 - t0; 
    cout << "Time elapsed during " << title << ": "; 
    cout << fp_ms.count(); 
    cout << "ms" << endl;
}


/** @brief Print elapsed time between t0 and t1 in nanoseconds.
 */
template<typename T>
void printNanoTime(T title,
               std::chrono::time_point<std::chrono::_V2::high_resolution_clock> t0,
               std::chrono::time_point<std::chrono::_V2::high_resolution_clock> t1)
{
    std::chrono::duration<double, std::nano> fp_ms = t1 - t0;
    cout << "Time elapsed during " << title << ": ";
    cout << fp_ms.count();
    cout << "ns" << endl;
}


/** @brief Shorthand for current system time.
 */
std::chrono::time_point<std::chrono::_V2::high_resolution_clock> now()
{
    return std::chrono::_V2::high_resolution_clock::now();
}


/** @brief Print elements in iterable to shell with comma separation.
 *
 *  Print elements in iterable to shell with comma separation.
 *
 *  @param[in] begin Iterable.begin()
 *  @param[in] end Iterable.end()
 *  @return Number of elements in iterable
 */
template <class It>
unsigned display(It begin, It end)
{
    unsigned r = 0;
    if (begin != end)
    {
        std::cout << *begin;
        ++r;
        for (++begin; begin != end; ++begin)
        {
            std::cout << ", " << *begin;
            ++r;
        }
    }
    return r;
}


//*****************************************************************************
//*****************************************************************************
//*** FUNCTIONS PART 2: *******************************************************
//*** Other helpful shorthands. ***********************************************
//*****************************************************************************

/** @brief Constructor for std::vector<int> that is equivalent to 
          `range(lowerLimit, upperLimit, 1)` in python.
 *
 *  Constructor for std::vector<int> that is equivalent to 
 *  `range(lowerLimit, upperLimit, 1)` in python.
 *
 *  @param[in] lowerLimit
 *  @param[in] upperLimit
 *  @return standard vector with elements in range [lowerLimit, upperLimit)
 */
std::vector<int> range(int lowerLimit, int upperLimit)
{
    std::vector<int> v(upperLimit-lowerLimit);
    std::iota(v.begin(), v.end(), lowerLimit);
    return v;
}


/** @brief Vector with all integers up to `n` that are not in `vec`.
 *
 *  Vector with all integers up to `n` that are not in `vec`. Useful for
 *  constructing sets of kernel nodes for a given set of output nodes.
 *  @param[in] n
 *  @param[in] vec 
 *  @return vector with all integers up to `n` that are not in `vec`
 */
std::vector<int> complementVec(int n, std::vector<int>& vec)
{
    std::vector<int> resVec;
    for (int i=0; i < n; i++)
    {
        if (std::find(vec.begin(), vec.end(), i) == vec.end())
        {
            resVec.push_back(i);
        };
    };
    return resVec;
}


/** @brief Save content of an std::vector to a file.
 *
 *  Save content of an std::vector to a file.
 *  @param[in] filename
 *  @param[in] vec
 *  @return void
 */
template<typename T>
void saveVec(const std::string& filename, std::vector<T>& vec)
{
    ofstream myfile (filename);
    if (myfile.is_open())
    {
        //myfile << "This is a line.\n";
        //myfile << "This is another line.\n";
        myfile << vec[0];
        for(int count = 1; count < vec.size(); count ++)
        {
            myfile << ", " << vec[count];
        }
        myfile.close();
        cout << "Saved data to " << filename << "." << endl;
    }
    else cout << "Unable to open file" << endl;;
    return;
}


//*****************************************************************************
//*****************************************************************************
//*** FUNCTIONS PART 3: *******************************************************
//*** Functions for entropy calculation. **************************************
//*****************************************************************************

/** @brief Compute log-determinant of a positive-definite square matrix.
 *
 *  Compute log-determinant of a positive-definite square matrix using
 *  `determinant()` from Eigen::Dense. CAUTION: May lead to overflow or
 *  underflow for large matrices. To avoid, use `stableLogDet()` instead in
 *  those cases.
 *
 *  @param[in] m Positive-definite square matrix (Eigen::MatrixXd)
 *  @return Log-determinant of matrix (double)
 */
template <typename Derived>
double logDet(const MatrixBase<Derived>& m)
{
    return log(m.determinant());
}

/** @brief Compute log-determinant of a positive-definite square matrix.
 *
 *  Compute log-determinant of a positive-definite square matrix. This function
 *  depends on compiler settings:
 *  If compiled without flag, it uses `determinant()` from Eigen::Dense on
 *  `m/m.mean()` and rescales result after taking the logarithm. This avoids
 *  overflow in the determinant. 
 *  If compiled with -D DYER, it uses a GitHub gist by Chris Dyer that uses an 
 *  LU decomposition with partial pivoting. (In initial tests, this was 2.5%
 *  faster than using `determinant()`.)
 *
 *  @param[in] m Positive-definite square matrix (Eigen::MatrixXd)
 *  @return Log-determinant of matrix (double)
 */
template <typename Derived>
double stableLogDet(const MatrixBase<Derived>& m)
{
#ifdef DYER
        double ld = 0;
        auto lu = m.partialPivLu();
        auto& LU = lu.matrixLU();
        double c = lu.permutationP().determinant(); // -1 or 1
        double lii;
        for (unsigned i = 0; i < LU.rows(); ++i)
        {
            lii = LU(i,i);
            if (lii < double(0)) c *= -1;
            ld += log(abs(lii));
        }
        ld += log(c);
        return ld;
#else
        double scale = m.mean();
        return log((m/scale).determinant()) + (double) (m.rows()) * log(scale);
#endif
}


/** @brief Compute a size-dependent summand for entropy of a system with
 *         multivariate Cauchy distribution.
 *  
 *  An estimate of the entropy of a system of random variables with a Cauchy 
 *  law  is the sum of a `0.5*logDet` of the system's scale matrix and a term 
 *  that only depends on the size of the system.
 *
 *  For formula for entropy estimator see page 70 in:
 *  Nadarajah & Kotz: "Mathematical Properties of the Multivariate t
 *  Distribution" Acta Applicandae Mathematicae (2005) 89: 53--74.  
 *  
 *  @param[in] n0 Size of system
 *  @return Size-dependent summand for entropy (double)
 */
double CauchySummand(int n)
{
    double d_n = (double) n;
    double out = 0.5*d_n*log(M_PI);
                 // need to use boost's lgamma and digamma because (for some 
                 // readon) std::lgamma and std:digamma don't work
                 //TODO includenext 3 lines again!
                 //+boost::math::lgamma(0.5)-boost::math::lgamma(0.5*d_n+0.5)
                 //+0.5*d_n*(boost::math::digamma(0.5*d_n)
                 //          -boost::math::digamma<double>(0.5));
    // return double
    return out;
}


/** @brief Compute a size-dependent summand for entropy of a system with
 *         multivariate Gauss distribution.
 *
 *  An estimate of the entropy of a system of random variables with a Gauss
 *  law  is the sum of a `0.5*logDet` of the system's scale matrix and a term
 *  that only depends on the size of the system.
 *
 *  @param[in] n0 Size of system
 *  @return Size-dependent summand for entropy (double)
 */
double GaussSummand(int n)
{
    // return double
    return 0.5*log(2*M_PI*exp(1))*(double) n;
}


/** @brief Compute entropy for multivariate system from scale matrix and size
 *         summand.
 *
 *  Compute entropy for multivariate system from scale matrix and size summand.
 *  (See `CauchySummand()` or `GaussSummand()` for explanation of  
 *  size-dependent summands.) CAUTION: May lead to overflow or underflow for 
 *  large system sizes. To avoid, use `stableEntropy` instead in those cases.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] sizeSummand Size-dependent summand to entropy (see CauchySummand
 *                         or GaussSummand)
 *  @return entropy (double)
 */
template <typename Derived>
double entropy(const MatrixBase<Derived>& m, double sizeSummand)
{
    return sizeSummand + 0.5*logDet(m);
}


/** @brief Compute entropy for multivariate system from scale matrix and size
 *         summand.
 *
 *  Compute entropy for multivariate system from scale matrix and size summand.
 *  (See `CauchySummand()` or `GaussSummand()` for explanation of
 *  size-dependent summands.) To avoid overflow, we use `stableLogDet()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] sizeSummand Size-dependent summand to entropy (see CauchySummand
 *                         or GaussSummand)
 *  @return entropy (double)
 */
template <typename Derived>
double stableEntropy(const MatrixBase<Derived>& m, double sizeSummand)
{
    return sizeSummand + 0.5*stableLogDet(m);
}


/** @brief Compute the sum of log-determinants of size-1 subsystems.
 *
 *  Compute the sum of log-determinants of size-1 subsystems from scale matrix 
 *  by taking the sum of logs of diagonal elements of the scale matrix.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @return sum of log-determinants of size-1 subsystems (double)
 */
template <typename Derived>
double atomicLogDet(const MatrixBase<Derived>& m) 
{
    return (log((ArrayXd) m.diagonal())).sum();
}
    

//AtomicEntropy
/** @brief Compute mean entropy for size-1 subsystems of a system from its
 *         scale matrix.
 *  
 *  Compute mean entropy for size-1 subsystems of a system from its
 *  scale matrix and size summand. 
 *  
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] sizeSummand Size-dependent summand to entropy for size-1 system
 *                         (see CauchySummand or GaussSummand)
 *  @return mean of atomic entropies (double)
 */
template <typename Derived>
double atomicEntropy(const MatrixBase<Derived>& m, double sizeSummand)
{
    return (double) (m.rows()) * sizeSummand + 0.5*atomicLogDet(m);
}


/** @brief Compute sum of log-determinant for subsystems consisting of output 
 *         system and a single kernel node (atom).
 *  
 *  Compute sum of  subsystem log-determinant, where each subsystem consists of
 *  all output nodes and one kernel node. For large output sets, consider using
 *  `stableAtomOutputLogDet()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance 
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance 
 *             of std::vector<int>)
 *  @return sum of subsystem log-determinants
 */
template <typename Derived>
double atomOutputLogDet(const MatrixBase<Derived>& m, 
                        const std::vector<int>& kernel, 
                        const std::vector<int>& output)
{
    double sumLogDeterminants = 0;
    int systemSize = m.rows();
    int outSize = output.size();

    // array of output nodes + 1
    std::vector<int> indices(output.size()+1);
    std::copy(output.begin(), output.end(), indices.begin()); 

    // loop over all non-output nodes
    for (std::vector<int>::const_iterator it = kernel.begin(); 
         it!=kernel.end(); ++it)
    {
        indices.back() = *it;
        sumLogDeterminants += logDet(m(indices,indices));
    }

    return sumLogDeterminants;
}


/** @brief Compute sum of log-determinant for subsystems consisting of output
 *         system and a single kernel node (atom).
 *
 *  Compute sum of  subsystem log-determinant, where each subsystem consists of
 *  all output nodes and one kernel node. Uses `stableLogDet()` to avoid
 *  overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance
 *             of std::vector<int>)
 *  @return sum of subsystem log-determinants
 */
template <typename Derived>
double stableAtomOutputLogDet(const MatrixBase<Derived>& m,
                              const std::vector<int>& kernel, 
                              const std::vector<int>& output)
{
    double sumLogDeterminants = 0;
    int systemSize = m.rows();
    int outSize = output.size();

    // array of output indices + 1
    std::vector<int> indices(output.size()+1);
    std::copy(output.begin(), output.end(), indices.begin()); 

    // loop over all non-output indices
    for (std::vector<int>::const_iterator it = kernel.begin(); 
         it!=kernel.end(); ++it)
    {
        indices.back() = *it;
        sumLogDeterminants += stableLogDet(m(indices,indices));
    }

    return sumLogDeterminants;
}

/** @brief Compute sum of atom-output entropies.
 *
 *  Compute sum of subsystem entropies, where each subsystem consists of
 *  all output nodes and one kernel node. For large output sets, consider using
 *  `stableAtomOutputEntropy()`instead.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance
 *             of std::vector<int>)
 *  @param[in] sizeSummand Size-dependent summand to entropy for size-1 system
 *                         (see CauchySummand or GaussSummand)
 *  @return sum of atom-output entropies
 */
template <typename Derived>
double atomOutputEntropy(const MatrixBase<Derived>& m,
                         const std::vector<int>& kernel, 
                         const std::vector<int>& output, double sizeSummand)
{
    return (0.5*atomOutputLogDet(m, kernel, output)
            + (double) (kernel.size())*sizeSummand);
}


/** @brief Compute sum of atom-output entropies.
 *
 *  Compute sum of subsystem entropies, where each subsystem consists of
 *  all output nodes and one kernel node. Uses `stableEntropy()` to avoid
 *  overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance
 *             of std::vector<int>)
 *  @param[in] sizeSummand Size-dependent summand to entropy for size-1 system
 *                         (see CauchySummand or GaussSummand)
 *  @return sum of atom-output entropies
 */
template <typename Derived>
double stableAtomOutputEntropy(const MatrixBase<Derived>& m, 
                               const std::vector<int>& kernel,
                               const std::vector<int>& output, 
                               double sizeSummand)
{
    return (0.5*stableAtomOutputLogDet(m, kernel, output)
            + (double) (kernel.size())*sizeSummand);
}


//*****************************************************************************
//*****************************************************************************
//*** FUNCTORS for Hinnant's combination iterators ****************************
//*****************************************************************************

/** @brief Functor for printing permutations or combinations
 *         (Credit: Howard Hinnant).
 */
class DISPLAY_SEQUENCE
{
    // variables in this scope
    unsigned len;
    std::uint64_t count;

    // specify how an instance of this class should read its inputs
    public:
        explicit DISPLAY_SEQUENCE(unsigned l) : len(l), count(0) {}

    // boolean function for itertools
    template <class It>
        bool operator()(It first, It last)  // called for each permutation
        {
            // count the number of times this is called
            ++count;
            // print out [first, mid) surrounded with [ ... ]
            std::cout << "[ ";
            unsigned r = display(first, last);
            // If [mid, last) is not empty, then print it out too
            //     prefixed by " | "
            if (r < len)
            {
                std::cout << " | ";
                display(last, std::next(last, len - r));
            }
            std::cout << " ]\n";
            return false;
        }

    // output is count of permutations or combinations
    operator std::uint64_t() const {return count;}
};


/** @brief Functor for computing the sum of log-determinants of principal
 *         submatrices.
 */
class SUM_LOG_MINORS
{
    // variables in this scope
    MatrixXd a;
    int len;
    double sumDeterminants;

    // specify how an instance of the class should read its inputs
    public:
        explicit SUM_LOG_MINORS(MatrixXd a0, int l0) 
                 : a(a0), len(l0), sumDeterminants(0) {}

    // boolean function for itertools
    template <class It>
        bool operator()(It first, It last)  // called for each permutation
        {
            // get array of indices
            std::vector<int> indices(len);
            std::copy(first, last, indices.begin()); 

            // add determinant of submatrix to sumDeterminants
            sumDeterminants += logDet(a(indices,indices));

            // if functor always returns false, we loop over all sequences
            return false;
        }

    // output sumDeterminants
    operator double() const {return sumDeterminants;}
};


/** @brief Functor for  computing the sum of log-determinants of principal 
 *         submatrices via `stableLogDet()`.
 */
class STABLE_SUM_LOG_MINORS
{
    // variables in this scope
    MatrixXd a;
    int len;
    double sumDeterminants;

    // specify how an instance of the class should read its inputs
    public:
        explicit STABLE_SUM_LOG_MINORS(MatrixXd a0, int l0) 
                 : a(a0), len(l0), sumDeterminants(0) {}

    // boolean function for itertools
    template <class It>
        bool operator()(It first, It last)  // called for each permutation
        {
            // get array of indices
            std::vector<int> indices(len);
            std::copy(first, last, indices.begin()); 

            // add determinant of submatrix to sumDeterminants
            sumDeterminants += stableLogDet(a(indices,indices));

            // if functor always returns false, we loop over all sequences
            return false;
        }

    // output sumDeterminants
    operator double() const {return sumDeterminants;}
};


/** @brief Functor for returning list of logi-determinants of principal
 *         submatrices.
 */
class LOG_MINORS
{
    // variables in this scope
    MatrixXd a;
    int len;
    std::vector<double> logDeterminants;

    // specify how an instance of the class should read its inputs
    public:
        explicit LOG_MINORS(MatrixXd a0, int l0) : a(a0), len(l0) {}

    // boolean function for itertools
    template <class It>
        bool operator()(It first, It last)  // called for each permutation
        {
            // get array of indices
            std::vector<int> indices(len);
            std::copy(first, last, indices.begin());

            // add determinant of submatrix to sumDeterminants
            logDeterminants.push_back(logDet(a(indices,indices)));

            // if functor always returns false, we loop over all sequences
            return false;
        }

    // output sumDeterminants
    operator std::vector<double>() const {return logDeterminants;}
};


/** @brief Functor for returning list of log-determinants of principal
 *         submatrices via `stableLogDet()`.
 */
class STABLE_LOG_MINORS
{
    // variables in this scope
    MatrixXd a;
    int len;
    std::vector<double> logDeterminants;

    // specify how an instance of the class should read its inputs
    public:
        explicit STABLE_LOG_MINORS(MatrixXd a0, int l0) : a(a0), len(l0) {}

    // boolean function for itertools
    template <class It>
        bool operator()(It first, It last)  // called for each permutation
        {
            // get array of indices
            std::vector<int> indices(len);
            std::copy(first, last, indices.begin());

            // add determinant of submatrix to sumDeterminants
            logDeterminants.push_back(stableLogDet(a(indices,indices)));

            // if functor always returns false, we loop over all sequences
            return false;
        }

    // output sumDeterminants
    operator std::vector<double>() const {return logDeterminants;}
};


//*****************************************************************************
//*****************************************************************************
//*** FUNCTIONS PART 4: *******************************************************
//*** Functions for computing mean subsystem entropies. ***********************
//*****************************************************************************


/** @brief Compute mean subsystem log-determinant from a system's scale matrix.
 *
 *  Compute mean log-determinants for subsystems of fixed size. If system is
 *  large, consider using `meanRescaledLogDet()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for computing sample mean. If 0,
 *             compute population mean
 *  @param[in] gen random-number generator passed by reference (instance of 
 *             std::tr1::mt19937)
 *  @return mean subsystem log-determinant
 */
template <typename Derived>
double meanLogDet(const MatrixBase<Derived>& m, int k, int sampleSize, 
                  std::tr1::mt19937& gen)
{
    double sumLogDeterminants = 0;
    int n = m.rows();

    // create vector of integers in range [0,n)
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);

    // get sample mean of LogDets
    if (sampleSize!=0)
    {
        // initialize vector for subsystem indices
        // For some reason, the following line does not work if I use `reserve`
        std::vector<int> indices(k);

        for (int i=0; i<sampleSize; i++)
        {
            // get a random combination of nodes 
            shuffle(v.begin(), v.end(), gen);
            std::copy(v.begin(), v.begin() + k, indices.begin());

            // add determinant of submatrix to sumDeterminants
            sumLogDeterminants += logDet(m(indices,indices));
        }
        // turn sum into mean
        sumLogDeterminants = sumLogDeterminants/(double) sampleSize;
    }

    // get population mean of LogDets
    else
    {
        // get number of all combinations
        int num;
        sumLogDeterminants = for_each_combination(v.begin(),
                                                  v.begin() + k,
                                                  v.end(),
                                                  SUM_LOG_MINORS(m,k));
        // turn sum into mean
        sumLogDeterminants /= (double) count_each_combination(k, n-k);
    }

    return sumLogDeterminants;
}


/** @brief Compute mean subsystem log-determinant from a system's scale matrix.
 *
 *  Compute mean log-determinants for subsystems of fixed size.
 *  Uses `stableLogDet()` to avoid overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for computing sample mean. If 0,
 *             compute population mean
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return mean subsystem log-determinant
 */
template <typename Derived>
double meanStableLogDet(const MatrixBase<Derived>& m, int k, int sampleSize, 
                        std::tr1::mt19937& gen)
{
    double sumLogDeterminants = 0;
    int n = m.rows();

    // create vector of integers in range [0,n)
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);

    // get sample mean of LogDets
    if (sampleSize!=0)
    {   
        // initialize vector for subsystem indices
        // For some reason, the following line does not work if I use `reserve`
        std::vector<int> indices(k);

        for (int i=0; i<sampleSize; i++)
        {    
            // get a random combination of nodes 
            shuffle(v.begin(), v.end(), gen); 
            std::copy(v.begin(), v.begin() + k, indices.begin());

            // add determinant of submatrix to sumDeterminants
            sumLogDeterminants += stableLogDet(m(indices,indices));
        }
        // turn sum into mean
        sumLogDeterminants = sumLogDeterminants/(double) sampleSize;
    }

    // get population mean of LogDets
    else
    {
        // get number of all combinations
        int num;
        sumLogDeterminants = for_each_combination(v.begin(),
                                                  v.begin() + k,
                                                  v.end(),
                                                  STABLE_SUM_LOG_MINORS(m,k));
        // turn sum into mean
        sumLogDeterminants /= (double) count_each_combination(k, n-k);
    }

  return sumLogDeterminants;
}


/** @brief Compute list of log-determinants for subsystems of fixed size.
 *
 *  Compute log-determinants for subsystems of fixed size from a system's scale
 *  matrix. If system is large, consider using `allRescaledLogDets()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for making list. If 0,
 *             make list of all subsystems
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return list of subsystem log-determinant (instance of std::vector<double>)
 */
template <typename Derived>
std::vector<double> allLogDets(const MatrixBase<Derived>& m, int k, 
                               int sampleSize,
                               std::tr1::mt19937& gen)
{
    std::vector<double> logDets;
    int n = m.rows();

    // get sample mean of log-determinants
    if (sampleSize!=0)
    {
        // reserve enough space for logDets
        logDets.reserve(sampleSize);

        // create vector of integers in range [0,n)
        std::vector<int> v(m.rows());
        std::iota(v.begin(), v.end(), 0);

        // initialise vector for subsystem indices
        // For some reason, the following line does not work if I use `reserve`
        std::vector<int> indices(k);
    
        // get log-determinants
        for (int i=0; i<sampleSize; i++)
        {
            // save a random combination of integers in [0,n) to integers
            shuffle(v.begin(), v.end(), gen);
            std::copy(v.begin(), v.begin() + k, indices.begin());

            // add determinant of submatrix to sumDeterminants
            logDets.push_back(logDet(m(indices,indices)));
        }
    }

    // get all log-determinants
    else
    {
        // get number of combinations
        int num = count_each_combination(k, n-k);

        // reserve enough space for logDets
        logDets.reserve(num);

        // create vector of integers in range [0,n)
        std::vector<int> v(n);
        std::iota(v.begin(), v.end(), 0);

        // get log-determinants
        logDets = for_each_combination(v.begin(),
                                       v.begin() + k,
                                       v.end(),
                                       LOG_MINORS(m,k));
    }
    return logDets;
}


/** @brief Compute list of log-determinants for subsystems of fixed size.
 *
 *  Compute log-determinants for subsystems of fixed size from a system's scale
 *  matrix. Uses `stableLogDet()` to avoid overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for making list. If 0,
 *             make list of all subsystems
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return list of subsystem log-determinant (instance of std::vector<double>)
 */
template <typename Derived>
std::vector<double> allStableLogDets(const MatrixBase<Derived>& m, int k, 
                                     int sampleSize, 
                                     std::tr1::mt19937& gen)
{
    std::vector<double> logDets;
    int n = m.rows();

    // get sample mean of log-determinants
    if (sampleSize!=0)
    {
        // reserve enough space for logDets
        logDets.reserve(sampleSize);

        // create vector of integers in range [0,n)
        std::vector<int> v(n);
        std::iota(v.begin(), v.end(), 0);

        // initialise vector for subsystem indices
        // For some reason, the following line does not work if I use `reserve`
        std::vector<int> indices(k);

        // get log-determinants    
        for (int i=0; i<sampleSize; i++)
        {
            // save a random combination of integers in [0,n) to integers
            shuffle(v.begin(), v.end(), gen);
            std::copy(v.begin(), v.begin() + k, indices.begin());

            // add determinant of submatrix to sumDeterminants
            logDets.push_back(stableLogDet(m(indices,indices)));
        }
    }

    // get all log-determinants
    else
    {
        // get number of combinations
        int num = count_each_combination(k, n-k);

        // reserve enough space for logDets
        logDets.reserve(num);

        // create vector of integers in range [0,n)
        std::vector<int> v(n);
        std::iota(v.begin(), v.end(), 0);

        // get log-determinants
        logDets = for_each_combination(v.begin(),
                                       v.begin() + k,
                                       v.end(),
                                       STABLE_LOG_MINORS(m,k));
    }
    return logDets;
}


/** @brief Compute mean subsystem Cauchy entropy from a system's scale matrix.
 *
 *  Compute mean subsystem entropy for subsystems of fixed size and variables
 *  with Cauchy distribution. If system is large, consider using 
 *  `meanRescaledCauchyEntropy()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for computing sample mean. If 0,
 *             compute population mean
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return mean subsystem Cauchy entropy
 */
template <typename Derived>
double meanCauchyEntropy(const MatrixBase<Derived>& m, int k, int sampleSize, 
                         std::tr1::mt19937& gen)
{
    double meanLD =  meanLogDet(m, k, sampleSize, gen);
    return CauchySummand(m.rows()) + 0.5*meanLD;
}


/** @brief Compute mean subsystem Gauss entropy from a system's scale matrix.
 *
 *  Compute mean subsystem entropy for subsystems of fixed size and variables
 *  with Gauss distribution. If system is large, consider using
 *  `meanRescaledGaussEntropy()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for computing sample mean. If 0,
 *             compute population mean
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return mean subsystem Gauss entropy
 */
template <typename Derived>
double meanGaussEntropy(const MatrixBase<Derived>& m, int k, int sampleSize, 
                        std::tr1::mt19937& gen)
{
    double meanLD = meanLogDet(m, k, sampleSize, gen);
    return GaussSummand(m.rows()) + 0.5*meanLD;
}


/** @brief Compute mean subsystem Cauchy entropy from a system's scale matrix.
 *
 *  Compute mean subsystem entropy for subsystems of fixed size and variables
 *  with Cauchy distribution. Uses `stableEntropy()` to avoid overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for computing sample mean. If 0,
 *             compute population mean
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return mean subsystem Cauchy entropy
 */
template <typename Derived>
double meanStableCauchyEntropy(const MatrixBase<Derived>& m, int k, 
                               int sampleSize, 
                               std::tr1::mt19937& gen)
{
    double meanLD =  meanStableLogDet(m, k, sampleSize, gen);
    return CauchySummand(m.rows()) + 0.5*meanLD;
}


/** @brief Compute mean subsystem Gauss entropy from a system's scale matrix.
 *
 *  Compute mean subsystem entropy for subsystems of fixed size and variables
 *  with Gauss distribution. Uses `stableEntropy()` to avoid overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for computing sample mean. If 0,
 *             compute population mean
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return mean subsystem Gauss entropy
 */
template <typename Derived>
double meanStableGaussEntropy(const MatrixBase<Derived>& m, int k, 
                              int sampleSize, 
                              std::tr1::mt19937& gen)
{
    double meanLD =  meanStableLogDet(m, k, sampleSize, gen);
    return GaussSummand(m.rows()) + 0.5*meanLD;
}


/** @brief Compute list of subsystem Cauchy entropies from scale matrix.
 *
 *  Compute list of subsystem entropies for fixed-size subsystems with
 *  variables with Cauchy distribution. If system is large, consider using
 *  `allRescaledCauchyEntropies()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for making list. If 0,
 *             make list of all subsystems
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return list of subsystem Cauchy entropies (an instance of 
 *          std::vector<double>)
 */
template <typename Derived>
std::vector<double> allCauchyEntropies(const MatrixBase<Derived>& m, int k, 
                                       int sampleSize, 
                                       std::tr1::mt19937& gen)
{
    // get sizeSummand
    double sizeSummand = CauchySummand(m.rows());

    // get LogDets
    std::vector<double> LogDets =  allLogDets(m, k, sampleSize, gen);

    // get entropy by multiplying by 0.5 and adding sizeSummand
    std::for_each(LogDets.begin(), LogDets.end(), 
                  [&sizeSummand](double& d) { d=0.5*d+sizeSummand;});

    return  LogDets; 
}


/** @brief Compute list of subsystem Gauss entropies from scale matrix.
 *
 *  Compute list of subsystem entropies for fixed-size subsystems with
 *  variables with Gauss distribution. If system is large, consider using
 *  `allRescaledGaussEntropies()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for making list. If 0,
 *             make list of all subsystems
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return list of subsystem Gauss entropies (an instance of
 *          std::vector<double>)
 */
template <typename Derived>
std::vector<double> allGaussEntropies(const MatrixBase<Derived>& m, int k, 
                                      int sampleSize, 
                                      std::tr1::mt19937& gen)
{
    // get sizeSummand
    double sizeSummand = GaussSummand(m.rows());

    // get LogDets
    std::vector<double> LogDets =  allLogDets(m, k, sampleSize, gen);

    // get entropy by multiplying by 0.5 and adding sizeSummand
    std::for_each(LogDets.begin(), LogDets.end(), 
                  [&sizeSummand](double& d) { d=0.5*d+sizeSummand;});

    return LogDets; 
}


/** @brief Compute list of subsystem Cauchy entropies from scale matrix.
 *
 *  Compute list of subsystem entropies for fixed-size subsystems with
 *  variables with Cauchy distribution. Uses `stableEntropy()` to avoid
 *  overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for making list. If 0,
 *             make list of all subsystems
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return list of subsystem Cauchy entropies (an instance of
 *          std::vector<double>)
 */
template <typename Derived>
std::vector<double> allStableCauchyEntropies(const MatrixBase<Derived>& m, 
                                             int k, 
                                             int sampleSize, 
                                             std::tr1::mt19937& gen)
{
    // get sizeSummand
    double sizeSummand = CauchySummand(m.rows());

    // get LogDets
    std::vector<double> LogDets =  allStableLogDets(m, k, sampleSize, gen);

    // get entropy by multiplying by 0.5 and adding sizeSummand
    std::for_each(LogDets.begin(), LogDets.end(), 
                  [&sizeSummand](double& d) { d=0.5*d+sizeSummand;});

    return LogDets; 
}


/** @brief Compute list of subsystem Gauss entropies from scale matrix.
 *
 *  Compute list of subsystem entropies for fixed-size subsystems with
 *  variables with Gauss distribution. Uses `stableEntropy()` to avoid
 *  overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] k size of subsystems
 *  @param[in] sampleSize number of samples for making list. If 0,
 *             make list of all subsystems
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return list of subsystem Gauss entropies (an instance of
 *          std::vector<double>)
 */
template <typename Derived>
std::vector<double> allStableGaussEntropies(const MatrixBase<Derived>&  m, 
                                            int k, 
                                            int sampleSize,
                                            std::tr1::mt19937& gen)
{
    // get sizeSummand
    double sizeSummand = GaussSummand(m.rows());

    // get LogDets
    std::vector<double> LogDets =  allStableLogDets(m, k, sampleSize, gen);

    // get entropy by multiplying by 0.5 and adding sizeSummand
    std::for_each(LogDets.begin(), LogDets.end(), 
                  [&sizeSummand](double& d) { d=0.5*d+sizeSummand;});

    return LogDets; 
}


//*****************************************************************************
//*****************************************************************************
//*** FUNCTIONS PART 5: *******************************************************
//*** Functions for computing redundancy. *************************************
//*****************************************************************************

/** @brief Compute redundancy from log-determinants.
 *
 *  Compute redundancy for a bipartition of a system into kernel and output.
 *  Uses log-determinants instead of entropies to compute mutual information.
 *  (This should be faster than using entropies.) For large systems, consider
 *  using `stableLogDetRedundancy()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance
 *             of std::vector<int>)
 *  @return redundancy
 */
template <typename Derived>
double logDetRedundancy(const MatrixBase<Derived>& m, 
                        const std::vector<int> kernel,
                        const std::vector<int>& output)
{
    // sizes
    int n = m.rows(); // system size (total)
    int p = output.size(); // output size

    // kernel matrix
    MatrixXd m_kernel = m(kernel,kernel);

    // Components of redundancy
    double sumHxi = atomicLogDet(m_kernel);
    double sumHxio = atomOutputLogDet(m, kernel, output);
    double Ho = logDet(m(output,output));
    double Hx = logDet(m_kernel);
    double Hxo = logDet(m);

    return 0.5*(sumHxi - sumHxio + (double) (kernel.size()-1) * Ho - Hx + Hxo);
}


/** @brief Compute redundancy from log-determinants.
 *
 *  Compute redundancy for a bipartition of a system into kernel and output.
 *  Uses log-determinants instead of entropies to compute mutual information.
 *  (This should be faster than using entropies.) Uses `stableLogDet()` to
 *  avoid overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance
 *             of std::vector<int>)
 *  @return redundancy
 */
template <typename Derived>
double stableLogDetRedundancy(const MatrixBase<Derived>& m, 
                              const std::vector<int> kernel, 
                              const std::vector<int> output)
{
    // sizes
    int n = m.rows(); // system size (total)
    int p = output.size(); // output size
    //print("total system size");
    //print(n);

    // kernel matrix
    auto m_kernel = m(kernel,kernel);
    //print("stableLogDetRedundancy:kernel");
    //printVec(kernel);
    //print("stableLogDetRedundancy:output");
    //printVec(output);

    // Components of redundancy
    double sumHxi = atomicLogDet(m_kernel);
    //print("stableLogDetRedundancy:sumHxi");
    //print(sumHxi);

    double sumHxio = atomOutputLogDet(m, kernel, output);
    //print("stableLogDetRedundancy:sumHxio");
    //print(sumHxio);

    double Ho = logDet(m(output,output));
    //print("stableLogDetRedundancy:Ho");
    //print(Ho);

    double Hx = stableLogDet(m_kernel);
    //print("stableLogDetRedundancy:Hx");
    //print(Hx);
    //print(m_kernel.mean());
    //print((m_kernel/m_kernel.mean()).determinant());
    //print(m_kernel.determinant());

    double Hxo = stableLogDet(m);
    //print("stableLogDetRedundancy:Hxo");
    //print(Hxo);

    return 0.5*(sumHxi - sumHxio + (double) (kernel.size()-1) * Ho - Hx + Hxo);
}


//*****************************************************************************
//*****************************************************************************
//*** FUNCTORS for redundancy calculation *************************************
//*****************************************************************************

/** @brief Functor for computing sum of subsystem redundancies from 
 *  log-determinants.
 */
class SUM_LOG_DET_REDUNDANCY
{
    // variables in this scope
    MatrixXd a;
    std::vector<int> output;
    int subSize;
    std::vector<int> newOutput;
    std::vector<int> newKernel;
    double sumRedundancies;

    // specify how an instance of the class should read its inputs
    public:
        explicit SUM_LOG_DET_REDUNDANCY(MatrixXd a0, std::vector<int> output0, 
                                        int subSize0) 
                 : a(a0), output(output0), subSize(subSize0),
                   newOutput(range(0,output0.size())), 
                   newKernel(range(output.size(),output.size() + subSize0)), 
                   sumRedundancies(0){}

    // boolean function for itertools
    template <class It>
        bool operator()(It first, It last)  // called for each permutation
        {
            // get array of indices
            std::vector<int> indices(output);
            indices.insert(indices.end(), first, last);

            // add determinant of submatrix to sumDeterminants
            sumRedundancies += logDetRedundancy(a(indices,indices), newKernel,
                                                newOutput);

            // if functor always returns false, we loop over all sequences
            return false;
        }

    // output sumDeterminants
    operator double() const {return sumRedundancies;}
};


/** @brief Functor for computing sum of subsystem redundancies from 
 *  log-determinants obtained via `stableLogDet()`.
 */
class SUM_STABLE_LOG_DET_REDUNDANCY
{
    // variables in this scope
    MatrixXd a;
    std::vector<int> output;
    int subSize;
    std::vector<int> newOutput;
    std::vector<int> newKernel;
    double sumRedundancies;

    // specify how an instance of the class should read its inputs
    public:
        explicit SUM_STABLE_LOG_DET_REDUNDANCY(MatrixXd a0, 
                                                 std::vector<int> output0, 
                                                 int subSize0)
                 : a(a0), output(output0), subSize(subSize0),
                   newOutput(range(0,output0.size())), 
                   newKernel(range(output.size(),output.size() + subSize0)), 
                   sumRedundancies(0){}

    // boolean function for itertools
    template <class It>
        bool operator()(It first, It last)  // called for each permutation
        {
            // get array of indices
            std::vector<int> indices(output);
            indices.insert(indices.end(), first, last);

            // add determinant of submatrix to sumDeterminants
            sumRedundancies += stableLogDetRedundancy(a(indices,indices), 
                                                        newKernel, newOutput);

            // if functor always returns false, we loop over all sequences
            return false;
        }

    // output sumDeterminants
    operator double() const {return sumRedundancies;}
};


//*****************************************************************************
//*****************************************************************************
//*** FUNCTIONS PART 6: *******************************************************
//*** Functions for computing mean redundancy and functional redundancy. ******
//*****************************************************************************

/** @brief Compute mean redundancy from log-determinants.
 *
 *  Compute mean redundancy for a bipartition of a system into kernel and 
 *  output. Take the mean over fixed-size subsystems of kernel. The output set
 *  remains unchanged. If system is large, consider using
 *  `meanRescaledLogDetRedundancy()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance
 *             of std::vector<int>)
 *  @param[in] kernelSize size of subkernel
 *  @param[in] sampleSize number of samples for computing sample mean. If 0,
 *             compute population mean
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return mean redundancy
 */
template <typename Derived>
double meanLogDetRedundancy(const MatrixBase<Derived>& m, 
                            std::vector<int> kernel, 
                            const std::vector<int>& output, 
                            int kernelSize, int sampleSize, 
                            std::tr1::mt19937& gen)
{
    double sumRedundancies = 0;

    // sizes
    int n = m.rows(); // system size (total)
    int p = output.size(); // output size

    // get sample mean of redundancies
    if (sampleSize!=0)
    {
        // initialize list of indices for subsystem including output set
        std::vector<int> indices(p);

        // add all elements of output set to index list
        std::copy (output.begin(), output.begin() + p, indices.begin());

        // create new index list for new kernel and new output
        std::vector<int> newOutput(output.size());
        std::iota(newOutput.begin(), newOutput.end(), 0);
        std::vector<int> newKernel(kernelSize);
        std::iota(newKernel.begin(), newKernel.end(), output.size());


        for (int i=0; i<sampleSize; i++)
        {
            //choose a random combination of kernel for subKernel
            shuffle(kernel.begin(), kernel.end(), gen);

            // add subKernel to indices
            indices.insert(indices.end(), kernel.begin(), 
                           kernel.begin() + kernelSize);

            // add determinant of submatrix to sumDeterminants
            sumRedundancies += logDetRedundancy(m(indices,indices), 
                                                newKernel, newOutput);

            // remove random kernel subset again
            indices.resize(p);
        }

        // turn sum of redundancies into mean redundancy
        sumRedundancies = sumRedundancies/ (double) sampleSize;
    }

    // get population mean of LogDets
    else
    {
        // get number of all combinations
        int num;
        sumRedundancies = for_each_combination(kernel.begin(),
                                               kernel.begin() + kernelSize,
                                               kernel.end(),
                                               SUM_LOG_DET_REDUNDANCY(m,output,
                                                                  kernelSize));
        // turn sum of redundancies into mean redundancy
        sumRedundancies /= (double) count_each_combination(kernelSize, n-p);
    }

    return sumRedundancies;
}

/** @brief Compute mean redundancy from log-determinants.
 *
 *  Compute mean redundancy for a bipartition of a system into kernel and
 *  output. Take the mean over fixed-size subsystems of kernel. The output set
 *  remains unchanged. Uses `stableLogDet()' to avoid overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance
 *             of std::vector<int>)
 *  @param[in] kernelSize size of subkernel
 *  @param[in] sampleSize number of samples for computing sample mean. If 0,
 *             compute population mean
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return mean redundancy
 */
template <typename Derived>
double meanStableLogDetRedundancy(const MatrixBase<Derived>& m, 
                                  std::vector<int> kernel, 
                                  const std::vector<int>& output, 
                                  int kernelSize, int sampleSize, 
                                  std::tr1::mt19937& gen)
{
    double sumRedundancies = 0;

    // sizes
    int n = m.rows(); // system size (total)
    int p = output.size(); // output size

    // get sample mean of redundancies
    if (sampleSize!=0)
    {
        // initialize list of indices for subsystem including output set
        std::vector<int> indices(p);

        // add all elements of output set to index list
        std::copy (output.begin(), output.begin() + p, indices.begin());

        // create new index list for new kernel and new output
        std::vector<int> newOutput(output.size());
        std::iota(newOutput.begin(), newOutput.end(), 0);
        std::vector<int> newKernel(kernelSize);
        std::iota(newKernel.begin(), newKernel.end(), output.size());


        for (int i=0; i<sampleSize; i++)
        {
            //choose a random combination of kernel for subKernel
            shuffle(kernel.begin(), kernel.end(), gen);

            // add subKernel to indices
            indices.insert(indices.end(), kernel.begin(), 
                           kernel.begin()+kernelSize);

            // add determinant of submatrix to sumDeterminants
            sumRedundancies += stableLogDetRedundancy(m(indices,indices), 
                                                        newKernel, newOutput);

            // remove random kernel subset again
            indices.resize(p);
        }

        // turn sum of redundancies into mean redundancy
        sumRedundancies = sumRedundancies/ (double) sampleSize;
    }

    // get population mean of LogDets
    else
    {
        // get number of all combinations
        int num;
        sumRedundancies = for_each_combination(kernel.begin(),
                                               kernel.begin() + kernelSize,
                                               kernel.end(),
                          SUM_STABLE_LOG_DET_REDUNDANCY(m, output, 
                                                          kernelSize));
        // turn sum of redundancies into mean redundancy
        sumRedundancies /= (double) count_each_combination(kernelSize, n-p);
    }

    return sumRedundancies;
}


/** @brief Compute functional redundancy from log-determinants.
 *
 *  Compute functional redundancy for a bipartition of a system into kernel and
 *  output from log-determinants. For systems with large kernels, consider using
 *  `stableFunctionalLogDetRedundancy()`.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance
 *             of std::vector<int>)
 *  @param[in] sampleSize number of samples for computing sample mean of 
 *             subsystem redundancy. If 0, compute population mean.
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return functional redundancy
 */
template <typename Derived>
double functionalLogDetRedundancy(const MatrixBase<Derived>& m, 
                                  std::vector<int> kernel,
                                  const std::vector<int>& output, 
                                  int sampleSize, 
                                  std::tr1::mt19937& gen)
{   
    //sizes
    int systemSize = m.rows();
    int outSize = output.size();
    int kernelSize = systemSize-outSize;

    // normalisation factor
    double norm = 2.0/ ((double) systemSize - 1.0);

    // entropy of the output system    
    double Ho = logDet(m(output,output));

    // redundancy of the complete system
    double Rxo = logDetRedundancy(m, kernel, output);

    // sum of subsystem redundancies
    double Rxio = 0;
    for (int i=1; i<kernelSize-1; i++)
    {
        Rxio +=  meanLogDetRedundancy(m, kernel, output, i, sampleSize, gen);
    }

    // functional redundancy
    return Rxo-norm*Rxio;
}


/** @brief Compute functional redundancy from log-determinants.
 *
 *  Compute functional redundancy for a bipartition of a system into kernel and
 *  output from log-determinants. Uses `stableLogDet()` to avoid overflow.
 *
 *  @param[in] m Scale matrix (instance of Eigen::MatrixXd)
 *  @param[in] kernel List of indices corresponding to kernel nodes (instance
 *             of std::vector<int>)
 *  @param[in] output List of indices corresponding to output nodes (instance
 *             of std::vector<int>)
 *  @param[in] sampleSize number of samples for computing sample mean of 
 *             subsystem redundancy. If 0, compute population mean.
 *  @param[in] gen random-number generator passed by reference (instance of
 *             std::tr1::mt19937)
 *  @return functional redundancy
 */
template <typename Derived>
double stableFunctionalLogDetRedundancy(const MatrixBase<Derived>& m,
                                        std::vector<int> kernel,
                                        const std::vector<int>& output,
                                        int sampleSize,
                                        std::tr1::mt19937& gen)
{   
    //sizes
    int systemSize = m.rows();
    int outSize = output.size();
    int kernelSize = systemSize-outSize;

    // normalisation factor
    double norm = 2.0/ ((double) systemSize - 1.0);
    //print("Norm");
    //print(norm);

    // entropy of the output system    
    //double Ho = logDet(m(output,output));
    //print("Ho");
    //print(Ho);

    // redundancy of the complete system
    //printVec(kernel);
    //printVec(output);
    double Rxo = stableLogDetRedundancy(m, kernel, output); //stable
    //print("Rxo");
    //print(Rxo);

    // sum of subsystem redundancies
    double Rxio = 0;
    for (int i=1; i<kernelSize-1; i++)
    {
        Rxio +=  meanStableLogDetRedundancy(m, kernel, output, i, //stable
                                            sampleSize, gen);
    }
    //print("Rxio");
    //print(Rxio);
    double Hx = stableLogDet(m(kernel,kernel));
    double Ho = logDet(m(output,output));
    double Hxo = stableLogDet(m);

    // functional redundancy
    return (Rxo-norm*Rxio)/(Hx+Ho-Hxo);
}


//*****************************************************************************
//*****************************************************************************
//*** FUNCTIONS PART 7: *******************************************************
//*** Functions for creating and processing time-series data and correlation **
//*** matrices. ***************************************************************
//*****************************************************************************

/** @brief Read a csv file and convert to an instance of Eigen MatrixXd.
 *
 *  Read a csv file and convert to an instance of Eigen MatrixXd (RowMajor).
 *
 *  @param[in] path Path to csv file
 *  @return an instance of MatrixXd containing the date from the csv file
 */
MatrixXd csv2MatrixXd (const std::string & path)
{
    //print("in load_csv...");
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<double, Dynamic, Dynamic, RowMajor>>
           (values.data(), rows, values.size()/rows);
}


/** @brief Read a csv file and convert to an instance of Eigen MatrixXd.
 *
 *  Read a specified line of a csv file and convert to an std::vector<int>.
 *
 *  @param[in] path Path to csv file
 *  @param[in] lineNum Number of the line to be read in.
 *  @return an instance of MatrixXd containing the date from the csv file
 */
std::vector<int> csv2Vec (const std::string & path, int lineNum)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<int> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        if (rows==lineNum)
        {
            while (std::getline(lineStream, cell, ',')) 
            {
                values.push_back(std::stoi(cell));
            }
            break;
        };
        ++rows;
    }
    return values;
}


/** @brief Compute scale matrix (covariance matrix) for multivariate time
 *         series.
 *
 *  Compute scale matrix (covariance matrix) for a time interval [t1,t2) of a 
 *  multivariate time series.
 *
 *  @param[in] m Multivariate time series. Cols are variables, rows are time
 *               points.
 *  @param[in] t1 Beginning of time interval (This point is included in 
 *                interval)
 *  @param[in] t2 End of time interval (This point in NOT included in interval)
 *  @return an instance of MatrixXd containing the date from the csv file
 */
MatrixXd scaleMatrix(const MatrixXd& m, int t1, int t2)
//template<Derived> //template does not work here. used MatrixXd instead
//MatrixXd scaleMatrix(const MatrixBase<Derived>& m, int t1, int t2)
{
    if (t2==0) { t2=m.cols(); }

    //MatrixXd subMatrix = m(seqN(0,m.rows()), range(t1,t2));
    MatrixXd subMatrix = m.block(0, t1, m.rows(), t2-t1);

    MatrixXd res;
    res.noalias() = (subMatrix * subMatrix.transpose()) / (double) (t2-t1);
    //res.noalias() = (subMatrix.transpose() * subMatrix) / (double) (t2-t1);

    return res;
}


// BernoulliAdjacency
// PriceAdjacency
// solveLyapunov
// OUCorrelationMatrix


//***************************************************************************
//***************************************************************************
//*** MAIN ******************************************************************
//***************************************************************************

int main_demo()
{
    // time read-in
    auto start = std::chrono::_V2::high_resolution_clock::now();
    MatrixXd input = csv2MatrixXd("/mi/share/scratch/schwarze2017/ecog_project/ECoG_preprocessedD_full.csv");
    auto end = std::chrono::_V2::high_resolution_clock::now();
    //printTime("read-in", start, end);

    // Get scale matrix
    Matrix<double, 82, 82> sMat;
    start = std::chrono::_V2::high_resolution_clock::now();
    sMat = scaleMatrix(input, 202000,205000);
    end = std::chrono::_V2::high_resolution_clock::now();

    printTime("short-time scale matrix 3", start, end);
    print("mean of SMat");
    print(sMat.mean());
    print("det(SMat)");
    print(sMat.determinant());
    print("det(SMat/mean)");
    print((sMat/sMat.mean()).determinant());
    print("logdet(SMat/mean)");
    print(log((sMat/sMat.mean()).determinant()));
    print("logdet mean");
    print(log(sMat.mean()));
    
    // test all functions
    std::vector<int> kernel = range(4,82);
    std::vector<int> output = range(0,4);
    end = std::chrono::_V2::high_resolution_clock::now();
    printTime("assign matrix", start, end);
    print("Get random number generator.");
    int seed=42;
    std::tr1::mt19937 gen(seed);

    double d;
    d = logDet(sMat);
    print("logDet");
    print(d);
    double m = sMat.mean();
    print("mean sMat");
    print(m);
    print("(sMat/m).determinant()");
    print((sMat/m).determinant());
    print("sMat.determinant()");
    print(sMat.determinant());
    d = stableLogDet(sMat);
    print("stableLogDet");
    print(d);
    d = entropy(sMat, 1.0);
    print("entropy");
    print(d);
    d = stableEntropy(sMat, 1.0);
    print("stableEntropy");
    print(d);
    d = atomicEntropy(sMat, 1.0);
    print("atomicEntropy");
    print(d);
    d = atomOutputLogDet(sMat, kernel, output);
    print("atomOutputLogDet");
    print(d);
    d = stableAtomOutputLogDet(sMat, kernel, output);
    print("stableAtomOutputLogDet");
    print(d);
    d = atomOutputEntropy(sMat, kernel, output, 1.0);
    print("atomOutputEntropy");
    print(d);
    d = stableAtomOutputEntropy(sMat, kernel, output, 1.0);
    print("stableAtomOutputEntropy");
    print(d);
    d = meanLogDet(sMat, 4, 10, gen);
    print("meanLogDet");
    print(d);
    d = meanStableLogDet(sMat, 4, 10, gen);
    print("meanStableLogDet");
    print(d);
    
    d = meanCauchyEntropy(sMat, 5, 10, gen); 
    print("meanCauchyEntropy");
    print(d);
    d = meanGaussEntropy(sMat, 5, 10, gen); 
    print("meanGaussEntropy");
    print(d);
    d = meanStableCauchyEntropy(sMat, 5, 10, gen); 
    print("meanStableCauchyEntropy");
    print(d);
    d = meanStableGaussEntropy(sMat, 5, 10, gen); 
    print("meanStableGaussEntropy");
    print(d);
    
    d = logDetRedundancy(sMat, kernel, output);
    print("logDetRedundancy");
    print(d);
    d = stableLogDetRedundancy(sMat, kernel, output);
    print("stableLogDetRedundancy");
    print(d);
    d = meanLogDetRedundancy(sMat, kernel, output, 5, 10, gen);
    print("meanLogDetRedundancy");
    print(d);
    d = meanStableLogDetRedundancy(sMat, kernel, output, 5, 10, gen);
    print("meanStableLogDetRedundancy");
    print(d);

    //start = now();
    //d = functionalLogDetRedundancy(sMat, kernel, output, 100, gen);
    //end = now();
    //printTime("functionalLogDetRedundancy",start,end);
    //print(d);

    start = now();
    d = stableFunctionalLogDetRedundancy(sMat, kernel, output, 100, gen);
    end = now();
    printTime("stableFunctionalLogDetRedundancy",start,end);
    print(d);

    return 0;
}

