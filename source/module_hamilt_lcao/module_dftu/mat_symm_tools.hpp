#include <iostream>
#include <vector>
#include <cmath>

namespace ModuleDFTU
{

// calculate the maximum off-diagonal element of the matrix A
void findMaxOffDiagonal(const double* A, int& p, int& q, int n) 
{
    double maxVal = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j && std::abs(A[i*n+j]) > maxVal) {
                maxVal = std::abs(A[i*n+j]);
                p = i;
                q = j;
            }
        }
    }
}

// calculate the Jacobi rotation matrix
void calcJacobiRotation(const double* A, int p, int q, double& c, double& s, int n) 
{
    double tau = (A[q*n+q] - A[p*n+p]) / (2.0 * A[p*n+q]);
    double t;

    if (tau >= 0) {
        t = 1.0 / (tau + sqrt(1.0 + tau * tau));
    } else {
        t = 1.0 / (tau - sqrt(1.0 + tau * tau));
    }

    c = 1.0 / sqrt(1.0 + t * t);
    s = t * c;
}

// decompose symmetric matrix A into P^-1 * A * P
void decomposeSymmetricMatrix(
    double* A, 
    double* P, 
    int n) 
{
    // initialize P as the identity matrix
    for (int i = 0; i < n; ++i) 
    {
        P[i*n+i] = 1.0;
    }

    double epsilon = 1e-10;
    int maxIterations = 100 * n * n;
    int iterations = 0;

    int p=0, q=0;
    double c=0, s=0;

    // iterate until convergence
    while (iterations < maxIterations) 
    {
        // find the maximum off-diagonal element
        findMaxOffDiagonal(A, p, q, n);

        // if converged
        if (std::abs(A[p*n+q]) < epsilon) {
            break;
        }

        // calculate the Jacobi rotation matrix
        calcJacobiRotation(A, p, q, c, s, n);

        // update the matrix A
        std::vector<double> A_new(A, A + n*n);
        A_new[p*n+p] = c * c * A[p*n+p] - 2.0 * s * c * A[p*n+q] + s * s * A[q*n+q];
        A_new[q*n+q] = s * s * A[p*n+p] + 2.0 * s * c * A[p*n+q] + c * c * A[q*n+q];
        A_new[p*n+q] = A_new[q*n+p] = 0.0;

        for (int i = 0; i < n; ++i) {
            if (i != p && i != q) {
                A_new[i*n+p] = A_new[p*n+i] = c * A[i*n+p] - s * A[i*n+q];
                A_new[i*n+q] = A_new[q*n+i] = s * A[i*n+p] + c * A[i*n+q];
            }
        }

        for(int i=0; i<n*n; i++)
        {
            A[i] = A_new[i];
        }

        // update the matrix P
        for (int i = 0; i < n; ++i) {
            double P_ip = P[i*n+p];
            double P_iq = P[i*n+q];
            P[i*n+p] = c * P_ip - s * P_iq;
            P[i*n+q] = s * P_ip + c * P_iq;
        }

        ++iterations;
    }
    return;
}

// restore the original matrix from P^-1 * A * P
void restoreSymmetricMatrix(
    double* A, 
    double* P, 
    int n) 
{
    double* AP = new double[n*n];

    // calculate AP
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            AP[i*n+j] = 0.0;
            for (int k = 0; k < n; ++k) {
                AP[i*n+j] += A[i*n+k] * P[k*n+j];
            }
        }
    }

    // calculate PTAP to A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i*n+j] = 0.0;
            for (int k = 0; k < n; ++k) {
                A[i*n+j] += P[k*n+i] * AP[k*n+j];
            }
        }
    }

    delete[] AP;
}

} // namespace ModuleDFTU