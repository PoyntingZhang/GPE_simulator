#pragma once
__device__ inline double rotation(double x, double y, double omega, double A, double t)
{
    return A*(sin(omega*t)*x+cos(omega*t)*y);
}

__device__ inline double shake(double z, double omega, double t, double A)
{
    return A*cos(omega*t)*z;
}
