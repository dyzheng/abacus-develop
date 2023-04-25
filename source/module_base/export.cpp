// this class used for dump data in DEBUG mode
#include "export.h"

#ifdef __DEBUG
namespace ModuleBase
{
bool out_alllog = false;

template<>
bool if_not_zero(const int& data)
{
    if(data == 0)
    {
        return false;
    }
    else
    {
        return true;
    }
}

template<>
bool if_not_zero(const double& data)
{
    if(std::abs(data) > 1e-15)
    {
        return true;
    }
    else 
    {
        return false;
    }
}

template<>
bool if_not_zero(const float& data)
{
    if(std::abs(data) > 1e-10)
    {
        return true;
    }
    else 
    {
        return false;
    }
}

template<>
bool if_not_zero(const std::complex<double>& data)
{
    if(std::abs(data.real()) > 1e-15 || std::abs(data.imag()) > 1e-15)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template<>
bool if_not_zero(const std::complex<float>& data)
{
    if(std::abs(data.real()) > 1e-15 || std::abs(data.imag()) > 1e-15)
    {
        return true;
    }
    else
    {
        return false;
    }
}

}

#endif