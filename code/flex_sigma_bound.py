import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import norm
from math import pi
import warnings
warnings.filterwarnings("error")


def radius_function(d, r, sigma, N=30000):
    """

    Calculates the Gaussian measure with ball B_d(r, 1), where the standard deviation is sigma.
    

    :param d: Dimensions.
    :param r: Distance between the center of the ball and the Gaussian.
    :param sigma: Standard deviation of the Gaussian distribution.
    :param N: Number of numerical integral points.
    :return: Probability of the Gaussian within the ball.
    
    """
    # try:
    if sigma < 1e-4: return 1
    def evaluater(t):
        return np.exp(-(r - t) ** 2 / (2 * sigma ** 2)) * gammainc((d-1)/2, (1 - t ** 2) / 2 / sigma ** 2)
    integral = 0
    for t in np.linspace(-1, 1, N):
        integral += evaluater(t) / N * 2
    integral -= (evaluater(-1) + evaluater(1)) / N
    return integral / np.sqrt(2 * pi * sigma ** 2)
    # except:
    #     raise ValueError("d={}, r={}, sigma={}".format(d, r, sigma))


def numerical_inverse(f, y, start=1., eps=1e-3):
    """
    
    Calculates the numerical inverse of monotonically increasing f at y.
    
    """
    MAXIMUM, MINIMUM = 1e3, 1e-3
    MAX_ITER = 20
    init_value = f(start)
    if init_value <= y:
        while init_value <= y:
            start *= 2
            if start >= MAXIMUM:
                return np.inf
            init_value = f(start)
        l, r = start / 2, start
    else:
        while init_value >= y:
            start *= 0.5
            init_value = f(start)
            if start <= MINIMUM:
                return 0
        l, r = start, start * 2
    iter = 0
    while 1:
        iter += 1
        mid = (l + r) / 2   
        err = f(mid) - y
        if abs(err) < eps: break
        if iter >= MAX_ITER:
            print("Warning: numerical inverse quit with error {:f}".format(err))
            break
        if err > 0:
            r = mid
        else:
            l = mid
    return mid


def derive_tighter_bound(d, sigma, lower, upper, p, R):
    # print(d, sigma, lower, upper, p, R)
    b_sigma_max = lambda r: radius_function(d, sigma ** 2 * R / (upper ** 2 - sigma ** 2) /r, sigma / r)
    b_sigma_min = lambda r: radius_function(d, sigma ** 2 * R / (sigma ** 2 - lower ** 2) /r, sigma / r)
    b_max = lambda r: radius_function(d, upper ** 2 * R/ (upper ** 2 - sigma ** 2)/r, upper / r)
    b_min = lambda r: radius_function(d, lower ** 2 * R/ (sigma ** 2 - lower ** 2)/r, lower / r)
    bound_lower = 1 - b_min(numerical_inverse(b_sigma_min, 1-p, eps=1e-6)) if sigma != lower else 1
    bound_upper = b_max(numerical_inverse(b_sigma_max, p, eps=1e-6)) if sigma != upper else 1
    # print("The bound to upper is {:.3f}, to lower is {:.3f}".format(bound_upper, bound_lower))
    return min(bound_lower, bound_upper)


def get_radius(d, sigma, lam, p):
    lower_p = lambda r: -derive_tighter_bound(d, sigma, sigma-lam*r, sigma+lam*r, p, r)
    return numerical_inverse(lower_p, -0.5, eps=1e-3)


if __name__ == "__main__":
    d = 724
    for r in np.linspace(0.3, 0.8, 6):
        upper = 0.24
        lower = 0.22
        sigma = 0.23
        n_bound = lambda p: sigma * norm.ppf(p) if p < 1 else np.inf
        p = numerical_inverse(n_bound, r, start=0.12)
        t_bound = lambda r: -derive_tighter_bound(d, sigma, lower, upper, p, r)
        print("P={:.5f}, r={:.2f}".format(p, r))
        print(numerical_inverse(t_bound, -0.5, eps=5e-3))
        print("Normal bound {:.3f}".format(sigma * norm.ppf(p)))
        # print(norm.ppf(p))
