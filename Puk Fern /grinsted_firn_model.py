import numpy as np
from scipy.optimize import root_scalar, least_squares
from scipy.special import erf

# from scipy.special import jv
# import density_core


sec_per_year = 365.25 * 24 * 60 * 60
rhow = 1000
rhoi = 917
R = 8.31446261815324  # J/mol/K
g = 9.82


def A_fun(T):
    A1 = 3.985e-13 * np.exp(-60e3 / (R * T))
    A2 = 1.916e3 * np.exp(-139e3 / (R * T))
    return np.maximum(A1, A2)


def thermal_conductivity(rho):
    return 2.1 * (rho / rhoi) ** 2  # from Arthern   (units= W/m/K)


def r_fun(a, b):
    return 2 / (3 * a) + 3 / (2 * b)


def ab_from_r(r, baratio):
    a = (4 * baratio + 9) / (6 * baratio * r)
    b = a * baratio
    return a, b


def forward_model(sigma, a, b, A, B=0, n=3):
    p = -np.trace(sigma) / 3
    tau = sigma + p * np.eye(3)
    te2 = 0.5 * np.tensordot(tau, tau)
    se2 = a * te2 + (b * p**2) / 3
    phi = A * se2 ** ((n - 1) / 2) + B
    return phi * (a * tau - 2 * b * p * np.eye(3) / 9)


def inverse_model(e, a, b, A, n=3):
    ed = e - np.trace(e) * np.eye(3) / 3  # deviatoric strain rate
    e_e2 = np.tensordot(ed, ed) / (2 * a) + 3 * np.trace(e) ** 2 / (4 * b)
    c = A ** (-1 / n) * e_e2 ** ((1 - n) / (2 * n))
    sigma = c * (ed / a + 3 * np.trace(e) * np.eye(3) / (2 * b))
    return sigma


def leastsquares_ezz(sigma_zz, a, b, A, e1=0, e2=0, forward_model=forward_model):
    # TODO: make model parameters pass_through
    if b == 0:
        return -e1 - e2
    f = lambda sxy: forward_model(np.diag([sxy[0], sxy[1], sigma_zz]), a, b, A, 0)
    e0 = f([0, 0])[2, 2]  # ezz if sxx and syy = 0
    res = least_squares(
        lambda sxy: np.diag(f(sxy))[:-1] - np.array([e1, e2]),
        x0=np.array([0.0, 0.0]),
        x_scale=np.max(np.abs(sigma_zz), 1e3),
        f_scale=np.abs(e0),
        tr_solver="exact",
        method="lm",
    )
    e = np.diag(f(res.x))
    return e[2, 2]


def iterative_root_of_cubic(k3, k2, k1, k0):
    # returns the real root.
    # using the padé approximant method here:
    # https://math.stackexchange.com/questions/2863186/numerically-find-cubic-polynomial-roots-where-coefficients-widely-vary-in-magnit
    x = -k2 / k3  # initialize where k1 and k0 are zero (todo consider k3==0)
    dx = np.inf
    while np.abs(dx) > (np.abs(x) / 1e8 + 1e-10):  # tolerance.
        f = k3 * x**3 + k2 * x**2 + k1 * x + k0
        if f == 0:
            return x  # protect against division by zero
        fp = 3 * k3 * x**2 + 2 * k2 * x + k1
        fpp = 2 * k2 + 6 * k3 * x
        dx = 0.5 * f * (f * fpp - 2 * fp**2) / (f**2 + fp**3 - f * fp * fpp)
        x = x + dx
    return x


def fetch_real_root(k3, k2, k1, k0):
    rts = np.roots([k3, k2, k1, k0])
    rts = rts[np.isreal(rts)]  # & (np.sign(rts) == np.sign(sigma_zz))]
    return np.real(rts[0])


def closed_form_cubic_root(k3, k2, k1, k0):
    # https://medium.com/@mephisto_Dev/solving-cubic-equation-using-cardanos-method-with-python-9465f0b92277
    p = (3 * k3 * k1 - k2**2) / (3 * k3**2)
    q = (2 * k2**3 - 9 * k3 * k2 * k1 + 27 * k3**2 * k0) / (27 * k3**3)
    delta = q**2 / 4 + p**3 / 27
    u = np.cbrt(-q / 2 + np.sqrt(delta))
    v = np.cbrt(-q / 2 - np.sqrt(delta))
    return u + v - k2 / (3 * k3)


def gagliardini_ezz(sigma_zz, a, b, A, e1=0, e2=0):
    # assuming e1=0 and e2=0
    # assuming n=3 (and no additional linear term in the rheology!)
    if b == 0:
        return -e1 - e2
    r = r_fun(a, b)
    p = (e1 + e2) / (3 * a) - 3 * (e1 + e2) / (2 * b)
    Asig3 = A * sigma_zz**3
    k0 = (
        Asig3 * ((e1**2 + e2**2 - e1 * e2) / (3 * a) + 3 * (e1 + e2) ** 2 / (4 * b))
        + p**3
    )
    k1 = -p * Asig3 - 3 * p**2 * r
    k2 = 0.5 * r * Asig3 + 3 * p * r**2
    k3 = -(r**3)
    return closed_form_cubic_root(k3, k2, k1, k0)


def a_fun(rho):
    # zwinger 2007
    r = rho / rhoi
    a1 = np.exp(13.22240 - 15.78652 * r)
    a2 = (1 + (2 / 3) * (1 - r)) * (r ** (-1.5))  # assuming n=3
    return np.where(r < 0.81, a1, a2)


def b_fun(rho):  # note the factor 3 difference between zwinger and JL
    r = rho / rhoi
    b1 = np.exp(15.09371 - 20.46489 * r)
    b2 = 0.75 * (((1 - r) ** (1 / 3)) / (3 * (1 - (1 - r) ** (1 / 3)))) ** (
        1.5
    )  # assuming n=3
    return np.where(r < 0.81, b1, b2) / 3


def singlecore_fit(core):
    sigma_zz = -core.overburden * g
    w = (core.bdot - core.overburden * (core.e1 + core.e2)) / core.rho
    ezz = -w * core.drho_dz / core.rho - core.e1 - core.e2

    a = np.full_like(core.z, np.nan)
    b = np.full_like(core.z, np.nan)
    for ix in range(len(core.z)):
        this_rho = core.rho[ix]
        this_ezz = ezz[ix] / sec_per_year
        this_sigma_zz = sigma_zz[ix]
        if this_sigma_zz > -10:  # we must have some load for this to work.
            continue

        baratio = b_fun(this_rho) / a_fun(this_rho)
        try:
            sol = root_scalar(
                lambda a: gagliardini_ezz(
                    this_sigma_zz,
                    a,
                    a * baratio,
                    A_fun(273.15 + core.T),
                    core.e1 / sec_per_year,
                    core.e2 / sec_per_year,
                )
                - this_ezz,
                x0=a_fun(this_rho),
                bracket=[1e-6, 1e6],
            )
            if sol.converged:
                a[ix] = sol.root
                b[ix] = sol.root * baratio
        except ValueError:
            pass
    return r_fun(a, b)


def multicore_fit(cores, rho=np.arange(350, 890, 10.0), min_a=1.0):
    # Takes a list of DensityCore's and tries makes the best fit a and b.
    # Assuming steady state.

    # allocate space for a(rho) and b(rho)
    a = np.full_like(rho, np.nan)
    b = np.full_like(rho, np.nan)

    # make a vectorized version of the gagliardni_ezz function
    gagli_vec = np.vectorize(gagliardini_ezz)

    # we can use a sigmoid transform to enforce limits on parameter search.
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    N_cores = len(cores)
    for ix in range(len(rho)):
        this_rho = rho[ix]
        # --------------- Prepare data for fitting ------------
        # for every site we need vectors of e1,e2,sigma_zz,e_zz,T
        e1 = np.full(N_cores, np.nan)
        e2 = np.full(N_cores, np.nan)
        sigma_zz = np.full(N_cores, np.nan)
        T = np.full(N_cores, np.nan)
        e_zz = np.full(N_cores, np.nan)
        for cix, core in enumerate(cores):
            e1[cix] = core.e1 / sec_per_year
            e2[cix] = core.e2 / sec_per_year
            T[cix] = core.T
            overburden = np.interp(this_rho, core.rho, core.overburden)
            sigma_zz[cix] = -g * overburden
            drho_dz = np.interp(this_rho, core.rho, core.drho_dz)
            w = (core.bdot - overburden * (core.e1 + core.e2)) / this_rho
            e_zz[cix] = -w * drho_dz / this_rho - core.e1 - core.e2
            z = np.interp(this_rho, core.rho, core.z)

        # now we have all we need for fitting.

        if np.min(z) < 1:  # dont attempt using the model without any load.
            # near surface has no load and are affected by seasonal temperatures.
            continue

        if np.any(np.isnan(overburden)):
            # TODO: dont skip if we have more than 2 cores.
            continue

        # transform the parameter space so that it is impossible to exceed theoretical limits.
        x2a = lambda x: np.exp(x[0]) + min_a  # a>=min_a
        # x2b = lambda x: x2a(x) * sigmoid(x[1]) * 9 / 2  # b<=9a/2 NICHOLAS
        # x2b = lambda x: x2a(x) * sigmoid(x[1]) * 3 / 2  # poisson>0
        x2nu = lambda x: sigmoid(x[1]) * 0.5
        x2b = lambda x: x2a(x) * (3 - 6 * x2nu(x)) / (2 * x2nu(x) + 2)

        deviance = (
            lambda x: gagli_vec(
                sigma_zz, x2a(x), x2b(x), A_fun(273.15 + T), e1=e1, e2=e2
            )
            * sec_per_year
            - e_zz
        )
        res = least_squares(
            deviance, x0=[1, 0 + (this_rho - 600) / 200], method="lm"
        )  # the LM method works very reliably
        if res.success:
            x = res.x
            a[ix], b[ix] = x2a(x), x2b(x)
    return rho, a, b


def density_profile(
    Tm,
    Ts,
    bdot,
    rho_s=350,
    z=np.linspace(0, np.sqrt(100), 100) ** 2,
    e1=0,
    e2=0,
    A_fun=A_fun,
    a_fun=a_fun,
    b_fun=b_fun,
    thermal_conductivity_fun=thermal_conductivity,
):

    # e1 = 0/sec_per_year
    # e2 = 0/sec_per_year
    # Tm = 273.15-30
    # Ts = 25
    # rho_s = 350
    # bdot = 300 / sec_per_year #kg/m2/s

    omega = np.pi * 2 / sec_per_year
    # c = 2009  # J/kg/K heat capacity. Note: should be a function of T.
    c = 185 + 6.89 * Tm  # J/kg/K heat capacity - fukusako 1990 eqn2 (90-273K)

    M = np.zeros_like(z)  # overburden
    rho = np.full_like(z, np.nan)
    rho[0] = rho_s
    logTamplitude = np.full_like(z, np.nan)

    logTamplitude[0] = np.log(Ts)

    for ii in range(0, len(z) - 1):
        dz = z[ii + 1] - z[ii]
        sigma_zz = -M[ii] * g
        w = (bdot - M[ii] * (e1 + e2)) / rho[ii]
        if w <= 0:
            break
        a = a_fun(rho[ii])
        b = b_fun(rho[ii])

        k = thermal_conductivity_fun(rho[ii])

        curTs = np.exp(logTamplitude[ii])
        if curTs > 1e-3:
            dlogAdT = (
                np.log(A_fun(Tm + curTs)) - np.log(A_fun(Tm))
            ) / Ts  # The warm Q is the most important
            Qprime = dlogAdT * R * (Tm + curTs / 2) ** 2
            Abar = (
                A_fun(Tm + curTs)
                * (Tm + curTs)
                * erf(np.pi * np.sqrt(0.5 * Qprime * curTs / R) / (Tm + curTs))
                / np.sqrt(2 * np.pi * Qprime * curTs / R)
            )
            # print(curTs, Abar / A_fun(Tm))
        else:
            Abar = A_fun(Tm)

        ezz = gagliardini_ezz(sigma_zz, a, b, Abar, e1=e1, e2=e2)
        print(f'sigma_zz{sigma_zz}, a {a}, b{b}, Abar{Abar}')

        drho_dz = -rho[ii] * (e1 + e2 + ezz) / w

        # UPDATE RHO

        if rho[ii] - rhoi == 0:
            rho[ii + 1] = rho[ii]
        else:
            rho[ii + 1] = rhoi - np.exp(
                np.log(rhoi - rho[ii]) - (drho_dz * dz) / (rhoi - rho[ii])
            )

        if rho[ii + 1] < 0:
            print("negative rho?")
            1 / 0

        M[ii + 1] = (
            M[ii] + 0.5 * (rho[ii] + rho[ii + 1]) * dz
        )  # HERES THE TRAPEZOIDAL LOAD..

        dkdz = (thermal_conductivity_fun(rho[ii + 1]) - k) / dz
        gamma = rho[ii] * c * w - dkdz
        # decay the amplitude
        if (4 * rho[ii] * c * omega) ** 2 - gamma**2 > 0:
            kz2 = (
                np.sqrt((4 * rho[ii] * c * omega) ** 2 - gamma**2) - gamma**4
            ) / (8 * k**2)
            ez = (np.sqrt(4 * k**2 * kz2 + gamma**2) - gamma) / (2 * k)
        else:
            print("NOT GOOD:", logTamplitude[ii])
            ez = 0
        if np.isnan(ez):  # TODO: add a real check ....
            print("first sqrt:", (4 * rho[ii] * c * omega) ** 2 - gamma**2)
            print("second sqrt:", 4 * k**2 * kz2 + gamma**2)
            ez = 0
        logTamplitude[ii + 1] = logTamplitude[ii] - ez * dz
    return z, rho
density_profile(Ts = 25, Tm = 273.15-30, bdot = 350/sec_per_year )

# e1 = 0/sec_per_year
# e2 = 0/sec_per_year
# Tm = 273.15-30
# Ts = 25
# rho_s = 350
# bdot = 300 / sec_per_year #kg/m2/s