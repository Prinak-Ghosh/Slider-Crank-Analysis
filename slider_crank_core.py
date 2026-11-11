"""
slider_crank_core.py
--------------------
Physics for the slider-crank mechanism (single-cylinder, 4-stroke engine).

Everything that is *calculation* lives here, so the interactive simulator
(slider_crank_simulator.py) and the batch study (analysis.py) use exactly the
same equations.

Conventions
  theta : crank angle measured from top dead centre (TDC), radians.
          One full 4-stroke cycle is 0 .. 4*pi
          (intake, compression, power, exhaust - 180 deg each).
  x     : piston-pin distance from the crank centre (maximum at TDC).
  Positive force = pushes the piston towards the crank.
  kinematics() is unit-agnostic: give r and l in mm and you get mm, mm/s,
  mm/s^2; give them in metres and you get SI.  The Engine / loads() part
  works in SI only.
"""

from dataclasses import dataclass
import numpy as np


# ------------------------------------------------------------------ geometry
def check_geometry(r, l):
    """Reject geometries that cannot physically rotate."""
    if r <= 0 or l <= 0:
        raise ValueError("crank radius and rod length must be positive")
    if l <= r:
        raise ValueError("rod must be longer than the crank (l > r)")


def rod_angle(theta, r, l):
    """Connecting-rod angle phi from the cylinder axis: sin(phi) = (r/l) sin(theta)."""
    return np.arcsin(r * np.sin(theta) / l)


# ---------------------------------------------------------------- kinematics
def kinematics(theta, r, l, omega):
    """Exact piston position, velocity, acceleration for constant omega."""
    check_geometry(r, l)
    s_root = np.sqrt(l**2 - (r * np.sin(theta))**2)

    x = r * np.cos(theta) + s_root

    dx_dth = -r * np.sin(theta) - (r**2 * np.sin(theta) * np.cos(theta)) / s_root

    term_a = -r * np.cos(theta)
    term_b = -(r**2 * (np.cos(theta)**2 - np.sin(theta)**2)) / s_root
    term_c = -((r**2 * np.sin(theta) * np.cos(theta))**2) / s_root**3
    d2x_dth2 = term_a + term_b + term_c

    return x, dx_dth * omega, d2x_dth2 * omega**2


def kinematics_approx(theta, r, l, omega):
    """Textbook approximation: expand sqrt() and keep the first two harmonics.

    x ~ l - r^2/(4l) + r cos(th) + r^2/(4l) cos(2 th)
    a ~ -r w^2 [ cos(th) + (r/l) cos(2 th) ]
    """
    x = l - r**2 / (4 * l) + r * np.cos(theta) + r**2 / (4 * l) * np.cos(2 * theta)
    v = -r * omega * (np.sin(theta) + (r / (2 * l)) * np.sin(2 * theta))
    a = -r * omega**2 * (np.cos(theta) + (r / l) * np.cos(2 * theta))
    return x, v, a


# -------------------------------------------------------------------- engine
@dataclass(frozen=True)
class Engine:
    """Single-cylinder 4-stroke engine, SI units."""
    bore: float = 0.080            # m
    r: float = 0.050               # crank radius, m  (stroke = 2r = 100 mm)
    l: float = 0.150               # connecting-rod length, m
    m_rec: float = 0.5             # reciprocating mass, kg
    comp_ratio: float = 8.0        # compression ratio
    n: float = 1.3                 # polytropic index (compression & expansion)
    p_intake: float = 0.90e5       # Pa, cylinder pressure during intake
    p_exhaust: float = 1.05e5      # Pa, cylinder pressure during exhaust
    p_atm: float = 1.00e5          # Pa, acts on the underside of the piston
    pressure_ratio: float = 3.0    # p(after combustion) / p(end of compression)

    @property
    def area(self):
        return np.pi * self.bore**2 / 4

    @property
    def swept(self):
        return self.area * 2 * self.r

    @property
    def clearance(self):
        return self.swept / (self.comp_ratio - 1)


def cylinder_volume(theta, eng):
    """V = Vc + A * (distance of the piston below TDC)."""
    x, _, _ = kinematics(theta, eng.r, eng.l, 1.0)
    return eng.clearance + eng.area * ((eng.r + eng.l) - x)


def gas_pressure(theta, eng):
    """Idealised cylinder pressure over a 720 deg cycle.

    intake      : constant p_intake
    compression : polytropic, p V^n = const, starting at BDC
    combustion  : instantaneous (constant-volume) pressure jump at TDC
    expansion   : polytropic from the peak pressure
    exhaust     : constant p_exhaust
    """
    th = np.mod(np.asarray(theta, dtype=float), 4 * np.pi)
    stroke = (th // np.pi).astype(int)
    V = cylinder_volume(th, eng)
    V_bdc = eng.clearance + eng.swept
    p_comp_end = eng.p_intake * eng.comp_ratio**eng.n
    p_peak = eng.pressure_ratio * p_comp_end

    p = np.where(stroke == 0, eng.p_intake, 0.0)
    p = np.where(stroke == 1, eng.p_intake * (V_bdc / V)**eng.n, p)
    p = np.where(stroke == 2, p_peak * (eng.clearance / V)**eng.n, p)
    p = np.where(stroke == 3, eng.p_exhaust, p)
    return p


# --------------------------------------------------------------------- loads
def loads(theta, eng, rpm):
    """Forces and crank torque over the cycle.

    F_gas = (p - p_atm) * A       gas force on the piston
    F_rec = m * a                 reciprocating-inertia term (a is the signed
                                  x-acceleration, so it is negative near TDC:
                                  the piston is being pulled away from the crank)
    F_net = F_gas + F_rec         force transmitted along the cylinder axis
    R     = F_net / cos(phi)      force in the connecting rod
    N     = F_net * tan(phi)      side thrust of the piston on the cylinder wall
    T     = F_net * r * sin(theta + phi) / cos(phi)    crank torque
    """
    theta = np.asarray(theta, dtype=float)
    omega = rpm * 2 * np.pi / 60.0
    x, v, a = kinematics(theta, eng.r, eng.l, omega)
    phi = rod_angle(theta, eng.r, eng.l)
    p = gas_pressure(theta, eng)

    F_gas = (p - eng.p_atm) * eng.area
    F_rec = eng.m_rec * a
    F_net = F_gas + F_rec

    return dict(
        theta=theta, omega=omega, x=x, v=v, a=a, phi=phi, p=p,
        V=cylinder_volume(theta, eng),
        F_gas=F_gas, F_rec=F_rec, F_net=F_net,
        R=F_net / np.cos(phi),
        N=F_net * np.tan(phi),
        T=F_net * eng.r * np.sin(theta + phi) / np.cos(phi),
    )


# ------------------------------------------------------------ energy checks
def flywheel_inertia(theta, T, omega, Cs):
    """Size a flywheel from the torque curve (equally spaced theta over one cycle).

    dE = max - min of the running integral of (T - T_mean)
    I  = dE / (Cs * omega^2)           Cs = coefficient of speed fluctuation
    """
    dth = theta[1] - theta[0]
    T_mean = T.mean()
    running = np.cumsum(T - T_mean) * dth
    dE = running.max() - running.min()
    return T_mean, dE, dE / (Cs * omega**2)


def indicated_work(V, p, p_atm):
    """Work per cycle = integral of (p - p_atm) dV around the closed loop."""
    dV = np.diff(np.append(V, V[0]))
    p_mid = 0.5 * (p + np.append(p[1:], p[0]))
    return np.sum((p_mid - p_atm) * dV)
