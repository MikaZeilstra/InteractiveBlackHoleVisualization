import sympy
from sympy.parsing.latex import parse_latex
import sympy as sp

#setup Symbols
c,hbar,G,sigma, pi_symbol, rg, R, Ma,M,F, x = sp.symbols("c hbar G sigma pi r_{g} R M_{a} M F x")

#Define constant values
c_val = 299792458
hbar_val = 1.054571817e-34
G_val = 6.6743e-11
sb_val = 5.670374419e-8
rg_val = 2

solar_mass = 2e30
solar_masses_per_year = solar_mass/(365*24*3600)

#Equations for schwarzschild radius and Temperature
schwarzR = parse_latex(r"2GM/c^2")
Temp = parse_latex(r"(F/\sigma)^{1/4}")


#R part equation
test = parse_latex(r"\frac{1}{(R-3/2)R^{5/2}}[R^{1/2}-3^{1/2}+\frac{(3/2)^{1/2}}{2}\ln\frac{R^{1/2} +(3/2)^{1/2}}{R^{1/2} -(3/2)^{1/2}}\frac{3^{1/2}-(3/2)^{1/2}}{{3^{1/2}+(3/2)^{1/2}}}]]")
#Other part of equation
test_mult = parse_latex(r"\frac{3GMM_a}{8\pi r_g^3 }")

#Sub in value for max temp
#test = test.subs([(R,4.8)])
test = test_mult * test

#Sub in schwarschild radius and constants
test = test.subs([(rg,schwarzR),(c, c_val),(G,G_val),(pi_symbol,sympy.pi)])

#Sub Flux into temperature equation
test = Temp.subs(F,test)

#Sub boltzman
test = test.subs(sigma,sb_val)

#Sub differing units
test = test.subs([(Ma,solar_masses_per_year*Ma),(M, solar_mass * M)])
print(sp.latex(test.simplify().evalf(5)))


test = test.subs([(Ma,0.1),(M,3.5e9)])

