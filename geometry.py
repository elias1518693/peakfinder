import sympy as sp

# Define the variable
t = sp.symbols('t')

# Define the curve function
gamma_t = (sp.sin(3*t)*sp.cos(t), sp.sin(3*t)*sp.sin(t), 0)

# Calculate the derivative
gamma_prime_t = sp.diff(gamma_t, t)

# Substitute t = pi/3 to get the direction vector at the point gamma(pi/3)
direction_vector = gamma_prime_t.subs(t, sp.pi/3)

# Check if the curve is regular for all t
regularity_check = sp.sqrt(sum(comp**2 for comp in direction_vector)).simplify() != 0

# Calculate the equation of the tangent line in the xy-plane at t = pi/3
tangent_line = tuple(-3/2 * t for t in direction_vector[:-1])

print("Is the curve regular for all t?", regularity_check)
print("Equation of the tangent line in the xy-plane:", tangent_line)
