using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
# using LsqFit
using Polynomials

# Define the function to compute residuals
function compute_residuals(fitted_curve, x_values, y_values)
    fitted_values = [fitted_curve(x) for x in x_values]
    residuals = y_values - fitted_values
    return residuals
end


am_tilde_0 = 0.3
Delta_g = -0.15
v = 0.0
bounds_k = pi/2
trunc = 4.0


name = "SanderDM_Thesis_2324/Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors_newv"

@load name gs_energy energies Bs anti_energies anti_Bs

energies .+= Delta_g
anti_energies .-= Delta_g

k_values = LinRange(-bounds_k, bounds_k,length(energies))

# energies = [sqrt(am_tilde_0^2 + sin(k/2)^2) for k in 2*k_values]

order = 20

fitted_curve = fit(2*k_values, real.(energies)[:], order)

# Compute the derivative of the fitted curve
fitted_derivative = derivative(fitted_curve, 1)

# Define the range of k-values for which you want to compute the derivative
k_values_for_derivative = LinRange(-bounds_k, bounds_k, 1000)  # Adjust the number of points as needed

# Compute the values of the derivative for the specified range
derivative_values = [fitted_derivative(k) for k in k_values_for_derivative]

# Calculate the standard deviation of the derivative values
std_dev_derivative = std(derivative_values)

println("Standard deviation of derivative: $std_dev_derivative")

deriv = derivative(fitted, 1)

println("Velocity at k = $(k) is $(deriv(k))")


plt = scatter(2*k_values, real.(energies)[:], markerstrokewidth=0, label="Data")
plot!(fitted, extrema(2*k_values)..., label="Interpolation")
display(plt)

k_values_finer = LinRange(-bounds_k, bounds_k, 1000)
max_value, index = findmax([deriv(k) for k in k_values_finer])

using Optim


# Initial guess for the parameter
x0 = [0.0]  # Initial guess as an array

# Define the function to optimize
function objective_function(params)
    return deriv(params[1])
end

# Optimize the parameter to minimize f(x)
result = optimize(objective_function, x0, LBFGS())

# Extract the optimized parameter
optimized_parameter = result.minimizer

# Minimal value of f(x)
minimal_value = result.minimum

println("Optimized parameter: ", optimized_parameter)
println("Minimal value of f(x): ", 2*minimal_value)


using CSV
CSV.write("SanderDM_Thesis_2324/Derivative_info_Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors_newv.csv", [(real.(energies)[:])], header=false)

using DelimitedFiles
open("SanderDM_Thesis_2324/Derivative_info_Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors_newv.csv", "w") do io
    writedlm(io, (real.(energies)[:]), ',')
end
