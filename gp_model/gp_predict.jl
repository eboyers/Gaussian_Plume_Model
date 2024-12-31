using NeuralPDE, Lux, DomainSets, Optimization, OptimizationOptimisers
using Distributions, DataFrames, CSV, Plots, Random, ModelingToolkit

filtered_df = CSV.read("/Users/ethanboyers/Desktop/gaussian_plume_dispersion_model/gp_model/santa_clara_ozone.csv", DataFrame)

# define Gaussian plume model
function gaussian_plume(x, y, Q, H, u, sig_y, sig_z)
    exp1 = exp(-0.5 * ((y^2) / sig_y^2))
    exp2 = exp(-0.5 * (((x - H)^2) / sig_z^2))
    return (Q / (2 * π * sig_y * sig_z * u)) * exp1 * exp2
end

# define params and vars
@parameters x y # no 'z' values in data, assume constant altitude
@variables c(..) # concentration function
Dxx = Differential(x)^2
Dyy = Differential(y)^2

#############################
### SITUATIONAL VARIABLES ###
#############################
mutable struct model_params # struct for model params
    Q::Float64      # source emission rate, g/s
    u::Float64      # horizontal wind velocity along plume centerline, m/s
    H::Float64      # height of plume centerline above ground level, m
    sig_y::Float64  # vertical standard deviation of emission distribution, m
    sig_z::Float64  # horizontal standard deviation of emission distribution, m
end

params = model_params(10.0, 5.0, 10.0, 5.0, 10.0) # init params (CHANGE FOR REAL)
println("Current Parameters: \n- Q = $(params.Q)\n- u = $(params.u)\n- H = $(params.H)\n- sig_y = $(params.sig_y)\n- sig_z = $(params.sig_z)\n")

# update params
function update_params!(params, new_Q, new_u, new_H, new_sig_y, new_sig_z)
    params.Q = new_Q
    params.u = new_u
    params.H = new_H
    params.sig_y = new_sig_y
    params.sig_z = new_sig_z
end

# function def
gauss_eq = c(x, y) ~ gaussian_plume(x, y, params.Q, params.H, params.u, params.sig_y, params.sig_z)
println("Solver: $gauss_eq\n")

# set up boundary and space domains
bcs = [c(x, 0) ~ 0.0, c(x, maximum(filtered_df.latitude)) ~ 0.0, 
       c(minimum(filtered_df.longitude), y) ~ 0.0, c(maximum(filtered_df.longitude), y) ~ 0.0]

domains = [x ∈ Interval(minimum(filtered_df.longitude), maximum(filtered_df.longitude)),
           y ∈ Interval(minimum(filtered_df.latitude), maximum(filtered_df.latitude))]

println("Boundary Conditions: $bcs\n")
println("Domain: $domains\n")

# build PINN
dim = 2 # assume constant altitude
chain = Lux.Chain(Dense(dim, 50, Lux.relu), # relu
                  Dense(50, 50, Lux.relu), 
                  Dense(50, 1))
println("Model Architecture: $chain\n")

discretization = PhysicsInformedNN(chain, QuadratureTraining())
println("Full PINN Model: $discretization\n")

# define PDE, discretize for solving
@named pde_system = PDESystem(gauss_eq, bcs, domains, [x, y], [c])
prob = discretize(pde_system, discretization)

# optimize with ADAM, solve, save optimized parameters
opt = OptimizationOptimisers.Adam(0.01)
res = Optimization.solve(prob, opt, maxiters=100)
trained_params = res.minimizer

function predict_model(params, x, y)
    phi = discretization.phi  # phi is PINN
    result = phi([x, y], params)  # evaluate PINN at coordinates with trained parameters
    return first(result)
end

#####################
### VISUALIZATION ###
#####################
lon_range = range(minimum(filtered_df.longitude), maximum(filtered_df.longitude), length=100)
lat_range = range(minimum(filtered_df.latitude), maximum(filtered_df.latitude), length=100)
grid_points = [(x = lon, y = lat) for lon in lon_range, lat in lat_range]

# Predict concentrations at each point on the grid
concentration_predictions = [predict_model(trained_params, point.x, point.y) for point in grid_points]
concentration_matrix = reshape(concentration_predictions, (length(lon_range), length(lat_range)))

heat_map = heatmap(lon_range, lat_range, concentration_matrix,
        title="Concentration Heatmap",
        xlabel="Longitude",
        ylabel="Latitude",
        color=:viridis)

savefig(heat_map, "gp_model/concentration_santa_clara_ca.png")
