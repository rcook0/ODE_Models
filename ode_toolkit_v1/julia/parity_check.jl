using Pkg; Pkg.add("DifferentialEquations")
include("julia/ode_models.jl"); using .ODEModels
t,y = ODEModels.integrate(ODEModels.lorenz63(), [1.0,1.0,1.0], (0.0,25.0), saveat=0.01)
size(y)
