
module ODEModels
using DifferentialEquations

function lorenz63(; σ=10.0, ρ=28.0, β=8.0/3.0)
    function f!(du,u,p,t)
        x,y,z = u
        du[1] = σ*(y-x)
        du[2] = x*(ρ-z)-y
        du[3] = x*y-β*z
    end
    return f!
end

function rossler(; a=0.2, b=0.2, c=5.7)
    function f!(du,u,p,t)
        x,y,z = u
        du[1] = -y - z
        du[2] = x + a*y
        du[3] = b + z*(x - c)
    end
    return f!
end

function vanderpol(; μ=3.0)
    function f!(du,u,p,t)
        x,v = u
        du[1] = v
        du[2] = μ*(1-x^2)*v - x
    end
    return f!
end

function duffing(; δ=0.2, α=-1.0, β=1.0, γ=0.3, ω=1.2)
    function f!(du,u,p,t)
        x,v = u
        du[1] = v
        du[2] = -δ*v - α*x - β*x^3 + γ*cos(ω*t)
    end
    return f!
end

function predator_prey(; α=1.0, β=0.1, δ=0.075, γ=1.5)
    function f!(du,u,p,t)
        x,y = u
        du[1] = α*x - β*x*y
        du[2] = δ*x*y - γ*y
    end
    return f!
end

function sir(; β=0.5, γ=0.2)
    function f!(du,u,p,t)
        S,I,R = u
        N = S+I+R
        du[1] = -β*S*I/(N+1e-12)
        du[2] =  β*S*I/(N+1e-12) - γ*I
        du[3] =  γ*I
    end
    return f!
end

const _REG = Dict{String, Function}()
_norm(s) = lowercase(replace(replace(strip(s), " "=>""), "_"=>""))
_register!(name, f) = (_REG[_norm(name)] = f)

_register!("lorenz", (kwargs...; kw...) -> lorenz63(; kw...))
_register!("lorenz63", (kwargs...; kw...) -> lorenz63(; kw...))
_register!("rossler", (kwargs...; kw...) -> rossler(; kw...))
_register!("vanderpol", (kwargs...; kw...) -> vanderpol(; kw...))
_register!("duffing", (kwargs...; kw...) -> duffing(; kw...))
_register!("predatorprey", (kwargs...; kw...) -> predator_prey(; kw...))
_register!("sir", (kwargs...; kw...) -> sir(; kw...))

function list_models()
    return sort(collect(keys(_REG)))
end

function get_model(name::AbstractString; kw...)
    key = _norm(name)
    haskey(_REG, key) || error("Unknown model: $name (normalized: $key)")
    return _REG[key](; kw...)
end

function integrate_model(name::AbstractString, y0, tspan; saveat=nothing, reltol=1e-6, abstol=1e-9, alg=:auto, kw...)
    f! = get_model(name; kw...)
    prob = ODEProblem(f!, y0, tspan)
    if alg == :auto
        sol = solve(prob, AutoTsit5(Rosenbrock23()), reltol=reltol, abstol=abstol, saveat=saveat)
    else
        sol = solve(prob, alg, reltol=reltol, abstol=abstol, saveat=saveat)
    end
    return sol.t, reduce(hcat, sol.u)'
end

end
