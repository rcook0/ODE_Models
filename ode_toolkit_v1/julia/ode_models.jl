module ODEModels
using DifferentialEquations
function linear_system(A)
    function f!(du,u,p,t); du[:] = A*u; end; return f!; end
function mass_spring_damper(; m=1.0,c=0.2,k=1.0,forcing=t->0.0)
    function f!(du,u,p,t); x,v=u; du[1]=v; du[2]=(forcing(t)-c*v-k*x)/m; end; return f!; end
function rlc_series(; R=1.0,L=1.0,C=1.0,E=t->0.0)
    function f!(du,u,p,t); q,i=u; du[1]=i; du[2]=(E(t)-R*i-q/C)/L; end; return f!; end
function pendulum_simple(; g=9.81,L=1.0,damping=0.0,drive_A=0.0,drive_omega=1.0)
    function f!(du,u,p,t); θ,ω=u; du[1]=ω; du[2]=-(g/L)*sin(θ)-damping*ω+drive_A*sin(drive_omega*t); end; return f!; end
function duffing(; δ=0.2,α=-1.0,β=1.0,γ=0.3,ω=1.2)
    function f!(du,u,p,t); x,v=u; du[1]=v; du[2]=-δ*v-α*x-β*x^3+γ*cos(ω*t); end; return f!; end
function vanderpol(; μ=3.0)
    function f!(du,u,p,t); x,v=u; du[1]=v; du[2]=μ*(1-x^2)*v-x; end; return f!; end
function lorenz63(; σ=10.0,ρ=28.0,β=8.0/3.0)
    function f!(du,u,p,t); x,y,z=u; du[1]=σ*(y-x); du[2]=x*(ρ-z)-y; du[3]=x*y-β*z; end; return f!; end
function lotka_volterra_pred_prey(; α=1.1,β=0.4,δ=0.1,γ=0.4)
    function f!(du,u,p,t); x,p=u; du[1]=α*x-β*x*p; du[2]=δ*x*p-γ*p; end; return f!; end
function competitive_lv(; r1=1.0,r2=0.8,K1=1.0,K2=1.0,a12=0.5,a21=0.6)
    function f!(du,u,p,t); x,y=u; du[1]=r1*x*(1-(x+a12*y)/K1); du[2]=r2*y*(1-(y+a21*x)/K2); end; return f!; end
function oregonator(; s=77.27,q=8.375e-6,f=0.161)
    function f!(du,u,p,t); x,y,z=u; du[1]=s*(y + x*(1-q-x) - f*x); du[2]=(-y - x*(1-q-x))/s; du[3]=f*(x-z); end; return f!; end
function integrate(f!, y0, tspan; alg=:auto, reltol=1e-6, abstol=1e-9, saveat=nothing)
    prob = ODEProblem(f!, y0, tspan)
    if alg == :auto
        sol = solve(prob, AutoTsit5(Rosenbrock23()), reltol=reltol, abstol=abstol, saveat=saveat)
    else
        sol = solve(prob, alg, reltol=reltol, abstol=abstol, saveat=saveat)
    end
    return sol.t, reduce(hcat, sol.u)'
end
end