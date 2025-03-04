# Expt 2

using DifferentialEquations,SciMLSensitivity,Zygote,LinearAlgebra,Plots
using SparseArrays, Random, Statistics, Optimisers, Flux, JLD2

#------------------------------------------#
#        Simulation input parameters       #
#------------------------------------------#

begin 
    global N = 256                      # number of cells 
    global L = 5π                       # total length of 1d sim
    global dt = 0.01                     # time step
    global T = 1.0                       # total time 
    global nt = T / dt                   # number of timesteps
    global dx = L/N                      # spatial step
    global x = 0.0 : dx : (L-dx)         # discretized spatial dimension 
    global xgrid = collect(x)            # grid 
    global tsteps = 0.0:dt:T             # discretized time dimension 
    global tspan = (0,T)                 # end points of time integration for ODEProbem

    global ω0 = 0.5                      # Coriolis factor from geyer.quirchmayr_2018 paper
    global c  = 2.5                      # Soliton velocity 
    global A  = 5.0                      # Amplitude of wave in IC
    global xS = 1.5                      # Soliton shift from origin 
    global γ  = 1                        # Less-resolved-timesteps-for-training factor ; dt_train = γ dt_true 
    global ckpt_save_freq = 20
    global epochs = 10000
    global lr  = 0.01         
end 

include("../../src/numerical_derivatives.jl");

checkpoint_fpath = "./ckpt_expt2.jld2"

resume_checkpoint = false

# derivative operators for train data
∂x1 = f1_secondOrder_central(N,dx)
∂x3 = f3_secondOrder_central(N,dx)

#ROM numerical scheme 
∂x3_rom = f3_sixthOrder_central(N,dx)

function gKdV_ic_sinusoid(x,A,c,ω0)
    return -A*sin.(x./c .+ pi)
end
u0 = gKdV_ic_sinusoid(x,A,c,ω0) ;

# true gKdV pde
function true_gKdV(u,p,t)
    return ω0 .* (∂x1 * u) - 3.0/2.0 .* u .* (∂x1 *u) - 1.0 ./ 6.0 .* (∂x3 * u) .* p
end

# learnable parameter p , not used in training data generation, initialized with ones
# to make adjoint computation feasible
# save only initial condition and last timestep
pones = ones(N) ;
true_gKdV_prob = ODEProblem(true_gKdV, u0, tspan,pones)
loss_true(u0, p) = sum(solve(true_advdiff_prob, Tsit5(), u0 = u0, p = p, saveat = dt));

## Create training data 
sol_true_gKdV = Array(solve(true_gKdV_prob,Tsit5(),alg_hints=[:stiff], dt=dt, saveat=dt, reltol = Float32(1e-6)));

model = Chain(
            Dense(N,20,tanh),
            Dense(20,20,tanh),
            Dense(20,20,tanh),
            Dense(20,N));

if resume_checkpoint == true
    f = jldopen(checkpoint_fpath,"r")
    p = f["p"];
    optim = f["state"];
    close(f)
    println("loaded checkpoint to RESUME training...")
    _, re = Flux.destructure(model);
    model = re(p); #load trained weights from checkpoint into model state.
elseif resume_checkpoint == false  
    println("Starting training from SCRATCH!!!")
    p, re = Flux.destructure(model); #extract random init weights from model for training.
    optim = Flux.setup(Flux.Adam(lr), p);
else
    println("UNKNOWN training state, assuming training from scratch")
    p, re = Flux.destructure(model); #extract random init weights from model for training.
    optim = Flux.setup(Flux.Adam(lr), p);
end

function learned_gKdV(u,p,t)
    ϕ = re(p)
    u = ϕ(u) - 1.0 ./ 6.0 .* (∂x3_rom * u)
    return u
end 

learned_gKdV_prob = ODEProblem(learned_gKdV, u0, tspan, p)
loss_learned(u0, p) = sum(solve(learned_gKdV_prob, Tsit5(), u0 = u0, p = p, saveat = dt, reltol = Float32(1e-6)));
# data loader definitions
batchsize=1 # only one initial condition is used for training.
# minimize the full trajectory or just last step? A choice to make.
traindata = (u0=Float32.(u0),y=Float32.(sol_true_gKdV[:,end]));


# Training loop over multiple epochs
losses = []
for epoch in 1:epochs
    @show epoch
    # Step 1: Compute gradient for backprop using truae solution and true adjoint
    loss, ∇p = Flux.withgradient(p) do p
        ŷ  = solve(learned_gKdV_prob, Tsit5(), u0 = u0, p = p, saveat = dt, reltol = Float32(1e-6));
        loss = Flux.mse(ŷ[:,end], traindata.y) # minimize the full trajectory or just last step? A choice to make.
        @show loss
        return loss
    end

    # Step 2: Optimizer update
    Flux.update!(optim, p, ∇p[1])
    
    if epoch%ckpt_save_freq == 0
        jldsave(checkpoint_fpath,p=p,gradp=∇p,state=optim)
        println("saved checkpoint")
    elseif epoch == epochs # to save final trained state.
        jldsave(checkpoint_fpath,p=p,gradp=∇p,state=optim)
        println("saved FINAL checkpoint")
    end
    
    push!(losses,loss)
end
plot(losses)
savefig("losses_expt2.png")

sol_learn_gKdV_trained  = Array(solve(learned_gKdV_prob, Tsit5(), u0 = u0, p = p, saveat = dt, reltol = Float32(1e-6)));
plot(sol_learn_gKdV_trained[:,end], label="predicted")
plot!(sol_true_gKdV[:,end], label="truth")
plot!(sol_true_gKdV[:,1], label="I.C.")
savefig("final_result_expt2.png")

err_field = sol_learn_gKdV_trained - sol_true_gKdV
println(sum(err_field))
contourf(err_field, xlabel="t", ylabel="x")
savefig("pred_error_expt2.png")
