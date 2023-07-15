### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ b8591bca-2318-11ee-17af-a7bc61c5e4d4
begin
	import Pkg; Pkg.activate()
	using CairoMakie, Turing, DataFrames, PlutoUI, CSV, Statistics, StatsBase

	update_theme!(fontsize=18, resolution=(0.9*500, 0.9*380))
end

# ╔═╡ 7e0a601d-9081-4d45-983a-7c39c8be28af
TableOfContents()

# ╔═╡ b110d3e4-fc4a-48c0-a8e9-50f0dba3837c
md"
# problem setup
!!! note
	this is a minimal/simplified tutorial of Bayesian statistical inversion (BSI) for inverse problems, following the problem setup and using the data in:
	> F. Waqar, S. Patel, C. Simon. \"A tutorial on the Bayesian statistical approach to inverse problems\" _APL Machine Learning_. (2023) [link](https://arxiv.org/abs/2304.07610)

	this tutorial is in the [Julia programming language](https://julialang.org/), within a [Pluto notebook](https://plutojl.org/), and largely relies on the probabalistic programming package [`Turing.jl`](https://turing.ml/).

## setup of lime heat transfer experiment
a cold lime fruit at temperature $\theta_0$ [°C] rests inside of a refrigerator. at time $t:=0$ [hr], we take the lime outside of the refrigerator and allow it exchange heat with the indoor air, which is at temperature $\theta^{\text{air}}$ [°C]. a temperature probe inserted into the lime allows us to measure the temperature of the lime, $\theta=\theta(t)$ [°C].
"

# ╔═╡ c6b2fd93-a19d-4198-9499-52b4d3484ef6
html"<img src=\"https://raw.githubusercontent.com/faaiqgwaqar/Inverse-Problems/main/tutorial/lime_setup.jpeg\" width=400>"

# ╔═╡ 8649c024-57e7-4918-a97c-b04cd3a0ce36
md"## forward model of the lime temperature
we treat the temperature of the lime as spatially uniform. our mathematical model for the temperature of the lime as a function of time $t$ [hr] is:
```math
\begin{equation}
    \theta (t)=\theta^{\text{air}}+(\theta_0-\theta^{\text{air}})e^{-t/\lambda}, \quad \text{for } t\geq 0. 
\end{equation} 
```
the (unknown) parameter $\lambda$ [hr] characterizes the dynamics of heat exchange between the air and the lime.
"

# ╔═╡ 0d7ba7e0-088a-4017-9561-2834856143e6
# model for lime temperature
function θ(t, λ, θ₀, θᵃⁱʳ)
    if t < 0.0
        return θ₀
    end
    return θᵃⁱʳ + (θ₀ - θᵃⁱʳ) * exp(-t / λ)
end

# ╔═╡ bc95e270-27f6-4812-8a83-0d9d0262e474
md"## probabilistic model of the temperature measurements
we use the forward model to construct a probabalistic model of the measured lime temperature. we assume any observed measurement $\theta_{\text{obs}}$ [°C] of the lime temperature at time $t \geq 0$ is a realization of a random variable $\Theta_{\text{obs}}$ with a Gaussian distribution
```math
\begin{equation}
    \Theta_{\text{obs}} \mid \lambda, \theta_0, \theta^{\text{air}}, \sigma \sim \mathcal{N}(\theta(t; \lambda, \theta_0, \theta^{\text{air}}), \sigma^2)
\end{equation}
```
with a mean governed by the model $\theta(t)$ and variance $\sigma^2$ owing to measurement noise and zero-mean residual variability. we treat multiple measurements as independent and identically distributed. this distribution of $\Theta_{\text{obs}}$ is conditioned on knowing the values of $\lambda, \theta_0, \theta^{\text{air}}, \sigma$. it will be used to construct the likelihood function.
"

# ╔═╡ 9f72d5fb-c614-4f21-bdb9-2fbc9ebe2361
md"# parameter identification

🔨 **task**: infer the parameter $\Lambda$ (capitalized $\lambda$, because we treat it as a random variable) in the model of the lime temperature.

**sub-tasks**: infer the:
* variance of the measurement noise, $\Sigma^2$.
* initial lime temperature $\Theta_0$, even though we will take a (noisy) measurement of it.
* air temperature $\Theta^{\text{air}}$, even though we will take a (noisy) measurement of it.

## setting up the heat transfer experiment

first, we setup the lime heat transfer experiment and make two measurements to characterize the experimental conditions.

🌡 we use the temperature probe to measure:
* the initial temperature of the lime $\theta_0$ at time $t=0$, giving $\theta_{0, \text{obs}}$.
* the air temperature $\theta^{\text{air}}$, giving $\theta^{\text{air}}_\text{obs}$.
"

# ╔═╡ 2908489a-e2f0-4470-850c-c46565b2ea0e
θ₀_obs = 6.54 # °C

# ╔═╡ b1f1da4d-fde8-4e08-aab4-a9942558fe4d
θᵃⁱʳ_obs = 18.47 # °C

# ╔═╡ 21822a96-a519-4585-92b1-c23d838fb5a6
md"## the prior distributions

next, we construct prior distributions to reflect the information and beliefs we have about the unknowns ($\lambda$, $\theta_0$, $\theta^{\text{air}}$, $\sigma$) _before_ we collect and consider time series data over the course of the lime heat transfer experiment.

**the unknown model parameter**. based on a back-of-the-envelope estimate of $\lambda$ and our confidence in this estimate, our prior for $\Lambda$ is a spread-out Gaussian distribution centered at our estimate for it, of 1 hr:
```math
\begin{equation}
    \Lambda \sim \mathcal{N}_{> 0} \left(1 \text{ hr}, (0.3\text{ hr})^2 \right).
\end{equation}
```

**variance of measurement noise**. based on the precision of the temperature probe,
```math
\begin{equation}
    \Sigma \sim \mathcal{U}([0\,^\circ \text{C}, 1\,^\circ \text{C}]),
\end{equation}
```
where $\mathcal{U}(\cdot)$ is a uniform distribution over the set $\cdot$.

**experimental conditions**. we construct informative distributions for the initial lime temperature and air temperature, since we have measured these:
```math
\begin{align}
    \Theta_0 & \sim \mathcal{N}(\theta_{0, \text{obs}}, \sigma^2) \\ 
    \Theta^{\text{air}} & \sim \mathcal{N}(\theta_{\text{obs}}^{\text{air}}, \sigma^2).
\end{align}
```
where $\sigma$ is unknown and the values $\theta_{0, \text{obs}}$ and $\theta_{\text{obs}}^{\text{air}}$ are in the cells above.
"

# ╔═╡ d44fc256-55af-490d-a643-21fb07950f7c
md"## the data

🌡 we measure the temperature of the lime at different times as it exchanges heat with the indoor air. this gives a time series data set $\{(t_i, \theta_{i, \text{obs}})\}_{i=1}^n$ we read in the raw data from a `.csv` file.
"

# ╔═╡ 2f593ee3-f6ca-4397-9e09-bd9761720f29
data = CSV.read("lime_temp_param_id.csv", DataFrame)

# ╔═╡ 7db78800-3c44-4b46-9d72-3ef041b21708
n = nrow(data)

# ╔═╡ 9bf8a9dd-a6d4-45e5-bf5f-cc830e6daae9
begin
	local fig = Figure()
	local ax  = Axis(fig[1, 1], xlabel="time, t [hr]", 
		ylabel="lime temperature, θ [°C]")
	scatter!(data[:, "t [hr]"], data[:, "θ_obs [°C]"], color=Cycled(1),
		label=rich("{(t", subscript("i"), ", θ", subscript("i, obs"), ")}"))
	scatter!([0], [θ₀_obs], color=Cycled(1))
	hlines!([θᵃⁱʳ_obs], color=Cycled(3), linestyle=:dash, 
		label=rich("θ", superscript("air"), subscript("obs")))
	axislegend(position=:rb)
	fig
end

# ╔═╡ 40a3e0f9-78b8-4028-a829-c3eb1a0d87dd
md"
💡 this data provides information about the unknown model parameter $\lambda$ _and_ the variance of the noise corrupting our measurements $\sigma$ (_and_, to a lesser-extent than our direct measurements of them, $\theta_0$ and $\theta^{\text{air}}$).

"

# ╔═╡ c0ff232c-5f9e-4612-9ad4-ae5c4c19217f
md"## the posterior distribution

we seek the posterior distribution
```math
\begin{equation}
\Lambda, \Theta_0, \Theta^{\text{air}}, \Sigma \mid \theta_{0, \text{obs}}, \theta_{\text{obs}}^{\text{air}}, \{(t_i, \theta_{i, \text{obs}})\}_{i=1}^n
\end{equation}
```
which follows from (i) our prior distributions and (ii) our probabilistic forward model, which provides the likelihood function.

in the probabilistic programming pardigm implemented `Turing.jl`, we obtain the posterior distribution by first coding the prior and probabilistic forward model. then, `Turing.jl` employs a Markov chain Monte Carlo algoirthm (NUTS) to draw (serially correlated) samples from the posterior distribution. we then approximate the posterior distribution with an empirical distribution constructed from these samples.
"

# ╔═╡ b96d3466-a21a-4383-b8fa-db9e1b69773f
@model function measure_lime_temp_time_series(data)
    # prior distributions
	λ    ~ truncated(Normal(1.0, 0.3), 0.0, nothing) # hr
    σ    ~ Uniform(0.0, 1.0) # °C
    θ₀   ~ Normal(θ₀_obs, σ)
    θᵃⁱʳ ~ Normal(θᵃⁱʳ_obs, σ)

    # probabilistic forward model
    for i = 1:nrow(data)
        tᵢ = data[i, "t [hr]"]
        θ̄ = θ(tᵢ, λ, θ₀, θᵃⁱʳ)
        data[i, "θ_obs [°C]"] ~ Normal(θ̄, σ)
	end
end

# ╔═╡ 09f1fed0-30e2-421e-8807-1264c8f142d0
begin
	mlts_model = measure_lime_temp_time_series(data)
		
	nb_samples = 2_500
	nb_chains = 4
	chain = DataFrame(
		sample(mlts_model, NUTS(), MCMCSerial(), nb_samples, nb_chains)
	)
end

# ╔═╡ 923ca5f2-8730-4387-8555-de2a53a79d3b
md"posterior mean $\Lambda$:"

# ╔═╡ 14a77fc0-798f-4507-93b8-e9f04d664a9b
μ_λ = mean(chain[:, :λ]) # hr

# ╔═╡ b9bd7754-e45b-4e47-a6ef-71045aded2b7
md"90% equal-tailed posterior credible interval for $\Lambda$:"

# ╔═╡ 2cba4785-44f2-4cb2-8e77-b63489609661
ci_λ = [percentile(chain[:, :λ], 5.0), percentile(chain[:, :λ], 95.0)]

# ╔═╡ 2c2b6580-2743-4a94-8285-c1166d63df1e
md"the histogram below serves as an approximation to the posterior distribution of $\Lambda$, with the other variables ($\Theta_0$, $\Theta^{\text{air}}$, and $\Sigma^2$) marginalized out. the vertical line denotes the mean of the posterior of $\Lambda$, and the black bar denotes the 90% equal-tailed credible interval for $\Lambda$."

# ╔═╡ 34651997-a213-4ebd-a8bc-b76b7d14bb69
begin
	local fig = Figure()
	local ax  = Axis(fig[1, 1], xlabel="λ [hr]", ylabel="# samples", 
		title="posterior dist'n of Λ")
	hist!(chain[:, :λ])
	ylims!(0, nothing)
	vlines!([μ_λ], linestyle=:dash, color=Cycled(2))
	lines!(ci_λ, zeros(2), color="black", linewidth=10)
	fig
end

# ╔═╡ dd688e57-d441-4fa3-ba4f-cfed0baf1725
md"samples from posterior models of lime temperature trajectories:"

# ╔═╡ f45b7e47-902d-4c9b-b28b-7a320bf8a16e
begin
	local fig = Figure()
	local ax  = Axis(fig[1, 1], xlabel="time, t [hr]", 
		ylabel="lime temperature, θ [°C]")

	n_trajectories = 25
	t = range(-0.2, 10.0, length=100)
	for row in eachrow(chain)
		lines!(ax, t, θ.(t, row[:λ], row[:θ₀], row[:θᵃⁱʳ]), 
			color=("orange", 0.05))
	end
	scatter!(data[:, "t [hr]"], data[:, "θ_obs [°C]"], color=Cycled(1),
		label=rich("{(t", subscript("i"), ", θ", subscript("i, obs"), ")}"))
	scatter!([0], [θ₀_obs], color=Cycled(1))
	hlines!([θᵃⁱʳ_obs], color=Cycled(3), linestyle=:dash, 
		label=rich("θ", superscript("air"), subscript("obs")))
	axislegend(position=:rb)

	xlims!(-0.2, 10)
	fig
end

# ╔═╡ Cell order:
# ╠═b8591bca-2318-11ee-17af-a7bc61c5e4d4
# ╠═7e0a601d-9081-4d45-983a-7c39c8be28af
# ╟─b110d3e4-fc4a-48c0-a8e9-50f0dba3837c
# ╟─c6b2fd93-a19d-4198-9499-52b4d3484ef6
# ╟─8649c024-57e7-4918-a97c-b04cd3a0ce36
# ╠═0d7ba7e0-088a-4017-9561-2834856143e6
# ╟─bc95e270-27f6-4812-8a83-0d9d0262e474
# ╟─9f72d5fb-c614-4f21-bdb9-2fbc9ebe2361
# ╠═2908489a-e2f0-4470-850c-c46565b2ea0e
# ╠═b1f1da4d-fde8-4e08-aab4-a9942558fe4d
# ╟─21822a96-a519-4585-92b1-c23d838fb5a6
# ╟─d44fc256-55af-490d-a643-21fb07950f7c
# ╠═2f593ee3-f6ca-4397-9e09-bd9761720f29
# ╠═7db78800-3c44-4b46-9d72-3ef041b21708
# ╠═9bf8a9dd-a6d4-45e5-bf5f-cc830e6daae9
# ╟─40a3e0f9-78b8-4028-a829-c3eb1a0d87dd
# ╟─c0ff232c-5f9e-4612-9ad4-ae5c4c19217f
# ╠═b96d3466-a21a-4383-b8fa-db9e1b69773f
# ╠═09f1fed0-30e2-421e-8807-1264c8f142d0
# ╟─923ca5f2-8730-4387-8555-de2a53a79d3b
# ╠═14a77fc0-798f-4507-93b8-e9f04d664a9b
# ╟─b9bd7754-e45b-4e47-a6ef-71045aded2b7
# ╠═2cba4785-44f2-4cb2-8e77-b63489609661
# ╟─2c2b6580-2743-4a94-8285-c1166d63df1e
# ╠═34651997-a213-4ebd-a8bc-b76b7d14bb69
# ╟─dd688e57-d441-4fa3-ba4f-cfed0baf1725
# ╠═f45b7e47-902d-4c9b-b28b-7a320bf8a16e
