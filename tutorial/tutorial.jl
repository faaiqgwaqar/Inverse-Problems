### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ b8591bca-2318-11ee-17af-a7bc61c5e4d4
begin
	import Pkg; Pkg.activate()
	using CairoMakie, Turing, DataFrames, PlutoUI, CSV, Statistics, StatsBase, ColorSchemes

	update_theme!(fontsize=18, resolution=(0.9*500, 0.9*380))
end

# ╔═╡ 7e0a601d-9081-4d45-983a-7c39c8be28af
TableOfContents()

# ╔═╡ b110d3e4-fc4a-48c0-a8e9-50f0dba3837c
md"
# a coding tutorial for Bayesian statistical inversion (BSI)
_contact email for feedback or questions_: $\texttt{cory.simon}$ [at] $\texttt{oregonstate.edu}$

!!! note
	this is a minimal/simplified coding tutorial for Bayesian statistical inversion (BSI), following the problem setup in and using the experimental data from our paper:
	> F. Waqar, S. Patel, C. Simon. \"A tutorial on the Bayesian statistical approach to inverse problems\" _APL Machine Learning_. (in revision) (2023) [link](https://arxiv.org/abs/2304.07610)

	this tutorial uses the [Julia programming language](https://julialang.org/), and this document is a [Pluto notebook](https://plutojl.org/). 

	to obtain an empirical approximation of the posterior distribution through sampling, we rely on the probabalistic programming library, [`Turing.jl`](https://turinglang.org/dev/docs/using-turing/), whose docs contain another relevant tutorial [\"Bayesian Estimation of Differential Equations\"](https://turinglang.org/dev/tutorials/10-bayesian-differential-equations/).

## problem setup

### setup of lime heat transfer experiment
a cold lime fruit at temperature $\theta_0$ [°C] rests inside of a refrigerator. at time $t:=0$ [hr], we take the lime outside of the refrigerator. thereafter, the lime exchanges heat with the indoor air, which is at temperature $\theta^{\text{air}}$ [°C]. we may take a measurement of the temperature of the lime $\theta=\theta(t)$ [°C] at some time $t\geq 0$ via a temperature probe inserted into the lime. see below.
"

# ╔═╡ c6b2fd93-a19d-4198-9499-52b4d3484ef6
html"<img src=\"https://raw.githubusercontent.com/faaiqgwaqar/Inverse-Problems/main/tutorial/lime_setup.jpeg\" width=400>"

# ╔═╡ 8649c024-57e7-4918-a97c-b04cd3a0ce36
md"### forward model of the lime temperature
our mathematical model for the temperature of the lime, approximated as spatially uniform, as a function of time $t$ [hr] is:
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
md"### probabilistic model of the temperature measurements
we use the forward model to construct a probabalistic model of the measured lime temperature. we assume any observed measurement $\theta_{\text{obs}}$ [°C] of the lime temperature at time $t \geq 0$ is a realization of a random variable $\Theta_{\text{obs}}$, a Gaussian distribution
```math
\begin{equation}
    \Theta_{\text{obs}} \mid \lambda, \theta_0, \theta^{\text{air}}, \sigma \sim \mathcal{N}(\theta(t; \lambda, \theta_0, \theta^{\text{air}}), \sigma^2)
\end{equation}
```
with a mean governed by the model $\theta(t)$ and variance $\sigma^2$ (another parameter). the variance in the measurement originates from measurement noise and zero-mean residual variability. we treat multiple measurements as independent and identically distributed. note, this distribution of $\Theta_{\text{obs}}$ is conditioned on knowing the values of $\lambda, \theta_0, \theta^{\text{air}}, \sigma$. after we collect the data, the likelihood function follows from this probabilistic model of the temperature measurements.
"

# ╔═╡ 9f72d5fb-c614-4f21-bdb9-2fbc9ebe2361
md"## parameter inference

🔨 **task**: infer the parameter $\Lambda$ (capitalized $\lambda$, because we treat it as a random variable) in the model of the lime temperature.

**sub-tasks**: infer the (also treated as random variables):
* variance of the measurement noise, $\Sigma^2$.
* initial lime temperature $\Theta_0$, even though we will take a (noisy) measurement of it.
* air temperature $\Theta^{\text{air}}$, even though we will take a (noisy) measurement of it.

### setting up the heat transfer experiment

first, we setup the lime heat transfer experiment and make two measurements to characterize the experimental conditions.

🌡 we use the temperature probe to measure:
* the initial temperature of the lime $\theta_0$ at time $t=0$, giving datum $\theta_{0, \text{obs}}$.
* the air temperature $\theta^{\text{air}}$, giving datum $\theta^{\text{air}}_\text{obs}$.
"

# ╔═╡ 2908489a-e2f0-4470-850c-c46565b2ea0e
θ₀_obs = 6.54 # °C

# ╔═╡ b1f1da4d-fde8-4e08-aab4-a9942558fe4d
θᵃⁱʳ_obs = 18.47 # °C

# ╔═╡ 21822a96-a519-4585-92b1-c23d838fb5a6
md"### the prior distributions

next, we construct prior distributions to reflect the information and beliefs we have about the unknowns ($\lambda$, $\theta_0$, $\theta^{\text{air}}$, $\sigma$) _before_ we collect and consider time series data over the course of the lime heat transfer experiment.

**the unknown model parameter**. based on a back-of-the-envelope estimate of $\lambda$ and our confidence in this estimate, our prior for $\Lambda$ is a spread-out, truncated-below-zero Gaussian distribution centered at our estimate of it ($\lambda\approx 1$ hr; see our paper):
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

**experimental conditions**. we construct informative prior distributions for the initial lime temperature and air temperature, since we have measured these:
```math
\begin{align}
    \Theta_0 & \sim \mathcal{N}(\theta_{0, \text{obs}}, \sigma^2) \\ 
    \Theta^{\text{air}} & \sim \mathcal{N}(\theta_{\text{obs}}^{\text{air}}, \sigma^2).
\end{align}
```
where $\sigma$ is unknown and the values $\theta_{0, \text{obs}}$ and $\theta_{\text{obs}}^{\text{air}}$ are in the cells above.
"

# ╔═╡ d44fc256-55af-490d-a643-21fb07950f7c
md"### the time series data

🌡 we measure the temperature of the lime at different times as it exchanges heat with the indoor air. this gives a time series data set $\{(t_i, \theta_{i, \text{obs}})\}_{i=1}^{10}$. we read in the raw data from a `.csv` file available [here](https://raw.githubusercontent.com/faaiqgwaqar/Inverse-Problems/main/tutorial/lime_temp_param_id.csv).
"

# ╔═╡ 2f593ee3-f6ca-4397-9e09-bd9761720f29
data = CSV.read("lime_temp_param_id.csv", DataFrame)

# ╔═╡ 9e86ddd1-0663-4cad-9e42-bf100900d96f
md"let's plot the time series data along with the measured initial lime temperature and air temperature."

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

👀 we can visually inspect the time series to roughly estimate $\lambda$, since it represents a time scale for the lime to thermally equilibrate with the air. specifically, at time $t=\lambda$, 
```math
\begin{equation}
\theta(\lambda)= \theta^\text{air} + e^{-1}(\theta_0- \theta^\text{air}) \implies \theta^\text{air}-\theta(\lambda) \approx 0.37(\theta^\text{air} - \theta_0)
\end{equation}
```
meaning the difference between the air and lime temperature at $t=\lambda$ is $\sim$37% of the initial difference. from \"eye-balling\" the plot, the time series data suggests $\lambda \approx 1$ hr, in agreement with our back-of-the-envelope calculation.
"

# ╔═╡ c0ff232c-5f9e-4612-9ad4-ae5c4c19217f
md"### the posterior distribution

we seek the posterior distribution of the values of the unknowns in the inverse problem, in light of the time series data:
```math
\begin{equation}
\Lambda, \Theta_0, \Theta^{\text{air}}, \Sigma \mid \{(t_i, \theta_{i, \text{obs}})\}_{i=1}^{10}.
\end{equation}
```
from Bayes's theorem, the posterior distribution follows from (i) our prior distributions and (ii) our probabilistic forward model (more precisely, from the likelihood function constructed from the forward model, which we do not explicitly code-up here).

in the probabilistic programming paradigm implemented in `Turing.jl`, we obtain the posterior distribution by:
1. coding the
    * prior distributions.
    * probabilistic forward model governing how each data point $(t_i, \theta_{i, \text{obs}})$ is generated.
2. calling the Markov chain Monte Carlo (NUTS) sampler to draw (serially correlated) samples from the posterior distribution. 
3. approximating the posterior distribution with an empirical distribution constructed from these samples.

!!! note
	see the docs of [`Turing.jl`](https://turinglang.org/dev/docs/using-turing/) for more details about its probabilistic programming approach to Bayesian inference.

step 1:
"

# ╔═╡ b96d3466-a21a-4383-b8fa-db9e1b69773f
# implementation in Turing.jl, for specifiying prior and forward model
@model function measure_lime_temp_time_series(data)
    # prior distributions
	λ    ~ truncated(Normal(1.0, 0.3), 0.0, nothing) # hr
    σ    ~ Uniform(0.0, 1.0) # °C
    θ₀   ~ Normal(θ₀_obs, σ) # °C
    θᵃⁱʳ ~ Normal(θᵃⁱʳ_obs, σ) # °C

    # probabilistic forward model
    for i = 1:nrow(data)
		# the time stamp
        tᵢ = data[i, "t [hr]"]
		# the model prediction
        θ̄ = θ(tᵢ, λ, θ₀, θᵃⁱʳ)
		# the probabilistic forward model
        data[i, "θ_obs [°C]"] ~ Normal(θ̄, σ)
	end
end

# ╔═╡ 6d83d92f-7010-4973-bcbf-5f26479f5ea2
md"step 2: (each row of `chain` may be treated as a sample from the posterior distribution)"

# ╔═╡ 09f1fed0-30e2-421e-8807-1264c8f142d0
begin
	mlts_model = measure_lime_temp_time_series(data)
		
	nb_samples = 2_500 # per chain
	nb_chains = 4      # independent chains
	chain = DataFrame(
		sample(mlts_model, NUTS(), MCMCSerial(), nb_samples, nb_chains)
	)
end

# ╔═╡ cc2ae7a6-99db-4c22-ac72-530631a3cdef
md"we compare the dist'n of $\Lambda$ over the `nb_chains=4` independent chains (a convergence diagnostic---they should approximately match)."

# ╔═╡ db781bcd-0950-4b7b-969f-08f1cc4a2449
begin
	local fig = Figure()
	local ax = Axis(fig[1, 1], xlabel="λ [hr]", ylabel="# samples")
	for (i, c) in enumerate(groupby(chain, "chain"))
		hist!(c[:, "λ"], color=(ColorSchemes.Accent_4[i], 0.5))
	end
	fig
end

# ╔═╡ 2c2b6580-2743-4a94-8285-c1166d63df1e
md"step 3: 
the histogram below serves as an approximation to the posterior distribution of $\Lambda$, with the other variables ($\Theta_0$, $\Theta^{\text{air}}$, and $\Sigma^2$) marginalized out. the vertical line denotes the mean of the posterior of $\Lambda$, and the black bar denotes the 90% equal-tailed credible interval for $\Lambda$."

# ╔═╡ 923ca5f2-8730-4387-8555-de2a53a79d3b
md"posterior mean $\Lambda$:"

# ╔═╡ 14a77fc0-798f-4507-93b8-e9f04d664a9b
μ_λ = mean(chain[:, "λ"]) # hr

# ╔═╡ 5cb9345b-4ad5-4a90-b5d1-5017c2be5b11
md"posterior standard deviation of $\Lambda$:"

# ╔═╡ 7d425e50-22be-4e38-8ac6-125ec81cf294
σ_λ = std(chain[:, "λ"]) # hr

# ╔═╡ b9bd7754-e45b-4e47-a6ef-71045aded2b7
md"90% equal-tailed posterior credible interval for $\Lambda$:"

# ╔═╡ 2cba4785-44f2-4cb2-8e77-b63489609661
ci_λ = [percentile(chain[:, "λ"], 5.0), percentile(chain[:, "λ"], 95.0)]

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

# ╔═╡ 65efd4db-9e7d-4674-abd7-4f30f957ab5a
md"🚀 voila, the histogram above represents our posterior beliefs about the parameter $\lambda$ in light of the data! from the standpoint of BSI, this posterior distribution constitutes the solution to the parameter inference problem, which (i) incorporates our prior information and (ii) quantifies uncertainty."

# ╔═╡ dd688e57-d441-4fa3-ba4f-cfed0baf1725
md"we also visualize the posterior distribution through showing samples from posterior models of lime temperature trajectories (orange)."

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

# ╔═╡ ff00cbb3-90ad-4dcf-9eca-78ae30fc1657
md"finally, we compute the mean and variance of the posterior for $\Sigma$, which we will used in the time reversal problem we tackle next."

# ╔═╡ dc54ee6b-75a8-444c-9d78-921b2230f0b8
μ_σ = mean(chain[:, "σ"])

# ╔═╡ 6c667be8-4b38-4977-8e50-5884bcadb7d0
σ_σ = std(chain[:, "σ"])

# ╔═╡ 476374b4-c5de-42d0-9f5e-cfe606fbac99
md"## time reversal
🔨 **task**: at time $t^\prime>0$ (the duration since the lime was taken out of the refrigerator), infer the initial (at time $t=0$) lime temperature $\Theta_0$.
"

# ╔═╡ 2b9ca0d3-828d-4016-a3d7-2b7b9ccecde7
t′ = 0.68261 # hr

# ╔═╡ 09316442-97e1-47d9-8c9b-4226ecb593cf
md"
**sub-tasks**: infer the:
* variance of the measurement noise, $\Sigma^2$, even though we have information about it from our parameter identification activity above.
* air temperature $\Theta^{\text{air}}$, even though we will take a (noisy) measurement of it.
* parameter $\Lambda$, even though we have a good estimate of it from the parameter inference we just did. the idea here is to propogate our remaining uncertainty about $\lambda$ into this inference for the time reversal problem.

### the heat transfer experiment

we conduct a _second_ lime heat transfer experiment.

🌡 to determine the condition of the experiment, we use the temperature probe to measure the air temperature $\theta^{\text{air}}$, giving datum $\theta^{\text{air}}_\text{obs}$.
"

# ╔═╡ 51ab4d7e-387f-40c9-99ab-33d817dee4f7
θᵃⁱʳ_obs_2 = 18.64 # °C

# ╔═╡ 4eda374d-772c-41da-b02b-0f07921fd64c
md"### the prior distributions

**the initial temperature of the lime**. based on a (generous) range of temperatures encountered in refrigerators, we impose a diffuse prior
```math
\begin{equation}
    \Theta_0\sim \mathcal{U}([0\,^\circ\text{C}, 20\,^\circ\text{C}]).
\end{equation}
```

**the model parameter**. 

we abide by the quote:
> yesterday's posterior is today's prior

based on the posterior for $\Lambda$ above, our informative prior on $\Lambda$ is:
```math
\begin{equation}
    \Lambda \sim \mathcal{N}_{> 0} \left(\mu_\lambda, \sigma_\lambda^2 \right).
\end{equation}
```
where $\mu_\lambda$ and $\sigma_\lambda$ are defined a few cells above.

**variance of measurement noise**. 

we abide by the quote:
> yesterday's posterior is today's prior

based on the posterior for $\Sigma$ above, our informative prior on $\Sigma$ is:
```math
\begin{equation}
    \Sigma \sim \mathcal{N}_{> 0}(\mu_\sigma, \sigma_\sigma^2),
\end{equation}
```
where $\mu_\sigma$ and $\sigma_\sigma$ are defined a few cells above.

**the air temperature**. we construct an informative prior distribution for the air temperature, since we have measured it:
```math
\begin{equation}
    \Theta^{\text{air}} \sim \mathcal{N}(\theta_{\text{obs}}^{\text{air}}, \sigma^2).
\end{equation}
```
where $\theta_{\text{obs}}^{\text{air}}$ is defined as `θᵃⁱʳ_obs_2` in the cell above.
"

# ╔═╡ 0f907a4b-5a78-4666-8fb1-222828ffdf84
md"### the datum

🌡 at time $t^\prime$, we measure the lime temperature $\theta_{\text{obs}}^\prime$.
"

# ╔═╡ 82b9b6a0-ba28-4dab-a9fc-7ba57c6ce27b
θ′_obs = 12.16 # °C

# ╔═╡ 2c6cc92a-4b1d-49ed-9f48-f352adc52b50
data2 = DataFrame("t [hr]"=>[t′], "θ_obs [°C]"=>[θ′_obs]) # put into a data frame for convenience

# ╔═╡ e5eaf808-8eed-4e97-837b-8c57d3f11edd
md"### the posterior distribution

again, following the probabalistic programming paradigm in `Turing.jl`, we (i) code-up our prior distribution and the probabalistic forward model, then (ii) draw samples from the posterior distribution for this time reversal problem:
```math
\begin{equation}
\Theta_0, \Lambda, \Theta^{\text{air}}, \Sigma \mid (t^\prime, \theta_{\text{obs}}^\prime)
\end{equation}
```
now, $\Theta_0$ is the primary unknown---after all, we imposed informative prior distributions on $\Lambda, \Theta^{\text{air}}, \Sigma$.
"

# ╔═╡ 64ad2f95-06c3-4a57-b0aa-f662715f8c19
@model function measure_lime_temp_later(data2)
    # prior distributions
	λ    ~ truncated(Normal(μ_λ, σ_λ), 0.0, nothing) # hr
    σ    ~ truncated(Normal(μ_σ, σ_σ), 0.0, nothing) # °C
    θ₀   ~ Uniform(0.0, 20.0) # °C
    θᵃⁱʳ ~ Normal(θᵃⁱʳ_obs_2, σ) # °C

    # probabilistic forward model
	t′ = data2[1, "t [hr]"]
	# the model prediction
	θ̄ = θ(t′, λ, θ₀, θᵃⁱʳ)
	# the likelihood
	data2[1, "θ_obs [°C]"] ~ Normal(θ̄, σ)
end

# ╔═╡ 350a8e6f-57a0-408e-9d3f-2ac674d38135
chain2 = DataFrame(
	sample(measure_lime_temp_later(data2), NUTS(), MCMCSerial(), nb_samples, nb_chains)
)

# ╔═╡ c87aa335-11d0-4dce-970e-d536435f7a31
md"posterior mean of $\Theta_0$"

# ╔═╡ 2c355bb2-ebfc-42a1-bb3b-e01a6666abef
μ_θ₀ = mean(chain2[:, "θ₀"])

# ╔═╡ a840d3cb-c1a9-42a9-8367-c6824fc00b1b
md"90% equal-tailed posterior credible interval for $\Theta_0$" 

# ╔═╡ b9ee5918-06cb-41bd-9d7f-7f40e245dab2
ci_θ₀ = [percentile(chain2[:, "θ₀"], 5.0), percentile(chain2[:, "θ₀"], 95.0)]

# ╔═╡ 1427ff0e-25f7-4436-8019-3d8f2dca7359
begin
	local fig = Figure()
	local ax  = Axis(fig[1, 1], xlabel="θ₀ [°C]", ylabel="# samples", 
		title="posterior dist'n of Θ₀")
	hist!(chain2[:, "θ₀"])
	ylims!(0, nothing)
	vlines!([μ_θ₀], linestyle=:dash, color=Cycled(2))
	lines!(ci_θ₀, zeros(2), color="black", linewidth=10)
	fig
end

# ╔═╡ 9938c3e5-20f7-455f-8113-b9e2f9e37eb7
md"🚀 voila, the distribution above represents our posterior beliefs about the initial lime temperature, in light of the datum! from the standpoint of BSI, this posterior distribution constitutes the solution to the time reversal problem, which (i) incorporates our prior information and (ii) quantifies uncertainty.
"

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
# ╟─9e86ddd1-0663-4cad-9e42-bf100900d96f
# ╠═9bf8a9dd-a6d4-45e5-bf5f-cc830e6daae9
# ╟─40a3e0f9-78b8-4028-a829-c3eb1a0d87dd
# ╟─c0ff232c-5f9e-4612-9ad4-ae5c4c19217f
# ╠═b96d3466-a21a-4383-b8fa-db9e1b69773f
# ╟─6d83d92f-7010-4973-bcbf-5f26479f5ea2
# ╠═09f1fed0-30e2-421e-8807-1264c8f142d0
# ╟─cc2ae7a6-99db-4c22-ac72-530631a3cdef
# ╠═db781bcd-0950-4b7b-969f-08f1cc4a2449
# ╟─2c2b6580-2743-4a94-8285-c1166d63df1e
# ╟─923ca5f2-8730-4387-8555-de2a53a79d3b
# ╠═14a77fc0-798f-4507-93b8-e9f04d664a9b
# ╟─5cb9345b-4ad5-4a90-b5d1-5017c2be5b11
# ╠═7d425e50-22be-4e38-8ac6-125ec81cf294
# ╟─b9bd7754-e45b-4e47-a6ef-71045aded2b7
# ╠═2cba4785-44f2-4cb2-8e77-b63489609661
# ╠═34651997-a213-4ebd-a8bc-b76b7d14bb69
# ╟─65efd4db-9e7d-4674-abd7-4f30f957ab5a
# ╟─dd688e57-d441-4fa3-ba4f-cfed0baf1725
# ╠═f45b7e47-902d-4c9b-b28b-7a320bf8a16e
# ╟─ff00cbb3-90ad-4dcf-9eca-78ae30fc1657
# ╠═dc54ee6b-75a8-444c-9d78-921b2230f0b8
# ╠═6c667be8-4b38-4977-8e50-5884bcadb7d0
# ╟─476374b4-c5de-42d0-9f5e-cfe606fbac99
# ╠═2b9ca0d3-828d-4016-a3d7-2b7b9ccecde7
# ╟─09316442-97e1-47d9-8c9b-4226ecb593cf
# ╠═51ab4d7e-387f-40c9-99ab-33d817dee4f7
# ╟─4eda374d-772c-41da-b02b-0f07921fd64c
# ╟─0f907a4b-5a78-4666-8fb1-222828ffdf84
# ╠═82b9b6a0-ba28-4dab-a9fc-7ba57c6ce27b
# ╠═2c6cc92a-4b1d-49ed-9f48-f352adc52b50
# ╟─e5eaf808-8eed-4e97-837b-8c57d3f11edd
# ╠═64ad2f95-06c3-4a57-b0aa-f662715f8c19
# ╠═350a8e6f-57a0-408e-9d3f-2ac674d38135
# ╟─c87aa335-11d0-4dce-970e-d536435f7a31
# ╠═2c355bb2-ebfc-42a1-bb3b-e01a6666abef
# ╟─a840d3cb-c1a9-42a9-8367-c6824fc00b1b
# ╠═b9ee5918-06cb-41bd-9d7f-7f40e245dab2
# ╠═1427ff0e-25f7-4436-8019-3d8f2dca7359
# ╟─9938c3e5-20f7-455f-8113-b9e2f9e37eb7
