### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ b8591bca-2318-11ee-17af-a7bc61c5e4d4
begin
	import Pkg; Pkg.activate()
	using CairoMakie, Turing, DataFrames, PlutoUI, CSV

	update_theme!(fontsize=18)
end

# ╔═╡ 7e0a601d-9081-4d45-983a-7c39c8be28af
TableOfContents()

# ╔═╡ b110d3e4-fc4a-48c0-a8e9-50f0dba3837c
md"
# problem setup
!!! note
	this is a minimal/simplified tutorial of Bayesian statistical inversion (BSI) for inverse problems, following the problem setup and using the data in:
	> F. Waqar, S. Patel, C. Simon. \"A tutorial on the Bayesian statistical approach to inverse problems\" _APL Machine Learning_. (2023) [link](https://arxiv.org/abs/2304.07610)

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
md"## probabalistic model of the temperature measurements
we use the forward model to construct a probabalistic model of the measured lime temperature. we assume any observed measurement $\theta_{\text{obs}}$ [°C] of the lime temperature at time $t \geq 0$ is a realization of a random variable $\Theta_{\text{obs}}$ with a Gaussian distribution
```math
\begin{equation}
    \Theta_{\text{obs}} \sim \mathcal{N}(\theta(t), \sigma^2)
\end{equation}
```
with a mean governed by the model $\theta(t)$ and variance $\sigma^2$ owing to measurement noise and zero-mean residual variability. we treat multiple measurements as independent and identically distributed. 
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

**the unknown model parameter**. based on a back-of-the-envelope estimate of $\lambda$ and our confidence in this estimate

```math
```
"

# ╔═╡ d44fc256-55af-490d-a643-21fb07950f7c
md"## the data

🌡 we employ the temperature probe to measure
"

# ╔═╡ 7c247192-35d3-4448-bd4f-c2443fae6457
md"🌡 time series data of the lime temperature"

# ╔═╡ 2f593ee3-f6ca-4397-9e09-bd9761720f29
data = CSV.read("lime_temp_param_id.csv", DataFrame)

# ╔═╡ c0ff232c-5f9e-4612-9ad4-ae5c4c19217f
md"## the posterior distribution"

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
# ╠═21822a96-a519-4585-92b1-c23d838fb5a6
# ╠═d44fc256-55af-490d-a643-21fb07950f7c
# ╠═7c247192-35d3-4448-bd4f-c2443fae6457
# ╠═2f593ee3-f6ca-4397-9e09-bd9761720f29
# ╟─c0ff232c-5f9e-4612-9ad4-ae5c4c19217f
