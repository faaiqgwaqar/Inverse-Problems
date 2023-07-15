### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ b8591bca-2318-11ee-17af-a7bc61c5e4d4
begin
	import Pkg; Pkg.activate()
	using CairoMakie, Turing, DataFrames, PlutoUI

	TableOfContents()
	update_theme!(fontsize=18)
end

# ╔═╡ b110d3e4-fc4a-48c0-a8e9-50f0dba3837c
md"
# problem setup
!!! note
	this is a light tutorial of Bayesian statistical inversion (BSI) for inverse problems, following the problem setup in:
	> F. Waqar, S. Patel, C. Simon. \"A tutorial on the Bayesian statistical approach to inverse problems\" _APL Machine Learning_. (2023) [link](https://arxiv.org/abs/2304.07610)

## experimental setup
a cold lime fruit at temperature $\theta_0$ [°C] rests inside of a refrigerator. at time $t:=0$ [hr], we take the lime outside of the refrigerator and allow it exchange heat with the indoor air, which is at temperature $\theta^{\text{air}}$ [°C]. a temperature probe inserted into the lime allows us to measure the temperature of the lime, $\theta=\theta(t)$ [°C].
"

# ╔═╡ c6b2fd93-a19d-4198-9499-52b4d3484ef6
html"<img src=\"https://raw.githubusercontent.com/faaiqgwaqar/Inverse-Problems/main/tutorial/lime_setup.jpeg\" width=400>"

# ╔═╡ 8649c024-57e7-4918-a97c-b04cd3a0ce36
md"## forward model
we treat the temperature of the lime as spatially uniform. our mathematical model for the temperature of the lime as a function of time $t$ [hr] is:
```math
\begin{equation}
    \theta (t)=\theta^{\text{air}}+(\theta_0-\theta^{\text{air}})e^{-t/\lambda}, \quad \text{for } t\geq 0. 
\end{equation} 
```
the (unknown) parameter $\lambda$ [hr] characterizes the dynamics of heat exchange between the air and the lime.
"

# ╔═╡ 9f72d5fb-c614-4f21-bdb9-2fbc9ebe2361
md"# parameter identification

"

# ╔═╡ Cell order:
# ╠═b8591bca-2318-11ee-17af-a7bc61c5e4d4
# ╟─b110d3e4-fc4a-48c0-a8e9-50f0dba3837c
# ╟─c6b2fd93-a19d-4198-9499-52b4d3484ef6
# ╟─8649c024-57e7-4918-a97c-b04cd3a0ce36
# ╟─9f72d5fb-c614-4f21-bdb9-2fbc9ebe2361
