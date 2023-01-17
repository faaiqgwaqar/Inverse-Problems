### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ 43bcf4b0-fbfc-11ec-0e23-bb05c02078c9
begin
	import Pkg; Pkg.activate()
	using DataFrames, Distributions, Turing, LinearAlgebra, Random, JLD2, ColorSchemes, StatsBase, Colors, PlutoUI, CairoMakie, FileIO
end

# ╔═╡ b1c06c4d-9b4d-4af3-9e9b-3ba993ca83a0
md"# Bayesian statistical inversion"

# ╔═╡ 1dea25e4-51ee-4f32-a97e-8ce316dfb371
begin
	import AlgebraOfGraphics as AoG
	AoG.set_aog_theme!(fonts=[AoG.firasans("Light"), AoG.firasans("Light")])
	the_resolution = (0.9*500, 0.9*380)
	update_theme!(
		fontsize=20, 
		linewidth=4,
		markersize=14,
		titlefont=AoG.firasans("Light"),
		resolution=the_resolution
	)
end

# ╔═╡ 509e3000-a94d-431c-9a4e-2ba1c6f148a3
import ScikitLearn as skl

# ╔═╡ cc8f82f7-a8db-4f45-8ccc-fa5b171eb3e7
skl.@sk_import neighbors: KernelDensity

# ╔═╡ edb44636-d6d4-400f-adc4-75b287a1f993
TableOfContents()

# ╔═╡ 7831a816-e8d4-49c5-b209-078e74e83c5f
isdir("figs") ?  nothing : mkdir("figs")

# ╔═╡ a081eb2c-ff46-4efa-a6cd-ee3e9209e14e
my_colors = AoG.wongcolors()

# ╔═╡ 8931e445-6664-4609-bfa1-9e808fbe9c09
the_colors = Dict("air"        => my_colors[1], 
	              "data"       => my_colors[2],
	              "model"      => my_colors[3], 
	              "prior"      => my_colors[4],
	              "posterior"  => my_colors[5])

# ╔═╡ 3ae0b235-5ade-4c30-89ac-7f0480c0da11
md"## the model"

# ╔═╡ a13ba151-99c1-47ae-b96e-dc90464990b6
function θ_model(t, λ, t₀, θ₀, θᵃⁱʳ)
    if t < t₀
        return θ₀
	end
    return θᵃⁱʳ .+ (θ₀ - θᵃⁱʳ) * exp(-(t - t₀) / λ)
end

# ╔═╡ ee7fd372-22b0-4bf5-a5e9-5e3a5b6e1843
function viz_model_only()
	ts_model = range(0.0, 5.0, length=400)


	
	fig = Figure()
	ax  = Axis(fig[1, 1], 
		       xlabel="time, (t - t₀) / λ", 
		       ylabel="lime temperature, θ(t)", 
		       yticks=([0, 1], ["θ₀", "θᵃⁱʳ"])
	)

	# draw model
	lines!(ts_model, [θ_model(tᵢ, 1, 0, 0, 1) for tᵢ in ts_model],
		   color=the_colors["model"])

	# draw air temp
	hlines!(ax, 1.0, style=:dash, 
			linestyle=:dot, label="θᵃⁱʳ", color=the_colors["air"])

	# adjustments
	ylims!(-0.1, 1.1)
	xlims!(0, 5)
	
	# inset w lime
	img = load("transparent_lime.png")
	ax_inset = Axis(fig, 
		             bbox= BBox(300, 420, 50, 300), aspect=DataAspect()
	)
	image!(ax_inset, rotr90(img))
	hidedecorations!(ax_inset)
	hidespines!(ax_inset)
	
	save("figs/model_soln.pdf", fig)
	return fig
end

# ╔═╡ 8ee1b06d-c255-4ae3-ac8b-06f7498dbf76
viz_model_only()

# ╔═╡ 38304191-f930-41a6-8545-4734a5ad4ecf
md"## helpers for BSI"

# ╔═╡ ff7e4fd8-e34b-478e-ab8a-2f35aba99ba6
function analyze_posterior(chain::Chains, param::Union{String, Symbol})
	θs = Array(chain[param])[:]
	
	μ = mean(θs)
	σ = std(θs)
	
	lb = percentile(θs, 5.0)
	ub = percentile(θs, 95.0)
	
	return (;μ=μ, σ=σ, lb=lb, ub=ub, samples=θs)
end

# ╔═╡ 788f5c20-7ebb-43e7-bd07-46aa6c9fd249
function get_kde_ρ(x::Vector{Float64}; support::Tuple{Float64, Float64}=(-Inf, Inf)) # returns a function
	bw = 1.06 * std(x) * (length(x)) ^ (-1/5)
	
	kde = KernelDensity(bandwidth=bw)
	kde.fit(reshape(x, length(x), 1))

	return y -> ((y < support[1]) || (y > support[2])) ? 0.0 : exp(kde.score_samples(reshape([y], 1, 1))[1])
end

# ╔═╡ 9e78c280-c19b-469b-8a2b-3c9f4b92a2e5
function viz_convergence(chain::Chains, var::String)
	var_range = range(0.9 * minimum(chain[var]), 1.1 * maximum(chain[var]), length=120)
	
	labels = Dict("λ" => "λ [hr]", "θ₀" => "θ₀[°C]")
	
	fig = Figure(resolution=(the_resolution[1], the_resolution[2]*2))
	axs = [Axis(fig[i, 1]) for i = 1:2]
	for (r, c) in enumerate(groupby(DataFrame(chain), "chain"))
		lines!(axs[1], c[:, "iteration"], c[:, var], linewidth=1)
		
		ρ = get_kde_ρ(c[:, var])
		lines!(axs[2], var_range, ρ.(var_range), label="chain $r", linewidth=1)
		xlims!(axs[2], var_range[1], var_range[end])
	end
	axs[1].xlabel = "iteration"
	xlims!(axs[1], 
		minimum(DataFrame(chain)[:, "iteration"])-1, 
		maximum(DataFrame(chain)[:, "iteration"])+1
	)
	axs[2].ylabel = "density"
	axs[1].ylabel = labels[var]
	axs[2].xlabel = labels[var]
	axislegend(axs[2])
	save("convergence_study_$var.pdf", fig)
	fig
end

# ╔═╡ b29797b9-7e2f-4d55-bc39-dba5ad7663de
md"## parameter identification

🥝 read in data.
"

# ╔═╡ 269ac9fa-13f3-443a-8669-e8f13d3518a6
run = 12

# ╔═╡ d32079ef-7ebd-4645-9789-1d258b13b66f
data = load("data_run_$run.jld2")["data"]

# ╔═╡ b2b83a4e-54b0-4743-80c2-d81ac2d394e2
θᵃⁱʳ = load("data_run_$run.jld2")["θᵃⁱʳ"]

# ╔═╡ 2da4df4f-7bd1-4a40-97f3-4861c486e2d6
function _viz_data!(ax, data::DataFrame, θᵃⁱʳ::Float64; incl_label=true, incl_t₀=true)
	max_t = maximum(data[:, "t [hr]"])

	if incl_t₀
		vlines!(ax, [0.0], color="gray", linewidth=1, label=incl_label ? "t₀" : nothing)
	end
	# air temp
	hlines!(ax, θᵃⁱʳ, style=:dash, linestyle=:dot, 
		label=incl_label ? rich("θ", superscript("air")) : nothing, color=the_colors["air"])
	# data
	scatter!(data[:, "t [hr]"], data[:, "θ [°C]"], 
		label=incl_label ? rich("{(t", subscript("i"), ", θ", subscript("i,obs"), ")}") : nothing, strokewidth=1, color=the_colors["data"])
	xlims!(-0.03*max_t, 1.03*max_t)
	ylims!(5, 20)
end

# ╔═╡ f6f7051d-95c0-4a15-86eb-74fb56d46691
md"🥝 priors"

# ╔═╡ ce178132-a07d-4154-83b4-5f536c8f77aa
σ_prior = Uniform(0.0, 1.0) # °C

# ╔═╡ 7b8f64b9-9776-4385-a2f0-38f78d76ef79
λ_prior = truncated(Normal(1.0, 0.3), 0.0, nothing) # hr

# ╔═╡ ecd4ea3f-1775-4c4e-a679-f8e15eaad3f7
@model function likelihood_for_λ(data)
    # Prior distributions.
    σ ~ σ_prior
	λ ~ λ_prior

	# use first and last data pts as prior.
	θ₀ ~ Normal(data[1, "θ [°C]"], σ)
	
	t₀ = 0.0

    # Observations.
    for i in 2:nrow(data)-1
		tᵢ = data[i, "t [hr]"]
		μ = θ_model(tᵢ, λ, t₀, θ₀, θᵃⁱʳ)
        data[i, "θ [°C]"] ~ Normal(μ, σ)
    end

    return nothing
end

# ╔═╡ 2e57666d-b3f4-451e-86fd-781217c1258d
model_λ = likelihood_for_λ(data)

# ╔═╡ bb3ae6a9-5d87-4b90-978e-8674f6c5bd99
chain_λ = sample(model_λ, NUTS(), MCMCSerial(), 2_500, 4; progress=true)

# ╔═╡ f35c7dcd-243a-4a16-8f7d-424c583aa99f
nrow(DataFrame(chain_λ))

# ╔═╡ 44963969-6883-4c7f-a6ed-4c6eac003dfe
viz_convergence(chain_λ, "λ")

# ╔═╡ a8257d2e-fca8-4bd9-8733-f4034836bbb9
σ_posterior = analyze_posterior(chain_λ, "σ")

# ╔═╡ 31c747b3-0ff1-4fae-9707-47f258d4018f
λ_posterior = analyze_posterior(chain_λ, "λ")

# ╔═╡ a1e622ae-7672-4ca2-bac2-7dcc0a500f1f
function viz_posterior_prior(chain::Chains, prior::Distribution, 
	                         var::String, savename::String;
	                         true_var=nothing)
	x = analyze_posterior(chain, var)

	# variable-specific stuff
	xlabels = Dict(
		"λ" => "time constant, λ [hr]",
		"θ₀" => "initial lime temperature, θ₀ [°C]"
	)
	lims = Dict("λ" => [0.0, 2.0], "θ₀" => [-0.5, 20.5])
	
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel=xlabels[var], ylabel="density")

	var_range = range(lims[var]..., length=5000)

	### posterior
	ρ_posterior_f = get_kde_ρ(x.samples, support=(0.0, Inf))
	ρ_posterior = ρ_posterior_f.(var_range)

	### prior
	ρ_prior = [pdf(prior, x) for x in var_range]

	# bands
	band!(var_range, zeros(length(var_range)), ρ_prior,
		  color=(the_colors["prior"], 0.2))
	band!(var_range, zeros(length(var_range)), ρ_posterior,
		  color=(the_colors["posterior"], 0.2))
	
	# lines
	lines!(var_range, ρ_prior, color=the_colors["prior"], label="prior")
	lines!(var_range, ρ_posterior, color=the_colors["posterior"], label="posterior")

	# ci
	lines!([x.lb, x.ub], zeros(2), color="black", 
		linewidth=6)

	# truth
	if ! isnothing(true_var)
		vlines!(true_var, color="black", linestyle=:dash, 
			linewidth=1, label=rich("held-out θ", subscript("0,obs")))
	end

	ylims!(0, nothing)
	xlims!(lims[var]...)

	axislegend()

	fig
end

# ╔═╡ 294e240f-c146-4ef3-b172-26e70ad3ed19
viz_posterior_prior(chain_λ, λ_prior, "λ", "param_id_prior_posterior.pdf")

# ╔═╡ bba69cd4-f56f-4e93-af03-f0b3f56e710e
function _viz_trajectories!(ax, data::DataFrame, θₐᵢᵣ::Float64, chain::Chains)
	# model
	t_lo = -0.5
	if :t₀ in names(chain)
		t_lo = minimum(chain["t₀"]) - 0.5
	end
	ts = range(t_lo, maximum(data[:, "t [hr]"]), length=100)
	for (i, row) in enumerate(eachrow(DataFrame(sample(chain, 100, replace=false))))
		t₀ = 0.0
		if :t₀ in names(chain)
			t₀ = row["t₀"]
		end
		lines!(ts, θ_model.(ts, row["λ"], t₀, row["θ₀"], θₐᵢᵣ),
			   color=(the_colors["model"], 0.1), label=i == 1 ? "model" : nothing)
	end
end

# ╔═╡ 7a01dfaf-fae1-4a8c-a8a2-1ac973bf3197
md"correlation of τ and σ"

# ╔═╡ f20159ad-7f8b-484e-95ea-afdac97f876a
begin
	local fig = Figure()
	local  ax = Axis(fig[1, 1], xlabel="σ", ylabel="λ")
	scatter!(DataFrame(chain_λ)[:, "σ"], DataFrame(chain_λ)[:, "λ"], 
		color=("red", 0.1))
	fig
end

# ╔═╡ f184e3ea-82f9-49f4-afb6-99c609d7936f
cor(DataFrame(chain_λ)[:, "σ"], DataFrame(chain_λ)[:, "λ"])

# ╔═╡ d8e026b9-8943-437e-a08b-2395de35d705
md"## time reversal problem

### determined
"

# ╔═╡ 30bd4bca-4af6-4e1a-8131-75ca18df7a59
label_for_heldout = rich("held-out (t", subscript("0"), ", θ", subscript("0,obs"), ")")

# ╔═╡ 7df25291-a600-449e-a194-3ec7c3f11361
other_run = 11

# ╔═╡ 8f145533-7208-4c25-9b1e-84370c7ac7ca
data_tr = load("data_run_$other_run.jld2")["data"]

# ╔═╡ 4cc1ebb3-9c22-4a05-9a09-82b81073aa79
θᵃⁱʳ_tr = load("data_run_$other_run.jld2")["θᵃⁱʳ"]

# ╔═╡ ac6f1d8d-4402-4737-82f6-4fd098b93b5e
md"use prior on τ from last outcome."

# ╔═╡ 4e68878f-c278-4218-8a52-ce86490981da
λ_prior_tr = truncated(Normal(λ_posterior.μ, λ_posterior.σ), 0.0, nothing)

# ╔═╡ d199b848-a86e-4d7c-bcd0-566f9d8ea052
σ_prior_tr = truncated(Normal(σ_posterior.μ, σ_posterior.σ), 0.0, nothing)

# ╔═╡ 8d358b8d-7432-421a-8661-4550c0457f97
θ₀_prior = Uniform(0.0, 20.0)

# ╔═╡ 8dbbbe1c-4eb6-4ac2-a447-bbaa500e03b4
@model function likelihood_for_θ₀(data, i_obs)
    # Prior distributions.
	θ₀ ~ θ₀_prior
	if data[i_obs, "θ [°C]"] > θ₀_prior.b
		error("prior makes no sense")
	end
	
	σ ~ σ_prior_tr
	λ ~ λ_prior_tr

    # Observation
	tᵢ = data[i_obs, "t [hr]"]
	μ = θ_model(tᵢ, λ, 0.0, θ₀, θᵃⁱʳ_tr)
	data[i_obs, "θ [°C]"] ~ Normal(μ, σ)

    return nothing
end

# ╔═╡ a3ee46bf-9266-4025-8678-e535d0077faf
function posterior_time_reversal(i_obs::Int)
	model_θ₀ = likelihood_for_θ₀(data_tr, i_obs)
	return sample(model_θ₀, NUTS(), MCMCSerial(), 2_500, 4; progress=true)
end

# ╔═╡ 62c5e645-285d-470e-b46b-00f0471b7329
i_obs = 17 # and try 10, 17

# ╔═╡ 9af1cae7-59b0-4521-a8f9-a000494b8471
function _viz_data!(ax, data::DataFrame, i_obs::Int; incl_test=false, incl_legend=true, incl_t₀=true)
	max_t = maximum(data[:, "t [hr]"])

	if incl_t₀
		vlines!(ax, [0.0], color="gray", linewidth=1, label=incl_legend ? "t₀" : nothing)
	end
	# air temp
	hlines!(ax, θᵃⁱʳ_tr, style=:dash, linestyle=:dot, 
		label=incl_legend ? rich("θ", superscript("air")) : nothing, color=the_colors["air"])
	# data
	scatter!(data[i_obs, "t [hr]"], data[i_obs, "θ [°C]"], 
		label=incl_legend ? rich("(t', θ'", subscript("obs"), ")") : nothing, strokewidth=1, color=the_colors["data"])
	if incl_test
		scatter!(data[1, "t [hr]"], data[1, "θ [°C]"], 
			label=label_for_heldout, strokewidth=1, color="white")
	end
	xlims!(-0.5, 1.03*max_t)
	ylims!(0, 20)
end

# ╔═╡ 1b450ca5-f58f-40d9-baee-84ae539aba31
function viz_data(data::DataFrame, θᵃⁱʳ::Float64; savename=nothing)
	fig = Figure()
	ax  = Axis(fig[1, 1], 
		       xlabel="time, t [hr]",
		       ylabel="lime temperature [°C]",
	)
	_viz_data!(ax, data, θᵃⁱʳ)
	axislegend(position=:rb)
	if ! isnothing(savename)
		save(savename, fig)
	end
	fig
end

# ╔═╡ cd46a3c7-ae78-4f3c-8ba6-c4a55d598843
function viz_trajectories(
				   data::DataFrame, 
				   θᵃⁱʳ::Float64,
	               chain::Chains;
				   savename=nothing
)
	fig = Figure()
	ax  = Axis(fig[1, 1], 
		       xlabel="time, t [hr]",
		       ylabel="lime temperature [°C]",
	)
	_viz_trajectories!(ax, data, θᵃⁱʳ, chain)
	_viz_data!(ax, data, θᵃⁱʳ, incl_label=false, incl_t₀=false)
	

	axislegend(position=:rb)
	if ! isnothing(savename)
		save(savename, fig)
	end

	fig
end

# ╔═╡ b00bc0b4-c33e-4f5e-98f9-68085bd3d94d
function viz_data(data::DataFrame, i_obs::Int; savename=nothing, incl_t₀=true)
	fig = Figure()
	ax  = Axis(fig[1, 1], 
		       xlabel="time, t [hr]",
		       ylabel="lime temperature [°C]",
	)
	_viz_data!(ax, data, i_obs, incl_t₀=incl_t₀)
	axislegend(position=:rb)
	if ! isnothing(savename)
		save(savename, fig)
	end
	fig
end

# ╔═╡ a4192388-5fca-4d61-9cc0-27029032b765
viz_data(data, θᵃⁱʳ, savename="figs/param_id_data.pdf")

# ╔═╡ 8e7ae1d5-fade-4b90-8dd7-e61e965f3609
viz_data(data_tr, i_obs)

# ╔═╡ 07b22d3a-d616-4c89-98c6-d7ee1cd314b6
data_tr[i_obs, :]

# ╔═╡ efdf4047-81ab-45db-9980-267df2bad314
chain_θ₀ = posterior_time_reversal(i_obs)

# ╔═╡ 6e4c92c2-ab69-4ac7-9144-05cc3b8b0dd9
nrow(DataFrame(chain_θ₀))

# ╔═╡ 3f954d0a-3f4e-43c9-b028-f2abdc83792a
viz_convergence(chain_θ₀, "θ₀")

# ╔═╡ db79cc93-0459-42b2-a800-6a1bc7eec1db
viz_posterior_prior(chain_θ₀, θ₀_prior, "θ₀", "posterior_tr", 
	true_var=data_tr[1, "θ [°C]"])

# ╔═╡ 9a4f8bc7-bbc7-42d2-acf2-992d740f9d8b
function viz_trajectories(
				   data::DataFrame, 
	               chain::Chains,
				   i_obs::Int;
				   savename=nothing,
				   incl_t₀=true
)
	fig = Figure()
	ax  = Axis(fig[1, 1], 
		       xlabel="time, t [hr]",
		       ylabel="lime temperature [°C]",
	)

	# trajectories
	_viz_trajectories!(ax, data, θᵃⁱʳ_tr, chain)
	
	# data
	_viz_data!(ax, data, i_obs, incl_test=true, incl_legend=false, incl_t₀=incl_t₀)
	
	axislegend(position=:rb)

	if :t₀ in names(chain)
		xlims!(minimum(DataFrame(chain)[:, "t₀"]), nothing)
	end
	
	if ! isnothing(savename)
		save(savename, fig)
	end

	fig
end

# ╔═╡ b6b05d1b-5e2f-4082-a7ef-1211024c700b
viz_trajectories(data, θᵃⁱʳ, chain_λ)

# ╔═╡ 5cd464bb-710a-4e57-a51a-2ebad433e874
viz_trajectories(data_tr, chain_θ₀, i_obs)

# ╔═╡ 1e5ba0b1-c129-410c-9048-89a75210fd40
md"### underdetermined"

# ╔═╡ 364f2880-6a27-49a0-b5d4-1c6fd6f43293
viz_data(data_tr, i_obs, incl_t₀=false)

# ╔═╡ 4d931a20-2ab7-43c7-91ed-8f4fd40648a5
t₀_prior = truncated(Normal(0.0, 0.25), -1.0, 1.0)

# ╔═╡ 8b1f8a44-612c-4032-93a7-7b0c21c47c31
@model function likelihood_for_θ₀_t₀(data, i_obs)
    # Prior distributions.
	θ₀ ~ θ₀_prior
	if data[i_obs, "θ [°C]"] > θ₀_prior.b
		error("prior makes no sense")
	end
	σ ~ σ_prior_tr
	λ ~ λ_prior_tr
	t₀ ~ t₀_prior

    # Observation
	tᵢ = data[i_obs, "t [hr]"]
	μ = θ_model(tᵢ, λ, t₀, θ₀, θᵃⁱʳ_tr)
	data[i_obs, "θ [°C]"] ~ Normal(μ, σ)

    return nothing
end

# ╔═╡ 845bdbf7-f30e-4f0c-a8db-6f272e76eec9
model_θ₀_t₀ = likelihood_for_θ₀_t₀(data_tr, i_obs)

# ╔═╡ 14bee7d1-dadc-41be-9ea0-1420cd68a121
chain_θ₀_t₀ = sample(model_θ₀_t₀, NUTS(), MCMCSerial(), 2_500, 4; progress=true)

# ╔═╡ 8b176631-b5a7-4c2b-afc7-9dacd0d22d0c
viz_trajectories(data_tr, chain_θ₀_t₀, i_obs, incl_t₀=false)

# ╔═╡ 7824672b-e69d-435d-a8ab-d62f014374d3
function get_ρ_posterior_t₀_θ₀(chain_θ₀_t₀::Chains)
	X = Matrix(DataFrame(chain_θ₀_t₀)[:, [:θ₀, :t₀]])

	# standardize for an isotropic kernel
	μ = mean(X, dims=1)
	σ = std(X, dims=1)
	X̂ = (X .- μ) ./ σ

	# fit KDE
	kde = KernelDensity(bandwidth=0.1)
	kde.fit(X̂)

	# return function that gives density
	ρ = x -> x[1] < 0 ? 0 : exp(kde.score_samples((reshape(x, 1, 2) .- μ) ./ σ)[1])
	return ρ
end

# ╔═╡ f8092ba3-54c7-4e2d-a885-f5ef6c6e094e
function viz_θ₀_t₀_distn(θ₀_prior::Distribution, 
	                     t₀_prior::Distribution, 
	                     chain_θ₀_t₀::Chains)
	the_colormaps = Dict(
		"posterior" => range(RGB(1.0, 1.0, 1.0), 
			RGB(the_colors["posterior"].r,the_colors["posterior"].g,the_colors["posterior"].b)
		),
		"prior" => range(RGB(1.0, 1.0, 1.0), 
			RGB(the_colors["prior"].r,the_colors["prior"].g,the_colors["prior"].b)
		),
	)
	
	fig = Figure(resolution=(600, 600))
	gl = fig[1, 1] = GridLayout()
	ax  = Axis(fig[2, 2], xlabel="t₀ [min]", ylabel="θ₀ [°C]", aspect=1)
	ax_t = Axis(fig[1, 2], ylabel="marginal\ndensity", yticks=[0])
	ax_r = Axis(fig[2, 3], xlabel="marginal\ndensity", xticks=[0])
	linkyaxes!(ax_r, ax)
	linkxaxes!(ax_t, ax)
	hidexdecorations!(ax_t)
	hideydecorations!(ax_r)

	t̂₀ = analyze_posterior(chain_θ₀_t₀, :t₀)
	θ̂₀ = analyze_posterior(chain_θ₀_t₀, :θ₀)
	
	t₀s = range(-0.6, 0.6, length=100)
	θ₀s = range(-0.5, 20.5, length=101)

	# joint!
	ρ_posterior = get_ρ_posterior_t₀_θ₀(chain_θ₀_t₀)

	ρs = Dict("prior" => zeros(length(θ₀s), length(t₀s)),
			  "posterior" => zeros(length(θ₀s), length(t₀s)))
	ρs["prior"] = [pdf(θ₀_prior, θ₀) * pdf(t₀_prior, t₀)
						for t₀ in t₀s, θ₀ in θ₀s]
	ρs["posterior"] = [ρ_posterior([θ₀, t₀])
						for t₀ in t₀s, θ₀ in θ₀s]
	for p in ["prior", "posterior"]
		contour!(ax, t₀s, θ₀s, ρs[p], linewidth=1, 
		         colormap=the_colormaps[p], label=p, levels=8)
	end
		# colormap=the_colormaps[posterior_or_prior], extendhigh=the_colors["distn2"])

	# marginals!
    ρ_posterior_θ₀ = get_kde_ρ(θ̂₀.samples)
	ρ_posterior_t₀ = get_kde_ρ(t̂₀.samples)
	lines!(ax_t, t₀s, ρ_posterior_t₀.(t₀s), 
		color=the_colors["posterior"], linewidth=2)
	lines!(ax_r, ρ_posterior_θ₀.(θ₀s), θ₀s,
		color=the_colors["posterior"], linewidth=2)
	# elseif posterior_or_prior == "prior"
		
	# 	# t₀
	# 	t₀s = [t₀_prior.a - ϵ, t₀_prior.a + ϵ, t₀_prior.b - ϵ, t₀_prior.b + ϵ]
	# 	ρ_t₀_prior = [pdf(t₀_prior, t₀) for t₀ in t₀s]
	lines!(ax_t, t₀s, [pdf(t₀_prior, t₀) for t₀ in t₀s], 
		color=the_colors["prior"], linewidth=2)
	lines!(ax_r, [pdf(θ₀_prior, θ₀) for θ₀ in θ₀s], θ₀s, 
		color=the_colors["prior"], linewidth=2)
	# 	band!(ax_t, t₀s, zeros(length(t₀s)), ρ_t₀_prior, 
	# 			color=(the_colors["distn2"], 0.4))
	# 	# T₀
	# 	T₀s = [T₀_prior.a - ϵ, T₀_prior.a + ϵ, T₀_prior.b - ϵ, T₀_prior.b + ϵ]
	# 	ρ_T₀_prior = [pdf(T₀_prior, T₀) for T₀ in T₀s]
	# 	lines!(ax_r, ρ_T₀_prior, T₀s, color=:black, linewidth=1)
	# 	band_direction_y!(ax_r, T₀s, 
	# 		   zeros(length(T₀s)), ρ_T₀_prior, (the_colors["distn2"], 0.4))
	# end
	# # truth
	scatter!(ax, data_tr[1, "t [hr]"], data_tr[1, "θ [°C]"], 
	         label=label_for_heldout, strokewidth=2, color=(:white, 0.0))
	
	
	# # ylims!(ax, 0, nothing)
	# # xlims!(ax, 0, 150.0)
	# # ylims!(ax_t, 0, nothing)
	# xlims!(ax_r, 0, nothing)
	# ylims!(ax_t, 0, nothing)

	colgap!(gl, 5)
	rowgap!(gl, 5)
	# xlims!(ax, minimum(t₀s), maximum(t₀s))
	# ylims!(ax, minimum(T₀s), maximum(T₀s))
	rowsize!(fig.layout, 1, Relative(.225))
	colsize!(fig.layout, 3, Relative(.225))
	# Colorbar(fig[2, 1], colormap=ColorSchemes.Wistia, ticks=[0], 
	# 	label="posterior density", flipaxis=false)
	# save("posterior_theta_0_t_0_i_obs_$(i_obs).pdf", fig)
	fig
	# ρs
end

# ╔═╡ 0b0af726-3eb7-4939-bdd5-7b76213d5485
viz_θ₀_t₀_distn(θ₀_prior, t₀_prior, chain_θ₀_t₀)

# ╔═╡ b14d545e-bc9e-493b-877f-899ec4ddc8fc
begin
	# show curve of solutions
	θ_0s = range(-1, 15.0, length=100)
	t′ = data2[i_obs, "t [hr]"]
	θ′ = data2[i_obs, "T [°C]"]
	λ̄ = analyze_posterior(chain_τ, "τ").μ
	t_0s = t′ .- λ̄ * log.((θ_0s .- fixed_params2.Tₐ) ./ (θ′ - fixed_params2.Tₐ))
end

# ╔═╡ 8ba02a50-98f8-4c83-9f4f-040a1aad8274
md"to check..."

# ╔═╡ Cell order:
# ╟─b1c06c4d-9b4d-4af3-9e9b-3ba993ca83a0
# ╠═43bcf4b0-fbfc-11ec-0e23-bb05c02078c9
# ╠═1dea25e4-51ee-4f32-a97e-8ce316dfb371
# ╠═509e3000-a94d-431c-9a4e-2ba1c6f148a3
# ╠═cc8f82f7-a8db-4f45-8ccc-fa5b171eb3e7
# ╠═edb44636-d6d4-400f-adc4-75b287a1f993
# ╠═7831a816-e8d4-49c5-b209-078e74e83c5f
# ╠═a081eb2c-ff46-4efa-a6cd-ee3e9209e14e
# ╠═8931e445-6664-4609-bfa1-9e808fbe9c09
# ╟─3ae0b235-5ade-4c30-89ac-7f0480c0da11
# ╠═a13ba151-99c1-47ae-b96e-dc90464990b6
# ╠═ee7fd372-22b0-4bf5-a5e9-5e3a5b6e1843
# ╠═8ee1b06d-c255-4ae3-ac8b-06f7498dbf76
# ╟─38304191-f930-41a6-8545-4734a5ad4ecf
# ╠═ff7e4fd8-e34b-478e-ab8a-2f35aba99ba6
# ╠═9e78c280-c19b-469b-8a2b-3c9f4b92a2e5
# ╠═788f5c20-7ebb-43e7-bd07-46aa6c9fd249
# ╟─b29797b9-7e2f-4d55-bc39-dba5ad7663de
# ╠═269ac9fa-13f3-443a-8669-e8f13d3518a6
# ╠═d32079ef-7ebd-4645-9789-1d258b13b66f
# ╠═b2b83a4e-54b0-4743-80c2-d81ac2d394e2
# ╠═2da4df4f-7bd1-4a40-97f3-4861c486e2d6
# ╠═1b450ca5-f58f-40d9-baee-84ae539aba31
# ╠═a4192388-5fca-4d61-9cc0-27029032b765
# ╟─f6f7051d-95c0-4a15-86eb-74fb56d46691
# ╠═ce178132-a07d-4154-83b4-5f536c8f77aa
# ╠═7b8f64b9-9776-4385-a2f0-38f78d76ef79
# ╠═ecd4ea3f-1775-4c4e-a679-f8e15eaad3f7
# ╠═2e57666d-b3f4-451e-86fd-781217c1258d
# ╠═bb3ae6a9-5d87-4b90-978e-8674f6c5bd99
# ╠═f35c7dcd-243a-4a16-8f7d-424c583aa99f
# ╠═44963969-6883-4c7f-a6ed-4c6eac003dfe
# ╠═a8257d2e-fca8-4bd9-8733-f4034836bbb9
# ╠═31c747b3-0ff1-4fae-9707-47f258d4018f
# ╠═a1e622ae-7672-4ca2-bac2-7dcc0a500f1f
# ╠═294e240f-c146-4ef3-b172-26e70ad3ed19
# ╠═bba69cd4-f56f-4e93-af03-f0b3f56e710e
# ╠═cd46a3c7-ae78-4f3c-8ba6-c4a55d598843
# ╠═b6b05d1b-5e2f-4082-a7ef-1211024c700b
# ╟─7a01dfaf-fae1-4a8c-a8a2-1ac973bf3197
# ╠═f20159ad-7f8b-484e-95ea-afdac97f876a
# ╠═f184e3ea-82f9-49f4-afb6-99c609d7936f
# ╟─d8e026b9-8943-437e-a08b-2395de35d705
# ╠═30bd4bca-4af6-4e1a-8131-75ca18df7a59
# ╠═7df25291-a600-449e-a194-3ec7c3f11361
# ╠═8f145533-7208-4c25-9b1e-84370c7ac7ca
# ╠═4cc1ebb3-9c22-4a05-9a09-82b81073aa79
# ╟─ac6f1d8d-4402-4737-82f6-4fd098b93b5e
# ╠═4e68878f-c278-4218-8a52-ce86490981da
# ╠═d199b848-a86e-4d7c-bcd0-566f9d8ea052
# ╠═8d358b8d-7432-421a-8661-4550c0457f97
# ╠═8dbbbe1c-4eb6-4ac2-a447-bbaa500e03b4
# ╠═a3ee46bf-9266-4025-8678-e535d0077faf
# ╠═62c5e645-285d-470e-b46b-00f0471b7329
# ╠═9af1cae7-59b0-4521-a8f9-a000494b8471
# ╠═b00bc0b4-c33e-4f5e-98f9-68085bd3d94d
# ╠═8e7ae1d5-fade-4b90-8dd7-e61e965f3609
# ╠═07b22d3a-d616-4c89-98c6-d7ee1cd314b6
# ╠═efdf4047-81ab-45db-9980-267df2bad314
# ╠═6e4c92c2-ab69-4ac7-9144-05cc3b8b0dd9
# ╠═3f954d0a-3f4e-43c9-b028-f2abdc83792a
# ╠═db79cc93-0459-42b2-a800-6a1bc7eec1db
# ╠═9a4f8bc7-bbc7-42d2-acf2-992d740f9d8b
# ╠═5cd464bb-710a-4e57-a51a-2ebad433e874
# ╟─1e5ba0b1-c129-410c-9048-89a75210fd40
# ╠═364f2880-6a27-49a0-b5d4-1c6fd6f43293
# ╠═4d931a20-2ab7-43c7-91ed-8f4fd40648a5
# ╠═8b1f8a44-612c-4032-93a7-7b0c21c47c31
# ╠═845bdbf7-f30e-4f0c-a8db-6f272e76eec9
# ╠═14bee7d1-dadc-41be-9ea0-1420cd68a121
# ╠═8b176631-b5a7-4c2b-afc7-9dacd0d22d0c
# ╠═7824672b-e69d-435d-a8ab-d62f014374d3
# ╠═f8092ba3-54c7-4e2d-a885-f5ef6c6e094e
# ╠═0b0af726-3eb7-4939-bdd5-7b76213d5485
# ╠═b14d545e-bc9e-493b-877f-899ec4ddc8fc
# ╟─8ba02a50-98f8-4c83-9f4f-040a1aad8274
