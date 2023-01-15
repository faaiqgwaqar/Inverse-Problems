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
	update_theme!(fontsize=20, linewidth=3, resolution=(500, 400))
end

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
	              "prior"      => my_colors[5],
	              "posterior"  => my_colors[end])

# ╔═╡ 3ae0b235-5ade-4c30-89ac-7f0480c0da11
md"## the forward model"

# ╔═╡ a13ba151-99c1-47ae-b96e-dc90464990b6
function θ_model(t::Float64, params)
    if t < 0.0
        return params.θ₀
	end
    return params.θᵃⁱʳ .+ (params.θ₀ - params.θᵃⁱʳ) * exp(-(t - params.t₀) / params.λ)
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
	_params = (t₀=0.0, θ₀=0, θᵃⁱʳ=1, λ=1)
	lines!(ts_model, [θ_model(tᵢ, _params) for tᵢ in ts_model],
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

# ╔═╡ b29797b9-7e2f-4d55-bc39-dba5ad7663de
md"## model parameter identification
"

# ╔═╡ 269ac9fa-13f3-443a-8669-e8f13d3518a6
run = 11

# ╔═╡ d32079ef-7ebd-4645-9789-1d258b13b66f
begin
	data = load("data_run_$run.jld2")["data"]
	data[:, "t [hr]"] = data[:, "t [min]"] / 60
	data
end

# ╔═╡ b8a3fc88-6e4d-457d-8582-f6302fb206ac
fixed_params = (T₀=load("data_run_$run.jld2")["T₀"], 
                Tₐ=load("data_run_$run.jld2")["Tₐ"])

# ╔═╡ ce178132-a07d-4154-83b4-5f536c8f77aa
σ_prior = Uniform(0.0, 1.0) # °C

# ╔═╡ 7b8f64b9-9776-4385-a2f0-38f78d76ef79
τ_prior_1 = Uniform(1.0 / 5, 5.0) # hr

# ╔═╡ ecd4ea3f-1775-4c4e-a679-f8e15eaad3f7
@model function likelihood_for_τ(data, fixed_params)
    # Prior distributions.
    σ ~ σ_prior
	τ ~ τ_prior_1

    # Observations.
    for i in 1:nrow(data)
		tᵢ = data[i, "t [hr]"]
		μ = T_model(tᵢ, τ, fixed_params.T₀, fixed_params.Tₐ)
        data[i, "T [°C]"] ~ Normal(μ, σ)
    end

    return nothing
end

# ╔═╡ 2e57666d-b3f4-451e-86fd-781217c1258d
model_τ = likelihood_for_τ(data, fixed_params)

# ╔═╡ bb3ae6a9-5d87-4b90-978e-8674f6c5bd99
chain_τ = sample(model_τ, NUTS(), MCMCSerial(), 2_500, 4; progress=true)

# ╔═╡ f35c7dcd-243a-4a16-8f7d-424c583aa99f
nrow(DataFrame(chain_τ))

# ╔═╡ 5478b192-677e-4296-8ce5-c6d0447898bc
bw = Dict("τ" => 0.01, "T₀" => 0.05)

# ╔═╡ ff7e4fd8-e34b-478e-ab8a-2f35aba99ba6
function analyze_posterior(chain::Chains, param::Union{String, Symbol})
	θs = Array(chain[param])[:]
	
	μ = mean(θs)
	σ = std(θs)
	
	lb = percentile(θs, 5.0)
	ub = percentile(θs, 95.0)
	
	return (;μ=μ, σ=σ, lb=lb, ub=ub, samples=θs)
end

# ╔═╡ a8257d2e-fca8-4bd9-8733-f4034836bbb9
analyze_posterior(chain_τ, "σ")

# ╔═╡ ff9735a0-1fea-4518-a06a-0af74687ba9c
function smart_bw(values::Array{Float64})
	return 1.06 * std(values) * (length(values)) ^ (-1/5)
end

# ╔═╡ 788f5c20-7ebb-43e7-bd07-46aa6c9fd249
function get_kde_ρ(x::Vector{Float64})
	bw = smart_bw(x)
	kde = KernelDensity(bandwidth=bw)
	kde.fit(reshape(x, length(x), 1))

	return y -> exp(kde.score_samples(reshape([y], 1, 1))[1])
end

# ╔═╡ 9e78c280-c19b-469b-8a2b-3c9f4b92a2e5
function viz_convergence(chain::Chains, var::String)
	var_range = range(0.9 * minimum(chain[var]), 1.1 * maximum(chain[var]), length=120)
	
	labels = Dict("τ" => L"$\lambda$ [hr]", "T₀" => L"$\theta_0$ [°C]")
	
	f, ax = subplots(2, 1, figsize=(10, 6))
	for (r, c) in enumerate(groupby(DataFrame(chain), "chain"))
		ax[1].plot(c[:, "iteration"], c[:, var], linewidth=1)
		
		ρ = get_kde_ρ(c[:, var])
		ax[2].plot(var_range, ρ.(var_range), label="chain $r", linewidth=1)
		ax[2].set_xlim(var_range[1], var_range[end])

	end
	ax[1].set_xlabel("iteration")
	ax[1].set_xlim(minimum(DataFrame(chain)[:, "iteration"])-1, maximum(DataFrame(chain)[:, "iteration"])+1)
	ax[2].set_ylabel("density")
	ax[2].set_ylim(ymin=0)
	ax[2].set_yticks([0])
	ax[1].set_ylabel(labels[var])
	ax[2].set_xlabel(labels[var])
	ax[2].legend()
	tight_layout()
	savefig("figs/convergence_study.pdf", format="pdf")
	f
end

# ╔═╡ 44963969-6883-4c7f-a6ed-4c6eac003dfe
viz_convergence(chain_τ, "τ")

# ╔═╡ 2378f74e-ccd6-41fd-89f5-6001b75ea741
alpha = 0.4

# ╔═╡ a1e622ae-7672-4ca2-bac2-7dcc0a500f1f
function viz_posterior_prior(chain::Chains, prior::Distribution, 
	                         var::String, savename::String;
	                         true_var=nothing)
	x = analyze_posterior(chain, var)

	# variable-specific stuff
	xlabels = Dict(
		"τ" => L"time constant, $\lambda$ [hr]",
		"T₀" => L"initial temperature, $\theta_0$ [°C]"
	)
	short_xlabels = Dict(
		"τ" => L"$\lambda$ [hr]",
		"T₀" => L"$\theta_0$ [°C]"
	)
	posterior_lims = Dict("τ" => [0.95, 1.2], "T₀" => [0.0, 15.0])
	
	fig, ax = myfig()
	xlabel(xlabels[var])
	ylabel("posterior density")

	###
	# posterior
	var_range = range(posterior_lims[var]..., length=150)
	ρ = get_kde_ρ(x.samples)
	plot(var_range, ρ.(var_range), color="black", label="posterior")
	fill_between(var_range, zeros(length(var_range)), ρ.(var_range),
				 color=the_colors["posterior"], alpha=alpha)
	
	plot([x.lb, x.ub], [0, 0], c="gray", linewidth=5, clip_on=false)

	if var == "T₀"
		axvline([0], color="gray", linewidth=1)
		xlim([-1, 16])
	else
		xlim(posterior_lims[var]...)
	end
	ylim(ymin=0)
	yticks([0])

	if ! isnothing(true_var)
		axvline([true_var], linestyle="dashed", color=the_colors["data"])
	end

	###
	# prior
	inset = ax.inset_axes([0.75, 0.7, 0.3, 0.3])
	inset.set_xlabel(short_xlabels[var])
	inset.set_ylabel("prior\ndensity")
	inset.set_ylim(ymin=0)
	
	var_range = [prior.a, prior.b]
	var_range = vcat(var_range .+ 0.0001, var_range .- 0.0001)
	sort!(var_range)
	ρ = [pdf(prior, x) for x in var_range]
	inset.plot(var_range, ρ, color="black", label="prior")
	inset.fill_between(var_range, zeros(length(var_range)), ρ,
				 color=the_colors["prior"], alpha=alpha)
	inset.axvline([0], color="gray", linewidth=1)
	inset.set_yticks([0])
	if var == "τ"
		inset.set_xlim(-0.5, 5.5)
	end
	if var == "T₀"
		inset.set_xticks([0, 10, 20])
		inset.set_xlim(-1, 21)
	end
	inset.set_ylim(0, maximum(ρ)*2)

	# # posterior
	println("ci = ", round.([x.lb, x.ub], digits=2))
	tight_layout()
	savefig("figs/" * savename, format="pdf")
	fig
end

# ╔═╡ 294e240f-c146-4ef3-b172-26e70ad3ed19
viz_posterior_prior(chain_τ, τ_prior_1, "τ", "param_id_prior_posterior.pdf")

# ╔═╡ cd46a3c7-ae78-4f3c-8ba6-c4a55d598843
function viz_b4_after_inference(
				   data::DataFrame, 
	               fixed_params::NamedTuple, 
	               chain::Chains;
				   i_obs=nothing
)
	max_t = maximum(data[:, "t [hr]"])
    t = range(0.0, max_t * 1.05, length=200)
	
	fig, axs = subplots(1, 2, sharey=true, sharex=true,
			figsize=(
				rcParams["figure.figsize"][1]*1.8, 
			    rcParams["figure.figsize"][2]
			)
		)

	if ! isnothing(i_obs)
		axs[2].scatter(data[:, "t [min]"] / 60.0, data[:, "T [°C]"], 
			    edgecolors="black",
				label=L"test data$\{(t_i, θ_{\rm{obs},i})\}$", color="white")
	end
	for i = 1:2
		for s in ["top","right"]
			if s == "bottom"
				continue
			end
			axs[i].spines[s].set_visible(false)
		end
		axs[i].set_xlabel(L"time, $t$ [hr]")
		axs[i].axhline([fixed_params.Tₐ], linestyle="dashed", zorder=0,
			color=the_colors["air"], label=i == 2 ? "" : L"$\theta^{\rm{air}}$")
		axs[i].axvline([0.0], color="gray", linewidth=1, zorder=0)
		if isnothing(i_obs)
			axs[i].scatter(data[:, "t [hr]"], data[:, "T [°C]"], 	
				edgecolors="black",
				label=i == 2 ? "" : L"$\{(t_i, θ_{\rm{obs},i})\}_{i=0}^N$", color=the_colors["data"])
		else
			axs[i].scatter(data[i_obs, "t [hr]"], data[i_obs, "T [°C]"], 	
				edgecolors="black",
				label=i == 2 ? "" : L"$(t_i\prime, θ_{\rm{obs}}\prime)$", color=the_colors["data"], zorder=1000)
		end
	end
	axs[1].set_ylabel("temperature [°C]")
	axs[1].set_title("before BSI")
	axs[2].set_title("after BSI")

	for (i, row) in enumerate(eachrow(DataFrame(sample(chain, 100, replace=false))))
		if isnothing(i_obs)
			axs[2].plot(t, T_model.(t, row[:τ], fixed_params.T₀, fixed_params.Tₐ),
				  color=the_colors["model"], alpha=0.1, 
				  label= (i == 1) ? L"$\theta(t;\lambda)$" : "")
		else
			axs[2].plot(t, T_model.(t, row[:τ], row[:T₀], fixed_params.Tₐ),
				  color=the_colors["model"], alpha=0.1, 
				  label=(i == 1) ? L"$\theta(t;\theta_0)$" : "")
		end
	end
	for i = 1:2
		axs[i].legend(loc="lower right", fontsize=16)
	end
	# end
	ylim(0, 20.0)
	xlim(-0.03*max_t, 10.2)
	tight_layout()
	if isnothing(i_obs)
		savefig("figs/param_id_b4_after_BSI.pdf", format="pdf")
	else
		savefig("figs/time_reversal_id_$(i_obs)_id_b4_after_BSI.pdf", format="pdf")
	end
	return fig
end

# ╔═╡ b6b05d1b-5e2f-4082-a7ef-1211024c700b
viz_b4_after_inference(data, fixed_params, chain_τ)

# ╔═╡ 7a01dfaf-fae1-4a8c-a8a2-1ac973bf3197
md"correlation of τ and σ"

# ╔═╡ f20159ad-7f8b-484e-95ea-afdac97f876a
begin
	local fig = figure()
	xlabel("σ")
	ylabel("τ")
	scatter(DataFrame(chain_τ)[:, "σ"], DataFrame(chain_τ)[:, "τ"], 
		c=the_colors["prior"], alpha=0.5)
	fig
end

# ╔═╡ f184e3ea-82f9-49f4-afb6-99c609d7936f
cor(DataFrame(chain_τ)[:, "σ"], DataFrame(chain_τ)[:, "τ"])

# ╔═╡ d8e026b9-8943-437e-a08b-2395de35d705
md"## time reversal problem"

# ╔═╡ 7df25291-a600-449e-a194-3ec7c3f11361
other_run = 12

# ╔═╡ 8f145533-7208-4c25-9b1e-84370c7ac7ca
begin
	data2 = load("data_run_$other_run.jld2")["data"]
	data2[:, "t [hr]"] = data2[:, "t [min]"] / 60.0
end

# ╔═╡ 0bff14a8-89eb-488c-88c6-e08a64e577ed
fixed_params2 = (T₀=load("data_run_$other_run.jld2")["T₀"], 
                 Tₐ=load("data_run_$other_run.jld2")["Tₐ"])

# ╔═╡ ac6f1d8d-4402-4737-82f6-4fd098b93b5e
md"use prior on τ from last outcome."

# ╔═╡ 4e68878f-c278-4218-8a52-ce86490981da
begin
	_τ_prior = analyze_posterior(chain_τ, :τ)
	τ_prior2 = truncated(Normal(_τ_prior.μ, _τ_prior.σ), 0.0, nothing)
end

# ╔═╡ d199b848-a86e-4d7c-bcd0-566f9d8ea052
begin
	_σ_prior = analyze_posterior(chain_τ, :σ)
	σ_prior2 = truncated(Normal(_σ_prior.μ, _σ_prior.σ), 0.0, nothing)
end

# ╔═╡ 54efdfb6-bb64-4834-8cd9-a3f126f731e9
_σ_prior

# ╔═╡ 8d358b8d-7432-421a-8661-4550c0457f97
T₀_prior = Uniform(0.0, fixed_params2.Tₐ)

# ╔═╡ 8dbbbe1c-4eb6-4ac2-a447-bbaa500e03b4
@model function likelihood_for_T₀(data, i_obs, Tₐ)
    # Prior distributions.
	T₀ ~ T₀_prior
	if data[i_obs, "T [°C]"] > T₀_prior.b
		error("prior makes no sense")
	end
	σ ~ σ_prior2
	τ ~ τ_prior2

    # Observation
	tᵢ = data[i_obs, "t [hr]"]
	μ = T_model(tᵢ, τ, T₀, Tₐ)
	data[i_obs, "T [°C]"] ~ Normal(μ, σ)

    return nothing
end

# ╔═╡ a3ee46bf-9266-4025-8678-e535d0077faf
function posterior_time_reversal(i_obs::Int)
	model_T₀ = likelihood_for_T₀(data2, i_obs, fixed_params2.Tₐ)
	chain_T₀ = sample(model_T₀, NUTS(), MCMCSerial(), 2_500, 4; progress=true)
end

# ╔═╡ 62c5e645-285d-470e-b46b-00f0471b7329
i_obs = 34 # and try 35 and 30

# ╔═╡ 07b22d3a-d616-4c89-98c6-d7ee1cd314b6
data2[i_obs, :]

# ╔═╡ efdf4047-81ab-45db-9980-267df2bad314
chain_T₀ = posterior_time_reversal(i_obs)

# ╔═╡ 6e4c92c2-ab69-4ac7-9144-05cc3b8b0dd9
nrow(DataFrame(chain_T₀))

# ╔═╡ 3f954d0a-3f4e-43c9-b028-f2abdc83792a
viz_convergence(chain_T₀, "T₀")

# ╔═╡ bd5602cd-8b6d-430f-a700-40b449d1da27
viz_posterior_prior(chain_T₀, T₀_prior, "T₀", "time_reversal_prior_posterior_id_$i_obs.pdf", true_var=data2[1, "T [°C]"])

# ╔═╡ ba77054e-1754-4c62-bce9-7e166bd99a6e
viz_b4_after_inference(data2, fixed_params2, chain_T₀, i_obs=i_obs)

# ╔═╡ e84e11c6-eba4-45de-82b7-d4f0c76e4c94
gridspec = PyPlot.matplotlib.gridspec

# ╔═╡ 8c8ce05d-45da-4a1a-bfce-457282e4237e
function ridge_plot()

	i_obs_list = 2:4:35

	fig = figure(figsize=(7.0*0.9, 4.8*0.9))
	gs = fig.add_gridspec(length(i_obs_list), hspace=-0.6)
	axs = gs.subplots(sharex=true, sharey=true)

	θ₀s = range(0.0, 15.0, length=100)
	the_ymax = 0.0
	for i = 1:length(i_obs_list)
		rect = axs[i].patch
		rect.set_alpha(0)
		for s in ["top", "right", "left", "bottom"]
			if s == "bottom"
				continue
			end
			axs[i].spines[s].set_visible(false)
		end
		axs[i].set_yticks([])
		
		axs[i].set_xlim([0, 15])
		t′ = data2[i_obs_list[i], "t [hr]"]
		axs[i].text(-0.05, 0.075, "t′ = $(round(t′, digits=2)) hr",
			transform=axs[i].transAxes)
		if i != length(i_obs_list)
			axs[i].set_xticks([])
		end
		# posterior
		chain_T₀ = posterior_time_reversal(i_obs_list[i])
		ρ = get_kde_ρ(analyze_posterior(chain_T₀, "T₀").samples)
		ρ_post = ρ.(θ₀s)
		axs[i].plot(θ₀s, ρ_post, color="black", linewidth=1)
		axs[i].fill_between(θ₀s, zeros(length(θ₀s)), ρ_post, 
					color=the_colors["prior"], label="prior", alpha=0.4)

		the_ymax = maximum(vcat(ρ_post, [the_ymax]))
	end
	axs[1].set_ylim(0, the_ymax * 1.05)
	axs[end].set_xlabel(L"initial temperature, $\theta_0$ [°C]")
	tight_layout()
	# savefig("posterior_tau.pdf", format="pdf")
	fig
end

# ╔═╡ 3893d1d9-e98e-4aa1-8723-41e1c2b158fd
ridge_plot()

# ╔═╡ 1e5ba0b1-c129-410c-9048-89a75210fd40
md"## the ill-posed inverse problem"

# ╔═╡ da778a83-aa3d-427f-9cd7-eede559c5c37
t₀_prior = truncated(Normal(0.0, 0.25), -1.0, 1.0)

# ╔═╡ 8b1f8a44-612c-4032-93a7-7b0c21c47c31
@model function likelihood_for_T₀_t₀(data, i_obs, Tₐ)
    # Prior distributions.
	T₀ ~ T₀_prior
	if data[i_obs, "T [°C]"] > T₀_prior.b
		error("prior makes no sense")
	end
	σ ~ σ_prior2
	τ ~ τ_prior2
	t₀ ~ t₀_prior

    # Observation
	tᵢ = data[i_obs, "t [hr]"]
	μ = T_model(tᵢ, τ, T₀, Tₐ, t₀)
	data[i_obs, "T [°C]"] ~ Normal(μ, σ)

    return nothing
end

# ╔═╡ 845bdbf7-f30e-4f0c-a8db-6f272e76eec9
model_T₀_t₀ = likelihood_for_T₀_t₀(data2, i_obs, fixed_params2.Tₐ)

# ╔═╡ 14bee7d1-dadc-41be-9ea0-1420cd68a121
chain_T₀_t₀ = sample(model_T₀_t₀, NUTS(), MCMCSerial(), 2_500, 4; progress=true)

# ╔═╡ aaca06d8-0e20-4c53-9097-d69fe1ae3d83
posterior_colormap = PyPlot.matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap",
	["white", the_colors["posterior"]])

# ╔═╡ d812222a-3d59-418e-a67c-4154e0fd6e23
prior_colormap = PyPlot.matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap",
	["white", the_colors["prior"]])

# ╔═╡ 7824672b-e69d-435d-a8ab-d62f014374d3
function get_ρ_posterior_t₀_T₀()
	X = Matrix(DataFrame(chain_T₀_t₀)[:, [:T₀, :t₀]])
	μ = mean(X, dims=1)
	σ = std(X, dims=1)
	X̂ = (X .- μ) ./ σ
	kde = KernelDensity(bandwidth=0.1)
	kde.fit(X̂)
	return x -> exp(kde.score_samples((reshape(x, 1, 2) .- μ) ./ σ)[1])
end

# ╔═╡ b14d545e-bc9e-493b-877f-899ec4ddc8fc
begin
	# show curve of solutions
	θ_0s = range(-1, 15.0, length=100)
	t′ = data2[i_obs, "t [hr]"]
	θ′ = data2[i_obs, "T [°C]"]
	λ̄ = analyze_posterior(chain_τ, "τ").μ
	t_0s = t′ .- λ̄ * log.((θ_0s .- fixed_params2.Tₐ) ./ (θ′ - fixed_params2.Tₐ))
end

# ╔═╡ 58a95e76-01db-48c4-981b-d212aff54029
function new_undetermined_viz()
	fig = figure(figsize=(6, 6))
	gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
	                      left=0.1, right=0.9, bottom=0.1, top=0.9,
	                      wspace=0.05, hspace=0.05)
	# Create the Axes.
	ax_joint = fig.add_subplot(gs[2, 1])
	ax_marg_x = fig.add_subplot(gs[1, 1], sharex=ax_joint)
	ax_marg_y = fig.add_subplot(gs[2, 2], sharey=ax_joint)
	for _ax in [ax_joint, ax_marg_x, ax_marg_y]
	    _ax.spines["right"].set_visible(false)
	    _ax.spines["top"].set_visible(false)
	    _ax.xaxis.set_ticks_position("bottom")
	    _ax.yaxis.set_ticks_position("left")
	end
	ax_marg_x.tick_params(axis="x", labelbottom=false)
    ax_marg_y.tick_params(axis="y", labelleft=false)
	tight_layout()
	# joint plot
	T₀s = range(T₀_prior.a, T₀_prior.b, length=101)
	t₀s = range(-1.0, 1.0, length=100)
	ρs_post = zeros(length(t₀s), length(T₀s))
	ρs_prior = zeros(length(t₀s), length(T₀s))
	ρ_post = get_ρ_posterior_t₀_T₀()
	for (i, T₀) in enumerate(T₀s)
		for (j, t₀) in enumerate(t₀s)
			ρs_post[j, i] = ρ_post([T₀, t₀])
			ρs_prior[j, i] = pdf(t₀_prior, t₀) * pdf(T₀_prior, T₀)
		end
	end
	ax_joint.contour(T₀s, t₀s, ρs_prior, cmap=prior_colormap)
	ax_joint.plot(θ_0s, t_0s, color="black", linewidth=1, linestyle="dashed")
	ax_joint.contour(T₀s, t₀s, ρs_post, cmap=posterior_colormap)
	

	# ax_joint.plot(
	# 	[T₀_prior.a, T₀_prior.a, T₀_prior.b, T₀_prior.b, T₀_prior.a], 
	# 	[t₀_prior.a, t₀_prior.b, t₀_prior.b, t₀_prior.a, t₀_prior.a], 
	# 	color=the_colors["prior"])
	
	# ax_joint.hexbin(DataFrame(chain_T₀_t₀)[:, :T₀], DataFrame(chain_T₀_t₀)[:, :t₀],
	# 	mincnt=1, gridsize=15, cmap=posterior_colormap, bins=[range(0, 1, length=3), range(0, 1, length=3)]
	# )
	# ax_joint.fill_between([T₀_prior.a, T₀_prior.b], [t₀_prior.a, t₀_prior.a],
	# 	[t₀_prior.b, t₀_prior.b], alpha=0.1)
	# 	color=the_colors["prior"], zorder=0, alpha=alpha)

	# marginal prior and posterior, T₀
	T₀s = [T₀_prior.a, T₀_prior.b]
	T₀s = vcat(T₀s .- 0.000001, T₀s .+ 0.000001)
	sort!(T₀s)
	ρ_prior = [pdf(T₀_prior, T₀) for T₀ in T₀s]


	ax_marg_x.plot(T₀s, ρ_prior, 
		color=the_colors["prior"], zorder=1)
	ax_marg_x.set_yticks([0])
	ax_marg_x.set_ylim(ymin=0)
		
	ρ = get_kde_ρ(DataFrame(chain_T₀_t₀)[:, :T₀])
	T₀s = collect(range(T₀_prior.a, T₀_prior.b, length=100))
	ρ_posterior = ρ.(T₀s)
	pushfirst!(ρ_posterior, 0.0)
	pushfirst!(T₀s, T₀_prior.a)
	ax_marg_x.plot(T₀s, ρ_posterior, 
		color=the_colors["posterior"], zorder=2)

	ax_marg_x.set_ylim(0, maximum(ρ_posterior)*1.1)
	
	# marginal prior, t₀
	t₀s = range(-1.0, 1.0, length=150)
	# t₀s = [t₀_prior.a, t₀_prior.b]
	# t₀s = vcat(t₀s .- 0.000001, t₀s .+ 0.0000001)
	# sort!(t₀s)
	ρ_prior = [pdf(t₀_prior, t₀) for t₀ in t₀s]

	# ax_marg_y.fill_betweenx(t₀s, zeros(4), ρ_prior, 
	# 	color=the_colors["prior"], zorder=0, alpha=alpha)
	ax_marg_y.plot(ρ_prior, t₀s,
		color=the_colors["prior"], zorder=1)
	ax_marg_y.set_xticks([0])
	ax_marg_y.set_xlim(xmin=0)

	ρ = get_kde_ρ(DataFrame(chain_T₀_t₀)[:, :t₀])
	# t₀s = collect(range(t₀_prior.a, t₀_prior.b, length=100))
	ρ_posterior = ρ.(t₀s)
	# pushfirst!(ρ_posterior, 0.0)
	# pushfirst!(t₀s, t₀_prior.a)
	# push!(ρ_posterior, 0.0)
	# push!(t₀s, t₀_prior.b)
	ax_marg_y.plot(ρ_posterior, t₀s, 
		color=the_colors["posterior"], zorder=2)

	ax_marg_y.set_xlim(0, maximum(ρ_posterior)*1.1)
	
	ax_joint.scatter([data2[1, "T [°C]"]], [data2[1, "t [hr]"]], 		
			color=the_colors["data"], edgecolor="black", zorder=10000, label=L"$(t_0, \theta_0)$")
	# ax_joint.legend()
	ax_joint.set_xlabel(L"initial temperature, $\theta_0$ [°C]")
	ax_joint.set_ylabel(L"time taken out of fridge, $t_0$ [hr]")
	ax_joint.set_ylim([-0.55, 0.55])
	ax_joint.set_xlim([-0.5, 15.5])
	tight_layout()
	savefig("figs/time_reversal_II_i_obs$i_obs.pdf", format="pdf", bbox_inches="tight")
	fig
end

# ╔═╡ 2c4dd342-4f55-4ad4-9ce8-5825544fdb98
new_undetermined_viz()

# ╔═╡ 8ba02a50-98f8-4c83-9f4f-040a1aad8274
md"to check..."

# ╔═╡ f7af1845-cae4-4eae-ab99-140e145d9b39
begin
	fig = figure()
	jp = sns.jointplot(
		x=DataFrame(chain_T₀_t₀)[:, :T₀], 
		y=DataFrame(chain_T₀_t₀)[:, :t₀], kind="kde"
	)
	jp.ax_joint.set_xlabel("θ_0")
	jp.ax_joint.set_ylabel("t_0")
	jp.ax_joint.plot(θ_0s, t_0s, color="r")

	jp.fig
end

# ╔═╡ b31a6a61-8999-49de-b9b4-01d1f4f0d48a
fixed_params2.Tₐ

# ╔═╡ da2ab292-058f-44c1-a2bf-77f874815873
A = [1 0; 0 0]

# ╔═╡ 4523845d-818a-4e13-8dca-175de7da55d5
contour(A)

# ╔═╡ Cell order:
# ╟─b1c06c4d-9b4d-4af3-9e9b-3ba993ca83a0
# ╠═43bcf4b0-fbfc-11ec-0e23-bb05c02078c9
# ╠═1dea25e4-51ee-4f32-a97e-8ce316dfb371
# ╠═edb44636-d6d4-400f-adc4-75b287a1f993
# ╠═7831a816-e8d4-49c5-b209-078e74e83c5f
# ╠═a081eb2c-ff46-4efa-a6cd-ee3e9209e14e
# ╠═8931e445-6664-4609-bfa1-9e808fbe9c09
# ╟─3ae0b235-5ade-4c30-89ac-7f0480c0da11
# ╠═a13ba151-99c1-47ae-b96e-dc90464990b6
# ╠═ee7fd372-22b0-4bf5-a5e9-5e3a5b6e1843
# ╠═8ee1b06d-c255-4ae3-ac8b-06f7498dbf76
# ╟─b29797b9-7e2f-4d55-bc39-dba5ad7663de
# ╠═269ac9fa-13f3-443a-8669-e8f13d3518a6
# ╠═d32079ef-7ebd-4645-9789-1d258b13b66f
# ╠═b8a3fc88-6e4d-457d-8582-f6302fb206ac
# ╠═ce178132-a07d-4154-83b4-5f536c8f77aa
# ╠═7b8f64b9-9776-4385-a2f0-38f78d76ef79
# ╠═ecd4ea3f-1775-4c4e-a679-f8e15eaad3f7
# ╠═2e57666d-b3f4-451e-86fd-781217c1258d
# ╠═bb3ae6a9-5d87-4b90-978e-8674f6c5bd99
# ╠═f35c7dcd-243a-4a16-8f7d-424c583aa99f
# ╠═5478b192-677e-4296-8ce5-c6d0447898bc
# ╠═9e78c280-c19b-469b-8a2b-3c9f4b92a2e5
# ╠═44963969-6883-4c7f-a6ed-4c6eac003dfe
# ╠═ff7e4fd8-e34b-478e-ab8a-2f35aba99ba6
# ╠═a8257d2e-fca8-4bd9-8733-f4034836bbb9
# ╠═ff9735a0-1fea-4518-a06a-0af74687ba9c
# ╠═788f5c20-7ebb-43e7-bd07-46aa6c9fd249
# ╠═2378f74e-ccd6-41fd-89f5-6001b75ea741
# ╠═a1e622ae-7672-4ca2-bac2-7dcc0a500f1f
# ╠═294e240f-c146-4ef3-b172-26e70ad3ed19
# ╠═cd46a3c7-ae78-4f3c-8ba6-c4a55d598843
# ╠═b6b05d1b-5e2f-4082-a7ef-1211024c700b
# ╟─7a01dfaf-fae1-4a8c-a8a2-1ac973bf3197
# ╠═f20159ad-7f8b-484e-95ea-afdac97f876a
# ╠═f184e3ea-82f9-49f4-afb6-99c609d7936f
# ╟─d8e026b9-8943-437e-a08b-2395de35d705
# ╠═7df25291-a600-449e-a194-3ec7c3f11361
# ╠═8f145533-7208-4c25-9b1e-84370c7ac7ca
# ╠═0bff14a8-89eb-488c-88c6-e08a64e577ed
# ╟─ac6f1d8d-4402-4737-82f6-4fd098b93b5e
# ╠═4e68878f-c278-4218-8a52-ce86490981da
# ╠═d199b848-a86e-4d7c-bcd0-566f9d8ea052
# ╠═54efdfb6-bb64-4834-8cd9-a3f126f731e9
# ╠═8d358b8d-7432-421a-8661-4550c0457f97
# ╠═8dbbbe1c-4eb6-4ac2-a447-bbaa500e03b4
# ╠═a3ee46bf-9266-4025-8678-e535d0077faf
# ╠═62c5e645-285d-470e-b46b-00f0471b7329
# ╠═07b22d3a-d616-4c89-98c6-d7ee1cd314b6
# ╠═efdf4047-81ab-45db-9980-267df2bad314
# ╠═6e4c92c2-ab69-4ac7-9144-05cc3b8b0dd9
# ╠═3f954d0a-3f4e-43c9-b028-f2abdc83792a
# ╠═bd5602cd-8b6d-430f-a700-40b449d1da27
# ╠═ba77054e-1754-4c62-bce9-7e166bd99a6e
# ╠═e84e11c6-eba4-45de-82b7-d4f0c76e4c94
# ╠═8c8ce05d-45da-4a1a-bfce-457282e4237e
# ╠═3893d1d9-e98e-4aa1-8723-41e1c2b158fd
# ╟─1e5ba0b1-c129-410c-9048-89a75210fd40
# ╠═da778a83-aa3d-427f-9cd7-eede559c5c37
# ╠═8b1f8a44-612c-4032-93a7-7b0c21c47c31
# ╠═845bdbf7-f30e-4f0c-a8db-6f272e76eec9
# ╠═14bee7d1-dadc-41be-9ea0-1420cd68a121
# ╠═aaca06d8-0e20-4c53-9097-d69fe1ae3d83
# ╠═d812222a-3d59-418e-a67c-4154e0fd6e23
# ╠═7824672b-e69d-435d-a8ab-d62f014374d3
# ╠═b14d545e-bc9e-493b-877f-899ec4ddc8fc
# ╠═58a95e76-01db-48c4-981b-d212aff54029
# ╠═2c4dd342-4f55-4ad4-9ce8-5825544fdb98
# ╟─8ba02a50-98f8-4c83-9f4f-040a1aad8274
# ╠═f7af1845-cae4-4eae-ab99-140e145d9b39
# ╠═b31a6a61-8999-49de-b9b4-01d1f4f0d48a
# ╠═da2ab292-058f-44c1-a2bf-77f874815873
# ╠═4523845d-818a-4e13-8dca-175de7da55d5
