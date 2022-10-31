### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ 43bcf4b0-fbfc-11ec-0e23-bb05c02078c9
using DataFrames, Distributions, Turing, LinearAlgebra,Random, JLD2, PyPlot, ColorSchemes, StatsBase, Colors, PlutoUI, PyCall, Seaborn

# ╔═╡ b1c06c4d-9b4d-4af3-9e9b-3ba993ca83a0
md"# Bayesian statistical inversion"

# ╔═╡ edb44636-d6d4-400f-adc4-75b287a1f993
TableOfContents()

# ╔═╡ 7831a816-e8d4-49c5-b209-078e74e83c5f
isdir("figs") ?  nothing : mkdir("figs")

# ╔═╡ ae477150-45db-47ed-a6a8-018541cfe485
mpl_tk = pyimport("mpl_toolkits.axes_grid1.inset_locator")

# ╔═╡ 2b4ee7f8-0cc0-458a-bb54-03c119dd2944
sns = pyimport("seaborn")

# ╔═╡ ad610936-99a3-42a1-800d-94e66051f605
import ScikitLearn as skl

# ╔═╡ cbd53ba5-34f0-42fc-8ac1-386a72e23e13
skl.@sk_import neighbors: KernelDensity

# ╔═╡ a9db257b-f2a7-4076-aa31-24208a2bfca6
fm = PyPlot.matplotlib.font_manager.fontManager.addfont("FiraMath-Regular.ttf")

# ╔═╡ f10bcc6f-85c8-44d9-aa9c-2d37ab1fdafd
function wongcolors()
    return [
        (0/255, 114/255, 178/255), # blue
        (230/255, 159/255, 0/255), # orange
        (0/255, 158/255, 115/255), # green
        (204/255, 121/255, 167/255), # reddish purple
    	(86/255, 180/255, 233/255), # sky blue
        (213/255, 94/255, 0/255), # vermillion
        (240/255, 228/255, 66/255), # yellow
    ]
end

# ╔═╡ 220beb01-2da2-444a-be94-795398228bdf
[RGB(c...) for c in wongcolors()]

# ╔═╡ 2ccf2c0d-1f31-4ebc-9427-4c36f221f66e
begin
	# https://ardoi.github.io/post/nicer_mpl/
	rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	gray = "444444"
	rcParams["axes.facecolor"] = "white"# "f5f5f5"
	rcParams["axes.edgecolor"] = gray
	rcParams["axes.labelcolor"] = gray
	rcParams["text.color"] = gray
	rcParams["xtick.color"] = gray
	rcParams["mathtext.fontset"] = "custom"
	rcParams["ytick.color"] = gray
	
	# rcParams["font.name"] = "Fira Math"
	rcParams["font.family"] = "Fira Math"
	rcParams["font.size"] = 18
	rcParams["legend.frameon"] = true
	rcParams["lines.linewidth"] = 3
	rcParams["lines.markersize"] = 8
	rcParams["grid.linestyle"] = "-"
	rcParams["grid.color"] = "white"
	rcParams["figure.figsize"] =  [6.0 * 0.9, 4.8 * 0.9]
end

# ╔═╡ 3749245a-67b2-4015-8a58-1b89c8c3b328
function myfig(;figsize=nothing)
    fig = figure(figsize=isnothing(figsize) ? Tuple(rcParams["figure.figsize"]) : figsize)
    ax = gca()
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    return fig, ax
end

# ╔═╡ a081eb2c-ff46-4efa-a6cd-ee3e9209e14e
my_colors = wongcolors()# ColorSchemes.Set3_5 # sns.color_palette("Set3_5")# wongcolors()# sns.color_palette() 

# ╔═╡ 8931e445-6664-4609-bfa1-9e808fbe9c09
the_colors = Dict("air"        => my_colors[1], 
	              "data"       => my_colors[2],
	              "model"      => my_colors[3], 
	              "prior"      => my_colors[5],
	              "posterior"  => my_colors[end])

# ╔═╡ ddee1dcf-41cd-4836-bd87-af688a009464
# the_colors = Dict(key => (c.r, c.g, c.b) for (key, c) in _the_colors)

# ╔═╡ 3ae0b235-5ade-4c30-89ac-7f0480c0da11
md"## the forward model"

# ╔═╡ a13ba151-99c1-47ae-b96e-dc90464990b6
function T_model(t, τ, T₀, Tₐ, t₀=0.0)
    if t < 0.0
        return T₀
	end
    return Tₐ .+ (T₀ - Tₐ) * exp(-(t - t₀) / τ)
end

# ╔═╡ 16710341-5ea5-47e9-98b1-2e54ae552956
ts_model = range(0.0, 4.0, length=400)

# ╔═╡ 346d44e8-7b20-4cfa-8f22-99c4e844f56d
function viz_model_only()
	fig, ax = myfig()
	xlabel(L"time, $(t-t_0)/\lambda$")
	ylabel(L"lime temperature, $\theta(t)$")
	yticks([0, 1], [L"$\theta_0$", L"$\theta^{\rm{air}}$"])
	plot(ts_model, 1.0 .- exp.(-ts_model), c=the_colors["model"])
	axhline([1.0], linestyle="dashed", label=L"$\theta_\infty$", c=the_colors["air"])
	ylim(-0.1, 1.1)
	xlim(0, 4)
	
	# read image file
	arr_image = plt.imread("transparent_lime.png", format="png")
	inset = mpl_tk.inset_axes(ax, width="30%", height="30%", loc=4)
	inset.imshow(arr_image)
	inset.set_axis_off()
	
	tight_layout()
	savefig("figs/model_soln.pdf", format="pdf")
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
bw = Dict("τ" => 0.01, "T₀" => 0.5)

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
function get_kde_ρ(x::Vector{Float64}, bw::Float64)
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
		
		ρ = get_kde_ρ(c[:, var], bw[var])
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
	ρ = get_kde_ρ(x.samples, bw[var])
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
		inset.set_xticks([0, 5, 10, 15])
		inset.set_xlim(-1, 16)
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
T₀_prior = Uniform(0.0, 15.0)

# ╔═╡ 8dbbbe1c-4eb6-4ac2-a447-bbaa500e03b4
@model function likelihood_for_T₀(data, i_obs, Tₐ)
    # Prior distributions.
	T₀ ~ T₀_prior
	if data[i_obs, "T [°C]"] < 10.0
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

# ╔═╡ 62c5e645-285d-470e-b46b-00f0471b7329
i_obs = 30 # and try 35 and 30

# ╔═╡ 07b22d3a-d616-4c89-98c6-d7ee1cd314b6
data2[i_obs, :]

# ╔═╡ efdf4047-81ab-45db-9980-267df2bad314
model_T₀ = likelihood_for_T₀(data2, i_obs, fixed_params2.Tₐ)

# ╔═╡ 287fd4e2-3afd-4540-be15-f2a486e36e37
chain_T₀ = sample(model_T₀, NUTS(), MCMCSerial(), 2_500, 4; progress=true)

# ╔═╡ 3f954d0a-3f4e-43c9-b028-f2abdc83792a
viz_convergence(chain_T₀, "T₀")

# ╔═╡ bd5602cd-8b6d-430f-a700-40b449d1da27
viz_posterior_prior(chain_T₀, T₀_prior, "T₀", "time_reversal_prior_posterior_id_$i_obs.pdf", true_var=data2[1, "T [°C]"])

# ╔═╡ ba77054e-1754-4c62-bce9-7e166bd99a6e
viz_b4_after_inference(data2, fixed_params2, chain_T₀, i_obs=i_obs)

# ╔═╡ e84e11c6-eba4-45de-82b7-d4f0c76e4c94
gridspec = PyPlot.matplotlib.gridspec

# ╔═╡ 8c8ce05d-45da-4a1a-bfce-457282e4237e
# ╠═╡ disabled = true
#=╠═╡
function compare_prior_posterior_T₀(chain::Chains, prior::Distribution, fancy::Bool)
	T₀ = analyze_posterior(chain, :T₀)

	ylabels = ["prior\ndensity", "posterior\ndensity"]

	if fancy
		fig = figure(figsize=(7.0*0.9, 4.8*0.9))
		gs = fig.add_gridspec(2, hspace=-0.6)
		axs = gs.subplots(sharex=true, sharey=true)
		for i = 1:2
			rect = axs[i].patch
			rect.set_alpha(0)
		    for s in ["top","right","left","bottom"]
				if s == "bottom"
					continue
				end
		        axs[i].spines[s].set_visible(false)
			end
			axs[i].set_yticks([0])
			axs[i].text(-0.25, 0.075, ylabels[i], 
				transform=axs[i].transAxes)
		end
		axs[2].set_xlabel(L"initial temperature, $\theta_0$ [°C]")
	else
		fig, ax = myfig()
		xlabel(L"initial temperature, $\theta_0$ [°C]")
		axs = [ax, ax]
	end
	# axs[1].set_ylabel("prior\ndensity")
	# axs[2].set_ylabel("posterior\ndensity")

	# prior
	θ₀s = vcat(range(-1.0, 16.0, length=100), [-0.001, 0.001, 14.9999, 15.0001])
	sort!(θ₀s)
	ρ_prior = [pdf(T₀_prior, θ₀) for θ₀ in θ₀s]
	axs[1].plot(θ₀s, ρ_prior, color="black", linewidth=1)
	axs[1].fill_between(θ₀s, zeros(length(θ₀s)), ρ_prior, 
				color=the_colors["prior"], label="prior")
	# posterior
	axs[2].plot([T₀.lb, T₀.ub], zeros(2), color="black", linewidth=10, zorder=100)
	# the data
	axs[2].axvline([data2[1, "T [°C]"]], linestyle="dashed", 
		color=the_colors["data"], zorder=1000)
	# the distn
	θ₀s = range(0.0, 15.0, length=120)
	ρ = get_kde_ρ(T₀.samples, 0.1)
	axs[2].plot(θ₀s, ρ.(θ₀s), color=the_colors["posterior"], linewidth=3, zorder=101, clip_on=false)
	# axs[2].fill_between(θ₀s, zeros(length(θ₀s)), ρ.(θ₀s), 
	# 		color=the_colors["posterior"], label="posterior", zorder=100)
	# the ci
	axs[2].plot([T₀.lb, T₀.ub], [0, 0], c="black", linewidth=10, zorder=1000)
	println("ci = ", round.([T₀.lb, T₀.ub], digits=2))

	axs[1].set_ylim(ymin=0)
	axs[1].set_xlim([-0.5, 15.5])
	if ! fancy
		legend(fontsize=15)
		ylabel("density")
	end
	tight_layout()
	# savefig("posterior_tau.pdf", format="pdf")
	fig
end
  ╠═╡ =#

# ╔═╡ 1e5ba0b1-c129-410c-9048-89a75210fd40
md"## the ill-posed inverse problem"

# ╔═╡ da778a83-aa3d-427f-9cd7-eede559c5c37
t₀_prior = Uniform(-0.5, 0.5)

# ╔═╡ 8b1f8a44-612c-4032-93a7-7b0c21c47c31
@model function likelihood_for_T₀_t₀(data, i_obs, Tₐ)
    # Prior distributions.
	T₀ ~ T₀_prior
	if data[i_obs, "T [°C]"] < 10.0
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

	# joint
	T₀s = range(T₀_prior.a, T₀_prior.b, length=101)
	t₀s = range(t₀_prior.a, t₀_prior.b, length=100)
	ρs_post = zeros(length(t₀s), length(T₀s))
	ρ_post = get_ρ_posterior_t₀_T₀()
	for (i, T₀) in enumerate(T₀s)
		for (j, t₀) in enumerate(t₀s)
			ρs_post[j, i] = ρ_post([T₀, t₀])
		end
	end
	ax_joint.contour(T₀s, t₀s, ρs_post, cmap=posterior_colormap)

	ax_joint.plot(
		[T₀_prior.a, T₀_prior.a, T₀_prior.b, T₀_prior.b, T₀_prior.a], 
		[t₀_prior.a, t₀_prior.b, t₀_prior.b, t₀_prior.a, t₀_prior.a], 
		color=the_colors["prior"])
	
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
		
	ρ = get_kde_ρ(DataFrame(chain_T₀_t₀)[:, :T₀], bw["T₀"])
	T₀s = collect(range(T₀_prior.a, T₀_prior.b, length=100))
	ρ_posterior = ρ.(T₀s)
	pushfirst!(ρ_posterior, 0.0)
	pushfirst!(T₀s, T₀_prior.a)
	ax_marg_x.plot(T₀s, ρ_posterior, 
		color=the_colors["posterior"], zorder=2)

	ax_marg_x.set_ylim(0, maximum(ρ_posterior)*1.1)
	
	# marginal prior, t₀
	t₀s = [t₀_prior.a, t₀_prior.b]
	t₀s = vcat(t₀s .- 0.000001, t₀s .+ 0.0000001)
	sort!(t₀s)
	ρ_prior = [pdf(t₀_prior, t₀) for t₀ in t₀s]

	# ax_marg_y.fill_betweenx(t₀s, zeros(4), ρ_prior, 
	# 	color=the_colors["prior"], zorder=0, alpha=alpha)
	ax_marg_y.plot(ρ_prior, t₀s,
		color=the_colors["prior"], zorder=1)
	ax_marg_y.set_xticks([0])
	ax_marg_y.set_xlim(xmin=0)

	ρ = get_kde_ρ(DataFrame(chain_T₀_t₀)[:, :t₀], 0.05)
	t₀s = collect(range(t₀_prior.a, t₀_prior.b, length=100))
	ρ_posterior = ρ.(t₀s)
	pushfirst!(ρ_posterior, 0.0)
	pushfirst!(t₀s, t₀_prior.a)
	push!(ρ_posterior, 0.0)
	push!(t₀s, t₀_prior.b)
	ax_marg_y.plot(ρ_posterior, t₀s, 
		color=the_colors["posterior"], zorder=2)

	ax_marg_y.set_xlim(0, maximum(ρ_posterior)*1.1)
	
	ax_joint.scatter([data2[1, "T [°C]"]], [data2[1, "t [hr]"]], 		
			color=the_colors["data"], edgecolor="black", zorder=10000)
	ax_joint.set_xlabel(L"initial temperature, $\theta_0$ [°C]")
	ax_joint.set_ylabel(L"time taken out of fridge, $t_0$ [hr]")
	ax_joint.set_ylim([-0.55, 0.55])
	ax_joint.set_xlim([-0.5, 15.5])
	tight_layout()
	savefig("figs/time_reversal_II.pdf", format="pdf")
	fig
end

# ╔═╡ 2c4dd342-4f55-4ad4-9ce8-5825544fdb98
new_undetermined_viz()

# ╔═╡ f7af1845-cae4-4eae-ab99-140e145d9b39
begin
	fig = figure()
	jp = sns.jointplot(
		x=DataFrame(chain_T₀_t₀)[:, :T₀], 
		y=DataFrame(chain_T₀_t₀)[:, :t₀], kind="kde"
	)
	jp.fig
end

# ╔═╡ da2ab292-058f-44c1-a2bf-77f874815873
A = [1 0; 0 0]

# ╔═╡ 4523845d-818a-4e13-8dca-175de7da55d5
contour(A)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ScikitLearn = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
Seaborn = "d2ef9438-c967-53ab-8060-373fdd9e13eb"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
ColorSchemes = "~3.19.0"
Colors = "~0.12.8"
DataFrames = "~1.3.6"
Distributions = "~0.25.76"
JLD2 = "~0.4.25"
PlutoUI = "~0.7.44"
PyCall = "~1.94.1"
PyPlot = "~2.11.0"
ScikitLearn = "~0.6.4"
Seaborn = "~1.1.1"
StatsBase = "~0.33.21"
Turing = "~0.21.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "f0852b9ff9e16f585f8edc6da01857498ed6eefd"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "5c26c7759412ffcaf0dd6e3172e55d783dd7610b"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.1.3"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "DensityInterface", "Setfield", "SparseArrays"]
git-tree-sha1 = "6320752437e9fbf49639a410017d862ad64415a5"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.5.2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "5c0b629df8a5566a06f5fef5100b53ea56e465a0"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "0091e2e4d0a7125da0e3ad8c7dbff9171a921461"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.6"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "d7a7dabeaef34e5106cdf6c2ac956e9e3f97f666"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.8"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "9ff1247be1e2aa2e740e84e8c18652bd9d55df22"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.3.8"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "67fcc7d46c26250e89fc62798fbe07b5ee264c6f"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.6"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e9f7992287edfc27b3cbe0046c544bace004ca5b"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.22"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "93c8ba53d8d26e124a5a8d4ec914c3a16e6a0970"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.3"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "7fe6d92c4f281cf4ca6f2fba0ce7b299742da7ca"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.37"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "a3704b8e5170f9339dff4e6cb286ad49464d3646"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.10.6"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "d7d816527558cb8373e8f7a746d88eb8a167b023"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.44.7"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "332a332c97c7071600984b3c31d9067e1a4e6e25"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "78bee250c6826e1cf805a88b7f1e86025275d208"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "46d2680e618f8abd007bce0c3026cb0c4a8f2032"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.12.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "992a23afdb109d0d2f8802a30cf5ae4b1fe7ea68"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "04db820ebcfc1e053bd8cbb8d8bccf0ff3ead3f7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.76"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "0c139e48a8cea06c6ecbbec19d3ebc5dcbd7870d"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.43"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "MacroTools", "OrderedCollections", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "7bc3920ba1e577ad3d7ebac75602ab42b557e28e"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.20.2"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterfaceCore", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "4cda4527e990c0cc201286e0a0bfbbce00abcfc2"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "1.0.0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "802bfc139833d2ba893dd9e62ba1767c88d708ae"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.5"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "a5e6e7f12607e90d71b09e6ce2c965e41b337968"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.1"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "a2657dd0f3e8a61dbe70fc7c122038bd33790af5"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.3.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "842dd89a6cb75e02e85fdd75c760cdc43f5d6863"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.6"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "3f91cd3f56ea48d4d2a75c2a65455c5fc74fa347"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "1c3ff7416cb727ebf4bab0491a56a296d7b8cf1d"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.25"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LRUCache]]
git-tree-sha1 = "d64a0aff6691612ab9fb0117b0995270871c5dfc"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.3.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "dfa6c5f2d5a8918dd97c7f1a9ea0de68c2365426"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.7.5"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random", "Requires", "UnPack"]
git-tree-sha1 = "408a29d70f8032b50b22155e6d7776715144b761"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "1.0.2"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "f5f347b828fd95ece7398f412c81569789361697"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.5.0"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "59ac3cc5c08023f58b9cd6a5c447c4407cede6bc"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "0a36882e73833d60dac49b00d203f73acfd50b85"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.7.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "4d5917a26ca33c66c8e5ca3247bd163624d35493"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "415108fd88d6f55cedf7ee940c7d4b01fad85421"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.9"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "8a9102cb805df46fc3d6effdc2917f09b0215c0b"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.10"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Pandas]]
deps = ["Compat", "DataValues", "Dates", "IteratorInterfaceExtensions", "Lazy", "OrderedCollections", "Pkg", "PyCall", "Statistics", "TableTraits", "TableTraitsUtils", "Tables"]
git-tree-sha1 = "0ccb570180314e4dfa3ad81e49a3df97e1913dc2"
uuid = "eadc2687-ae89-51f9-a5d9-86b5a6373a9c"
version = "1.6.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "6c01a9b494f6d2a9fc180a08b182fcb06f0958a0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "6e33d318cf8843dade925e35162992145b4eb12f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.44"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "53b8b07b721b77144a0fbbbc2675222ebf40a02d"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.94.1"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "f9d953684d4d21e947cb6d642db18853d43cb027"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "3c009334f45dfd546a16a57960a821a1a023d241"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.5.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "612a4d76ad98e9722c8ba387614539155a59e30c"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.0"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "Tables", "ZygoteRules"]
git-tree-sha1 = "3004608dc42101a944e44c1c68b599fa7c669080"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.32.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "a3db467ce768343235032a1ca0830fc64158dadf"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.8"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "cdc1e4278e91a6ad530770ebb327f9ed83cf10c4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.3"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Preferences", "RecipesBase", "RecursiveArrayTools", "RuntimeGeneratedFunctions", "StaticArraysCore", "Statistics", "Tables"]
git-tree-sha1 = "e078c600cb15f9ad1a21cd58fc1c01a29aecb908"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.62.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.ScikitLearn]]
deps = ["Compat", "Conda", "DataFrames", "Distributed", "IterTools", "LinearAlgebra", "MacroTools", "Parameters", "Printf", "PyCall", "Random", "ScikitLearnBase", "SparseArrays", "StatsBase", "VersionParsing"]
git-tree-sha1 = "ccb822ff4222fcf6ff43bbdbd7b80332690f168e"
uuid = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
version = "0.6.4"

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[deps.Seaborn]]
deps = ["Pandas", "PyCall", "PyPlot", "Reexport", "Test"]
git-tree-sha1 = "c7d0011bfb487a40501ad9383e24f1908809e1ed"
uuid = "d2ef9438-c967-53ab-8060-373fdd9e13eb"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "f86b3a049e5d05227b10e15dbb315c5b90f14988"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.9"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f53e34e784ae771eb9ccde4d72e578aa453d0554"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.6"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "d963aad627fd7af56fbbfee67703c2f7bfee9dd7"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.22"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "77fea79baa5b22aeda896a8d9c6445a74500a2c2"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.74"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "68fb67dab0c11de2bb1d761d7a742b965a9bc875"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.21.12"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─b1c06c4d-9b4d-4af3-9e9b-3ba993ca83a0
# ╠═43bcf4b0-fbfc-11ec-0e23-bb05c02078c9
# ╠═edb44636-d6d4-400f-adc4-75b287a1f993
# ╠═7831a816-e8d4-49c5-b209-078e74e83c5f
# ╠═ae477150-45db-47ed-a6a8-018541cfe485
# ╠═2b4ee7f8-0cc0-458a-bb54-03c119dd2944
# ╠═ad610936-99a3-42a1-800d-94e66051f605
# ╠═cbd53ba5-34f0-42fc-8ac1-386a72e23e13
# ╠═a9db257b-f2a7-4076-aa31-24208a2bfca6
# ╠═f10bcc6f-85c8-44d9-aa9c-2d37ab1fdafd
# ╠═220beb01-2da2-444a-be94-795398228bdf
# ╠═2ccf2c0d-1f31-4ebc-9427-4c36f221f66e
# ╠═3749245a-67b2-4015-8a58-1b89c8c3b328
# ╠═a081eb2c-ff46-4efa-a6cd-ee3e9209e14e
# ╠═8931e445-6664-4609-bfa1-9e808fbe9c09
# ╠═ddee1dcf-41cd-4836-bd87-af688a009464
# ╟─3ae0b235-5ade-4c30-89ac-7f0480c0da11
# ╠═a13ba151-99c1-47ae-b96e-dc90464990b6
# ╠═16710341-5ea5-47e9-98b1-2e54ae552956
# ╠═346d44e8-7b20-4cfa-8f22-99c4e844f56d
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
# ╠═62c5e645-285d-470e-b46b-00f0471b7329
# ╠═07b22d3a-d616-4c89-98c6-d7ee1cd314b6
# ╠═efdf4047-81ab-45db-9980-267df2bad314
# ╠═287fd4e2-3afd-4540-be15-f2a486e36e37
# ╠═3f954d0a-3f4e-43c9-b028-f2abdc83792a
# ╠═bd5602cd-8b6d-430f-a700-40b449d1da27
# ╠═ba77054e-1754-4c62-bce9-7e166bd99a6e
# ╠═e84e11c6-eba4-45de-82b7-d4f0c76e4c94
# ╠═8c8ce05d-45da-4a1a-bfce-457282e4237e
# ╟─1e5ba0b1-c129-410c-9048-89a75210fd40
# ╠═da778a83-aa3d-427f-9cd7-eede559c5c37
# ╠═8b1f8a44-612c-4032-93a7-7b0c21c47c31
# ╠═845bdbf7-f30e-4f0c-a8db-6f272e76eec9
# ╠═14bee7d1-dadc-41be-9ea0-1420cd68a121
# ╠═aaca06d8-0e20-4c53-9097-d69fe1ae3d83
# ╠═7824672b-e69d-435d-a8ab-d62f014374d3
# ╠═58a95e76-01db-48c4-981b-d212aff54029
# ╠═2c4dd342-4f55-4ad4-9ce8-5825544fdb98
# ╠═f7af1845-cae4-4eae-ab99-140e145d9b39
# ╠═da2ab292-058f-44c1-a2bf-77f874815873
# ╠═4523845d-818a-4e13-8dca-175de7da55d5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
