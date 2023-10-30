### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 94182d54-3428-11ec-3014-d9cedb79f68a
begin 
	import Pkg; Pkg.activate() 
	using Roots, CairoMakie, ColorSchemes, Roots, Optim, Printf, PlutoUI, Random, Colors, JLD2, DataFrames
end

# ╔═╡ 028c2d73-a4bb-46f2-9b5e-22273c2ca5ce
TableOfContents()

# ╔═╡ 4ce9857b-9723-4f5a-b348-10d7631c546c
import AlgebraOfGraphics

# ╔═╡ f2e95386-e5bc-4b90-9c32-43808a8ec8f6
begin
	AlgebraOfGraphics.set_aog_theme!(
		fonts=[AlgebraOfGraphics.firasans("Light"), AlgebraOfGraphics.firasans("Light")])
	update_theme!(
		fontsize=20, 
		linewidth=4,
		markersize=14,
		titlefont=AlgebraOfGraphics.firasans("Light"),
		resolution=(0.8*500, 0.8*380)
	)
end

# ╔═╡ 5a3522aa-4bd9-498b-bbf8-9e49523f22de
function draw_axes!(ax)
	vlines!(ax, [0.0], color="gray", linewidth=1)
	hlines!(ax, [0.0], color="gray", linewidth=1)
end

# ╔═╡ 7d8291bd-59d2-4b98-8dbb-424702bc7412
my_colors = AlgebraOfGraphics.wongcolors()

# ╔═╡ 4b58f77f-b155-416f-8907-46b22d809c72
colors = Dict(zip(["data", "model", "air"], my_colors[[2, 3 , 1]]))

# ╔═╡ 3d8112be-ace6-4174-b31b-9896c556a019
function viz_air_temp(ax, Tₐ::Float64)
	hlines!(ax, Tₐ, 
		linestyle=:dash, label="θᵃⁱʳ", 
		linewidth=3, color=colors["air"])
end

# ╔═╡ eaf92206-2a16-4b54-aa8d-a3c27fbe03e0
md"# read data"

# ╔═╡ d5e96b70-0647-4979-9f61-6e9e4c906106
data = load("data_run_11.jld2")["data"]

# ╔═╡ 081a1336-eb60-4198-9f44-a33973486460
Tₐ = load("data_run_11.jld2")["θᵃⁱʳ"]

# ╔═╡ 6348b57f-6ad4-4dd5-b964-1f88d7c5860e
T₀ = data[1, "θ [°C]"]

# ╔═╡ 70721425-1988-48d9-b226-b5b36bc525a8
begin
	my_colormap = ColorSchemes.haline
	T_to_color(T) = get(my_colormap, T, (0.0, 20.0))
end

# ╔═╡ 5e4c2385-1171-4414-bc3c-a83bfff6abee
begin
	t = collect(range(-10.0/60, 7.5, length=150))
	push!(t, -0.001)
	push!(t, 0.001)
	sort!(t)
	
	the_ylims = (-0.5, 20.5)
	the_xlabel = "time, t [min]"
	the_ylabel = "temperature [°C]"
end

# ╔═╡ b5b9d1ad-92f1-43aa-8f08-49c5d09d3e71
begin
	function T(t::Number, params::NamedTuple)
		if t < params.t₀
			return params.T₀
		else
			return params.T₀ + (Tₐ - params.T₀) * (1 - exp(-(t - params.t₀) / params.τ))
		end
	end
	
	T(t::Array{Float64, 1}, params::NamedTuple) = [T(tᵢ, params) for tᵢ in t]
end

# ╔═╡ e9212acc-7874-44ba-9921-73780c2c8e03
function viz_model()
	fig = Figure()
	ax  = Axis(fig[1, 1], 
		       xlabel="t / τ", 
		       ylabel="θ(t)", 
		       yticks=([0, 1], ["θ₀", "θᵃⁱʳ"])
	)
	ts_model = range(0.0, 4.0, length=200)
	lines!(ts_model, 1.0 .- exp.(-ts_model), color=colors["model"])
	hlines!(ax, 1.0, style=:dash, 
			linestyle=:dot, label="θᵃⁱʳ", color=colors["air"])
	ylims!(-0.1, 1.1)
	xlims!(0, 4)
	save("model_soln.pdf", fig)
	return fig
end

# ╔═╡ 299faaae-4817-467c-b9b0-9147663dbc17
viz_model()

# ╔═╡ 331319e3-4d5f-4980-90e7-b13bf60c5867
δ = 0.75

# ╔═╡ bd89a236-15a9-45f0-85c9-e63b64c70728
md"# param identification"

# ╔═╡ a606ff46-73ac-44e0-95fe-b7b8cc6d0273
function cost(τ::Float64, data::DataFrame)
	params = (; τ=τ, t₀=0, T₀=data[1, "θ [°C]"])
    ℓ = 0.0
    for row in eachrow(data)
		tᵢ = row["t [hr]"]
		Tᵢ = row["θ [°C]"]
		
        ℓ += (Tᵢ - T(tᵢ, params)) ^ 2
    end
    return ℓ / nrow(data)
end

# ╔═╡ 3cd1926a-2a6d-43b9-8dcc-fa1605b69b68
opt_res = optimize(τ -> cost(τ, data), 0.0, 200.0)

# ╔═╡ 9cbbe92a-c751-4d43-8672-98668fd46719
τ_opt = opt_res.minimizer

# ╔═╡ d683fdfb-b8ed-4729-a4e9-7c8c6ea77e3e
function viz_loss(;τ_opt::Union{Float64, Nothing}=nothing)
	τs = range(0.0, 2.0, length=100)
	
	fig = Figure()
	ax  = Axis(fig[1, 1], 
			   xlabel="time constant, λ [min]", 
			   ylabel="loss, ℓ(τ) [(°C)²]")
	lines!(τs, [cost(τ, data) for τ in τs])
	xlims!(0.0, 2)
	if ! isnothing(τ_opt)
		vlines!(ax, τ_opt, linestyle=:dash, linewidth=1)
	end
	save("loss.pdf", fig)
	fig
end

# ╔═╡ 9f44e375-2369-43d7-a562-497c76b9cb1a
viz_loss(τ_opt=τ_opt)

# ╔═╡ f091f0f1-aee9-403f-90f9-e58556f6c8dc
md"# the forward problem"

# ╔═╡ 6fec04ae-64bb-4e7f-b82e-96579368f9c5
data2 = load("data_run_12.jld2")["data"]

# ╔═╡ 51f5aa61-207b-4033-8b31-6bf6676254e0
Tₐ2 = load("data_run_12.jld2")["θᵃⁱʳ"]

# ╔═╡ fd0ac159-1328-40ac-94cd-fdd4cc83f031
T₀2 = data[1, "θ [°C]"]

# ╔═╡ bfd12fda-7dbb-4a9d-80ec-a506bb908336
function viz_param_id(savename::String; plot_trajectory::Bool=true, 
                     viz_errors::Bool=false, replicate::Bool=false)
	params = (; T₀=T₀, Tₐ=Tₐ, t₀=0.0, τ=τ_opt)
	if replicate
		params = (; T₀=T₀2, Tₐ=Tₐ2, t₀=0.0, τ=τ_opt)
	end
	
	fig = Figure()
	ax  = Axis(fig[1, 1], 
			   xlabel=the_xlabel, 
			   ylabel=the_ylabel)
	draw_axes!(ax)
	viz_air_temp(ax, Tₐ)
	if replicate
		scatter!(data2[:, "t [hr]"], data2[:, "θ [°C]"], 
			color=colors["data"], strokewidth=1, label="{(tᵢ, θᵢ)}")
	else
		scatter!(data[:, "t [hr]"], data[:, "θ [°C]"], 
		color=colors["data"], strokewidth=1, label="{(tᵢ, θᵢ)}")
	end
	if plot_trajectory
		lines!(t, T(t, params), color=colors["model"])
	end
	if viz_errors
		errorbars!(data[:, "t [hr]"], data[:, "θ [°C]"], δ, δ, whiskerwidth=10, linewidth=3)
		if plot_trajectory
			lo_params = (; T₀=T₀, Tₐ=Tₐ, t₀=0.0, τ=τ_ci[2])
			hi_params = (; T₀=T₀, Tₐ=Tₐ, t₀=0.0, τ=τ_ci[1])

			band!(t, T(t, hi_params), T(t, lo_params), color=(colors["model"], 0.2))
		end
	end

	vlines!(ax, [0.0], color="gray", linewidth=1)
	xlims!(minimum(t), maximum(t))
	if plot_trajectory
		if ! viz_errors
			axislegend(@sprintf("λ = %.1f min", τ_opt), position=:rb)
		else
			# axislegend(@sprintf("τ ∈ [%.1f, %.1f] min", τ_ci[1], τ_ci[2]), position=:rb)
		end
	else
		axislegend(position=:rb)
	end
	ylims!(the_ylims...)
	save(savename, fig)
	fig
end

# ╔═╡ d76da2e4-dd8c-46d8-b084-88216b00353c
viz_param_id("id_tau_setup.pdf", plot_trajectory=false)

# ╔═╡ fa0bdadb-48a8-47c7-bc7c-e8be316228ce
viz_param_id("id_tau_soln.pdf", plot_trajectory=true)

# ╔═╡ 65e6a026-34ad-4df0-8176-00f713818ad0
viz_param_id("replicate_fit.pdf", plot_trajectory=true, replicate=true)

# ╔═╡ f07f7b5c-7723-4030-bf88-f060c8466a9c
function viz_forward_problem(savename::String; 
				             viz_error::Bool=true, plot_trajectory::Bool=true)
	params = (; T₀=T₀2, Tₐ=Tₐ2, t₀=0.0, τ=τ_opt)

	fig = Figure()
	ax  = Axis(fig[1, 1], 
		       xlabel=the_xlabel, 
		       ylabel=the_ylabel)
	draw_axes!(ax)
	
	if viz_error && plot_trajectory
		lo_params = (; T₀=T₀ + δ, Tₐ=Tₐ, t₀=0.0, τ=τ_opt)
		hi_params = (; T₀=T₀ - δ, Tₐ=Tₐ, t₀=0.0, τ=τ_opt)
		T_top = T(t, hi_params)
		T_bot = T(t, lo_params)
		
		band!(t, T_bot, T_top, color=(colors["model"], 0.2))
		
		lines!(t, T_top, linewidth=1, color=colors["model"])
		lines!(t, T_bot, linewidth=1, color=colors["model"])
	end
	
	viz_air_temp(ax, Tₐ)
	
	if plot_trajectory
		lines!(t, T(t, params), color=colors["model"])
	end

	if viz_error
		errorbars!([0.0], [params.T₀], δ, δ, whiskerwidth=10, linewidth=3)
	end
	
	scatter!([0.0], [params.T₀], 
		linestyle=:dash, label="(t₀, θ₀)", 
		strokewidth=2, color=colors["data"])
	xlims!(minimum(t), maximum(t))
	axislegend(@sprintf("λ = %.1f min", τ_opt), position=:rb)
	ylims!(the_ylims...)
	save(savename, fig)
	return fig
end

# ╔═╡ cf4bada1-6b5d-40ab-909a-02ee580c5f40
viz_forward_problem("forward_setup.pdf", viz_error=false, plot_trajectory=false)

# ╔═╡ d0b1d7d3-9f85-43e3-af8b-17d73c72777e
viz_forward_problem("forward_soln.pdf", viz_error=false, plot_trajectory=true)

# ╔═╡ 80b088dc-be57-46a5-8d18-5ed573837058
viz_forward_problem("forward_setup_error.pdf", viz_error=true, plot_trajectory=false)

# ╔═╡ a9f095c4-50a1-41f7-9f66-01e11914f1c0
viz_forward_problem("forward_soln_error.pdf", viz_error=true, plot_trajectory=true)

# ╔═╡ ee98f55a-6196-4b18-87d6-4fb00996e0be
md"# inverse problems"

# ╔═╡ e16652ba-d11e-4f87-b3d6-5f019e8c2b5d
viz_param_id("id_tau_setup_errors.pdf", plot_trajectory=false, viz_errors=true)

# ╔═╡ ca3b6534-9099-4ed8-8627-b85527307f5a
viz_param_id("id_tau_soln_errors.pdf", plot_trajectory=true, viz_errors=true)

# ╔═╡ bb2fa8fb-32be-47a2-a659-7c3fa209d068
md"## time reversal"

# ╔═╡ 4e3e686a-02e6-4fc7-89a7-3aebfdb6b351
function viz_changing_T₀()
	fig = Figure()
	ax  = Axis(fig[1, 1], 
			   xlabel=the_xlabel, 
			   ylabel=the_ylabel)
	T₀_range = range(0.0, 20.0, length=10)
	for T₀ in T₀_range
		params = (; T₀=T₀, Tₐ=Tₐ, t₀=0.0, τ=τ_opt)
		lines!(t, T(t, params), 
			   color=T_to_color(T₀)
		)
	end
	Colorbar(fig[1,2], limits=(0, 20), 
		colormap=my_colormap, label="initial temperature, θ₀ [°C]")
	text!(@sprintf("τ = %.1f min\nθᵃⁱʳ = %.1f °C", τ_opt, Tₐ), position=(200, 3), textsize=16, font=AlgebraOfGraphics.firasans("light"))
	ylims!(the_ylims...)
	xlims!(minimum(t), maximum(t))
	# axislegend(position=:rc)
	save("range_of_initial_conditions.pdf", fig)
	return fig
end

# ╔═╡ 28a37905-7196-4779-8eb9-b06563b090e5
viz_changing_T₀()

# ╔═╡ 82b478ac-6156-48ca-8f59-206388e255a7
function find_T₀(t′, T′, t₀=0.0)
	function g(T₀)
		params = (; T₀=T₀, Tₐ=Tₐ, t₀=t₀, τ=τ_opt)
		return T′ - T(t′, params)
	end
	return find_zero(g, (0.0, 10.0), Bisection())
end

# ╔═╡ 196eb94f-28d1-4e75-8670-8b87e36f0048
function find_t₀(t′, T′)
	function g(t₀)
		params = (; T₀=T₀, Tₐ=Tₐ, t₀=t₀, τ=τ_opt)
		return T′ - T(t′, params)
	end
	return find_zero(g, (2, 10), Bisection())
end

# ╔═╡ 9e0255b0-13d0-4fd5-8aa0-19e04117545d
function viz_time_reversal(savename::String, t′::Float64; viz_soln::Bool=true, viz_error::Bool=false)
	# data is (t′, T′).
	params = (; T₀=T₀, Tₐ=Tₐ, t₀=0.0, τ=τ_opt)
	T′ = T(t′, params)
	
	t_filtered = filter(x -> x <= t′, t)

	# solution to inverse problem
	T₀_ci  = [find_T₀(t′, T′ - δ), find_T₀(t′, T′ + δ)]
	params_lo = (; T₀=T₀_ci[1], Tₐ=Tₐ, t₀=0.0, τ=τ_opt)
	params_hi = (; T₀=T₀_ci[2], Tₐ=Tₐ, t₀=0.0, τ=τ_opt)
	T_lo = T(t_filtered, params_lo)
	T_hi = T(t_filtered, params_hi)
	
	fig = Figure()
	ax  = Axis(fig[1, 1], 
			   xlabel=the_xlabel, 
			   ylabel=the_ylabel,
		       # xticks=([0, 10, 20, 30, t′], ["0", "10", "20", "30", "t′"])
	)
	draw_axes!(ax)
	viz_air_temp(ax, Tₐ)
	
	if viz_soln
		lines!(t_filtered, T(t_filtered, params), color=colors["model"])
	end

	if viz_soln & viz_error
		band!(t_filtered, T_lo, T_hi, 
			  color=(colors["model"], 0.2))
		lines!(t_filtered, T_lo, linewidth=1, color=colors["model"])
		lines!(t_filtered, T_hi, linewidth=1, color=colors["model"])
	end
	# hlines!(ax5, Tₐ, 
	# 	linestyle=:dash, label="air temperature, Tₐ", 
	# 	linewidth=3, color=my_colors[3])
	if viz_error
		errorbars!([t′], [T′], δ, δ, whiskerwidth=10, linewidth=3)
	end
	scatter!([t′], [T′], 
		color=colors["data"], strokewidth=2,
		label="(t′, θ(t′))"
	)
	if viz_soln
		scatter!([0.0], [T₀], 
			label="(t₀, θ₀)", 
			strokewidth=2, color=colors["model"])
	end

	ylims!(the_ylims...)
	xlims!(minimum(t), maximum(t))
	axislegend(@sprintf("λ = %.1f min", τ_opt), position=:rb)
	save(savename, fig)
	return fig
end

# ╔═╡ 2482c26b-f570-4b04-8fe5-ac75a82d5660
viz_time_reversal("time_reversal_setup.pdf", 1.4, viz_soln=false)

# ╔═╡ eaf6b0f5-ccb0-4e5c-bae1-cc82e34aee85
viz_time_reversal("time_reversal_soln.pdf", 1.4, viz_soln=true)

# ╔═╡ 17b6d817-66ae-48e9-a2f1-847be0e6b02d
viz_time_reversal("time_reversal_setup_errors.pdf", 120.0, viz_soln=false, viz_error=true)

# ╔═╡ d54e803f-621c-48d3-b50b-aadae0d71296
viz_time_reversal("time_reversal_soln_errors.pdf", 120.0, viz_soln=true, viz_error=true)

# ╔═╡ a6574e8c-4ed7-4ec0-9b21-6f38c0141580
viz_time_reversal("time_reversal_setup_errors_long.pdf", 120.0, viz_soln=false, viz_error=true)

# ╔═╡ cdb664e8-cf11-42e5-bd99-bd333f1e5d3a
viz_time_reversal("time_reversal_soln_errors_long.pdf", 120.0, viz_soln=true, viz_error=true)

# ╔═╡ 95f4a495-b3f5-476a-bccb-9ebe2141dac3
md"## infinite solutions"

# ╔═╡ d2d0d681-0323-48fb-b521-c6fe4e006357
function ic_params_list(t′::Float64; n::Int=10, t_start=0.0)
	params = (; T₀=T₀, Tₐ=Tₐ, t₀=0.0, τ=τ_opt)
	T′ = T(t′, params)

	t₀s = range(t_start, t′, length=n)[1:end-1]
	param_list = []
	for t₀ in t₀s
		T₀ = find_T₀(t′, T′, t₀)
		params = (; T₀=T₀, Tₐ=Tₐ, t₀=t₀, τ=τ_opt)
		push!(param_list, params)
	end
	return param_list
end

# ╔═╡ f7a341b1-8d6f-4263-b2a7-73ea0df11722
function viz_infinite_soln(t′::Float64, savename::String; viz_solns::Bool=false)
	ic_params = ic_params_list(t′)
	t_filtered = filter(x -> x <= t′, t)
	params = (; T₀=T₀, Tₐ=Tₐ, t₀=0.0, τ=τ_opt)
	T′ = T(t′, params)
	
	fig = Figure()
	ax  = Axis(fig[1, 1], 
			   xlabel=the_xlabel, 
			   ylabel=the_ylabel)
	draw_axes!(ax)
	viz_air_temp(ax, Tₐ)
	if viz_solns
		for (i, params) in enumerate(ic_params)
			color = get(ColorSchemes.viridis, params.t₀, (0.0, t′))
			lines!(t_filtered, T(t_filtered, params), 
				   color=color
			)
			scatter!([params.t₀], [params.T₀], 
				# label= (i==length(ic_params)) ? "initial condition, (t₀, T₀)" : nothing,
				marker=:rect,
				strokewidth=2, color=color
			)
		end
	end
	scatter!([t′], [T′],
		color=colors["data"], strokewidth=1,
		label="(t′, θ(t′))"
	)

	ylims!(the_ylims...)
	xlims!(minimum(t), maximum(t))
	if viz_solns
		scatter!([NaN], [NaN],
			color=:white, strokewidth=1,marker=:rect,
			label="(t₀, θ₀)"
		)
	
		Colorbar(fig[1,2], limits=(0, t′), 
			colormap=ColorSchemes.viridis, label="initial time, t₀ [min]")
	end
	axislegend(@sprintf("τ = %.1f min\n(t₀, θ₀) = ?", τ_opt), position=:rb)
	save(savename, fig)
	fig
end

# ╔═╡ 0f23ddd3-115c-47b6-8fe1-44ba5c117a35
viz_infinite_soln(120.0, "infinite_soln.pdf", viz_solns=false)

# ╔═╡ 5da6eb96-85d7-42fc-9962-e15245838282
viz_infinite_soln(120.0, "infinite_soln_trajectory.pdf", viz_solns=true)

# ╔═╡ 1ae1eab3-f0f1-453d-8bd7-e269db431d7d
function viz_infinite_soln2(t′::Float64, savename::String)
	ic_params = ic_params_list(t′, n=170, t_start=-25.0)
	params = (; T₀=T₀, Tₐ=Tₐ, t₀=0.0, τ=τ)
	T′ = T(t′, params)
	
	fig = Figure(resolution = (500, 400))
	ax  = Axis(fig[1, 1], 
			   xlabel="initial time, t₀ [min]", 
			   ylabel="initial temperature, θ₀ [°C]")
	
	draw_axes!(ax)
	viz_air_temp(ax, Tₐ)
	lines!([p.t₀ for p in ic_params], [p.T₀ for p in ic_params], color=my_colors[4])
	scatter!([t′], [T′],
		color="black", strokewidth=1,
		label="(t′, θ(t′))"
	)
	axislegend(@sprintf("τ = %.1f min", τ), position=:rb)
	ylims!(the_ylims...)
	xlims!(minimum(t), maximum(t))
	
	save(savename, fig)
	fig
end

# ╔═╡ b2eb5dc4-3167-4462-93ea-c937b8e9ff1d
viz_infinite_soln2(120.0, "viz_line_of_solns.pdf")

# ╔═╡ Cell order:
# ╠═94182d54-3428-11ec-3014-d9cedb79f68a
# ╠═028c2d73-a4bb-46f2-9b5e-22273c2ca5ce
# ╠═4ce9857b-9723-4f5a-b348-10d7631c546c
# ╠═f2e95386-e5bc-4b90-9c32-43808a8ec8f6
# ╠═5a3522aa-4bd9-498b-bbf8-9e49523f22de
# ╠═3d8112be-ace6-4174-b31b-9896c556a019
# ╠═7d8291bd-59d2-4b98-8dbb-424702bc7412
# ╠═4b58f77f-b155-416f-8907-46b22d809c72
# ╟─eaf92206-2a16-4b54-aa8d-a3c27fbe03e0
# ╠═d5e96b70-0647-4979-9f61-6e9e4c906106
# ╠═081a1336-eb60-4198-9f44-a33973486460
# ╠═6348b57f-6ad4-4dd5-b964-1f88d7c5860e
# ╠═70721425-1988-48d9-b226-b5b36bc525a8
# ╠═5e4c2385-1171-4414-bc3c-a83bfff6abee
# ╠═b5b9d1ad-92f1-43aa-8f08-49c5d09d3e71
# ╠═e9212acc-7874-44ba-9921-73780c2c8e03
# ╠═299faaae-4817-467c-b9b0-9147663dbc17
# ╠═331319e3-4d5f-4980-90e7-b13bf60c5867
# ╟─bd89a236-15a9-45f0-85c9-e63b64c70728
# ╠═a606ff46-73ac-44e0-95fe-b7b8cc6d0273
# ╠═3cd1926a-2a6d-43b9-8dcc-fa1605b69b68
# ╠═9cbbe92a-c751-4d43-8672-98668fd46719
# ╠═d683fdfb-b8ed-4729-a4e9-7c8c6ea77e3e
# ╠═9f44e375-2369-43d7-a562-497c76b9cb1a
# ╠═bfd12fda-7dbb-4a9d-80ec-a506bb908336
# ╠═d76da2e4-dd8c-46d8-b084-88216b00353c
# ╠═fa0bdadb-48a8-47c7-bc7c-e8be316228ce
# ╠═65e6a026-34ad-4df0-8176-00f713818ad0
# ╟─f091f0f1-aee9-403f-90f9-e58556f6c8dc
# ╠═6fec04ae-64bb-4e7f-b82e-96579368f9c5
# ╠═51f5aa61-207b-4033-8b31-6bf6676254e0
# ╠═fd0ac159-1328-40ac-94cd-fdd4cc83f031
# ╠═f07f7b5c-7723-4030-bf88-f060c8466a9c
# ╠═cf4bada1-6b5d-40ab-909a-02ee580c5f40
# ╠═d0b1d7d3-9f85-43e3-af8b-17d73c72777e
# ╠═80b088dc-be57-46a5-8d18-5ed573837058
# ╠═a9f095c4-50a1-41f7-9f66-01e11914f1c0
# ╟─ee98f55a-6196-4b18-87d6-4fb00996e0be
# ╠═e16652ba-d11e-4f87-b3d6-5f019e8c2b5d
# ╠═ca3b6534-9099-4ed8-8627-b85527307f5a
# ╟─bb2fa8fb-32be-47a2-a659-7c3fa209d068
# ╠═4e3e686a-02e6-4fc7-89a7-3aebfdb6b351
# ╠═28a37905-7196-4779-8eb9-b06563b090e5
# ╠═82b478ac-6156-48ca-8f59-206388e255a7
# ╠═196eb94f-28d1-4e75-8670-8b87e36f0048
# ╠═9e0255b0-13d0-4fd5-8aa0-19e04117545d
# ╠═2482c26b-f570-4b04-8fe5-ac75a82d5660
# ╠═eaf6b0f5-ccb0-4e5c-bae1-cc82e34aee85
# ╠═17b6d817-66ae-48e9-a2f1-847be0e6b02d
# ╠═d54e803f-621c-48d3-b50b-aadae0d71296
# ╠═a6574e8c-4ed7-4ec0-9b21-6f38c0141580
# ╠═cdb664e8-cf11-42e5-bd99-bd333f1e5d3a
# ╟─95f4a495-b3f5-476a-bccb-9ebe2141dac3
# ╠═d2d0d681-0323-48fb-b521-c6fe4e006357
# ╠═f7a341b1-8d6f-4263-b2a7-73ea0df11722
# ╠═0f23ddd3-115c-47b6-8fe1-44ba5c117a35
# ╠═5da6eb96-85d7-42fc-9962-e15245838282
# ╠═1ae1eab3-f0f1-453d-8bd7-e269db431d7d
# ╠═b2eb5dc4-3167-4462-93ea-c937b8e9ff1d
