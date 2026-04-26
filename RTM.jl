using CSV
using DataFrames
using Dates
using Statistics
using JuMP
using Gurobi
import MathOptInterface as MOI

# =========================================================
# 0. 파일 경로
# =========================================================
const RT_CO_FILE   = "20251010_rt_co.csv"
const LOAD_FILE    = "20251011_rf_al.csv"
const ELIG_FILE    = "RTM_20251010_.csv"   # 파일명 확인

# =========================================================
# 1. 사용자 설정 파라미터
# =========================================================
const VOLL = 10_000.0

# Reserve procurement cost ($/MW)
const C_REG  = 17.34
const C_SPIN = 4.32
const C_SUPP = 1.01

# =========================================================
# ORDC settings
# shortage s measured in MW
# =========================================================
const ORDC_P1 = 600.0
const ORDC_P2 = 1100.0
const ORDC_P4 = 6000.0

# System reserve requirements from the problem statement
const REG_REQ   = 600.0
const SPIN_MIN  = 1500.0   # reg + spin >= 1500
const SPIN_MAX  = 1800.0   # reg + spin <= 1800
const TOTAL_REQ = 4400.0   # reg + spin + supp >= 4400

# Solve only one day (10/10/2025)
const TARGET_DATE = Date(2025, 10, 10)

const SPILL_PENALTY = 1.0

# =========================================================
# Pmin settings
# PMIN_SCALE = 0.0 -> 기존처럼 Pmin = 0
# PMIN_SCALE = 0.5 -> Economic Min의 50% 적용
# PMIN_SCALE = 1.0 -> Economic Min 전체 적용
# APPLY_PMIN_ONLY_TO_ONLINE_UNITS = true 이면 해당 5분 cleared MW > 0인 발전기만 Pmin 적용
# =========================================================
const PMIN_SCALE = 0.5
const APPLY_PMIN_ONLY_TO_ONLINE_UNITS = true


# =========================================================
# 2. 유틸 함수
# =========================================================

"""
문자열 -> DateTime 파싱
예: "10/10/2025 00:00:00"
"""
function parse_est_datetime(s)
    return DateTime(strip(s), dateformat"m/d/y H:M:S")
end

"""
offer block 상한(MW1~MW10)이 누적치로 들어있으므로
증분 block 폭으로 바꾼다.
예:
MW1=165, MW2=229, MW3=293 이면
block width = [165, 64, 64, ...]
"""
function compute_block_widths(row::DataFrameRow; mw_prefix="MW")
    cum = Float64[]
    for j in 1:10
        v = row[Symbol("$(mw_prefix)$(j)")]
        push!(cum, ismissing(v) || (v isa AbstractFloat && isnan(v)) ? NaN : Float64(v))
    end

    widths = fill(0.0, 10)
    prev = 0.0
    for j in 1:10
        if isnan(cum[j])
            widths[j] = 0.0
        else
            widths[j] = max(cum[j] - prev, 0.0)
            prev = cum[j]
        end
    end
    return widths
end

"""
Price1~Price10 추출
"""
function compute_block_prices(row::DataFrameRow)
    prices = fill(0.0, 10)
    for j in 1:10
        v = row[Symbol("Price$(j)")]
        prices[j] = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? 0.0 : Float64(v)
    end
    return prices
end

"""
5분 ramp capability 계산:
k번째 5분 ramp = abs(ClearedMW_k - ClearedMW_{k+1})
k=1의 경우 같은 시간 내 나머지 ramp 평균값 사용
"""
function compute_unit_max_ramp(row::DataFrameRow)
    cleared = Float64[]

    for k in 1:12
        v = row[Symbol("Cleared MW$(k)")]
        val = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? 0.0 : Float64(v)
        push!(cleared, val)
    end

    diffs = [abs(cleared[k+1] - cleared[k]) for k in 1:11]

    if isempty(diffs)
        return 0.0
    else
        return maximum(diffs)
    end
end

"""
cleared MW가 12개 모두 0이면 off로 간주 -> 아래로 변경
해당 5분 구간 k에서 cleared MW가 0보다 크면 online으로 간주
"""
function infer_online_status_at_k(row::DataFrameRow, k::Int)
    v = row[Symbol("Cleared MW$(k)")]
    val = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? 0.0 : Float64(v)
    return val > 1e-6 ? 1 : 0
end
# =========================================================
# 3. 데이터 로드
# =========================================================

function load_rt_co(path::String)
    df = CSV.read(path, DataFrame)
    rename!(df, Symbol.(names(df)))

    df[!, :dt] = parse_est_datetime.(string.(df[!, Symbol("Mkthour Begin (EST)")]))
    df[!, :date] = Date.(df.dt)
    df[!, :hour_idx] = hour.(df.dt) .+ 1

    df = filter(r -> r.date == TARGET_DATE, df)
    return df
end

function load_eligibility(path::String)
    df = CSV.read(path, DataFrame)
    rename!(df, Symbol.(names(df)))

    out = select(df, :Resource_ID, :Reg_Elig, :Spin_Elig, :Supp_Elig)
    rename!(out, :Resource_ID => :UnitCode)

    out = combine(groupby(out, :UnitCode),
        :Reg_Elig  => first => :Reg_Elig,
        :Spin_Elig => first => :Spin_Elig,
        :Supp_Elig => first => :Supp_Elig
    )

    return out
end

function load_actual_load(path::String)
    df = CSV.read(path, DataFrame; header=6)
    rename!(df, Symbol.(names(df)))

    # 필요한 컬럼 이름 확인
    hour_col = :HourEnding
    load_col = Symbol("MISO ActualLoad (MWh)")
    day_col  = Symbol("Market Day")

    # 빈 행 / 헤더 반복 행 제거
    df = filter(r -> begin
        h = r[hour_col]
        !(ismissing(h) || strip(string(h)) == "" || strip(string(h)) == "HourEnding")
    end, df)

    # 날짜 필터 먼저
    df = filter(r -> strip(string(r[day_col])) == "10/10/2025", df)

    # 안전한 숫자 변환 함수
    safe_parse_int(x) = begin
        s = strip(string(x))
        if isempty(s) || lowercase(s) == "missing"
            missing
        else
            try
                parse(Int, s)
            catch
                missing
            end
        end
    end

    safe_parse_float(x) = begin
        if ismissing(x)
            missing
        else
            s = strip(string(x))
            if isempty(s) || lowercase(s) == "missing"
                missing
            else
                # 쉼표 제거
                s = replace(s, "," => "")
                try
                    parse(Float64, s)
                catch
                    missing
                end
            end
        end
    end

    df[!, :HourEnding_num] = safe_parse_int.(df[!, hour_col])
    df[!, :MISO_ActualLoad_num] = safe_parse_float.(df[!, load_col])

    # 숫자 변환 실패한 행 제거
    df = filter(r -> !ismissing(r[:HourEnding_num]) && !ismissing(r[:MISO_ActualLoad_num]), df)

    load_map = Dict{Int, Float64}()
    for r in eachrow(df)
        load_map[r[:HourEnding_num]] = r[:MISO_ActualLoad_num]
    end

    return load_map
end

# =========================================================
# 4. 시간별 정리 데이터 생성
# =========================================================

function build_hourly_unit_data(rt_co::DataFrame, elig::DataFrame)
    rt2 = deepcopy(rt_co)
    rename!(rt2, Symbol("Unit Code") => :UnitCode)

    merged = leftjoin(rt2, elig, on=:UnitCode)

    for col in [:Reg_Elig, :Spin_Elig, :Supp_Elig]
        merged[!, col] = coalesce.(merged[!, col], false)
    end

    hourly = Dict{Int, DataFrame}()

    for h in 1:24
        hh = filter(r -> r.hour_idx == h, merged)
        hourly[h] = hh
    end

    return hourly
end

# =========================================================
# 5. 단일 5분 구간 SCED 모델
# =========================================================

"""
ORDC penalty:
z1: 0 ~ 10%R     at 600
z2: 10 ~ 12%R    at 1100
z3: 12 ~ 50%R    linearly increasing marginal price from 1100 to 6000
z4: > 50%R       at 6000

Phi(s) =
600*z1
+ 1100*z2
+ 1100*z3 + (4900/(0.76R))*z3^2
+ 6000*z4
"""
function solve_sced_5min(hour_df::DataFrame, D::Float64, h::Int, k::Int)

    n = nrow(hour_df)
    I = 1:n
    J = 1:10

    block_width = Dict{Tuple{Int,Int},Float64}()
    block_price = Dict{Tuple{Int,Int},Float64}()
    ramp_5min   = Dict{Int,Float64}()

    pmin = zeros(n)
    pmax = zeros(n)
    reg_elig  = falses(n)
    spin_elig = falses(n)
    supp_elig = falses(n)

    for i in I
        row = hour_df[i, :]
        widths = compute_block_widths(row)
        prices = compute_block_prices(row)
        ramp_5min[i] = compute_unit_max_ramp(row)
        #ramp_5min[i] = max(50.0, compute_unit_max_ramp(row))

        for j in J
            block_width[(i,j)] = widths[j]
            block_price[(i,j)] = prices[j]
        end

        
        status_k = infer_online_status_at_k(row, k)

        pmin_raw = (ismissing(row[Symbol("Economic Min")]) || (row[Symbol("Economic Min")] isa AbstractFloat && isnan(row[Symbol("Economic Min")]))) ? 0.0 : Float64(row[Symbol("Economic Min")])
        pmax_raw = (ismissing(row[Symbol("Economic Max")]) || (row[Symbol("Economic Max")] isa AbstractFloat && isnan(row[Symbol("Economic Max")]))) ? 0.0 : Float64(row[Symbol("Economic Max")])

        cleared_k = row[Symbol("Cleared MW$(k)")]
        gclear_k = (ismissing(cleared_k) || (cleared_k isa AbstractFloat && isnan(cleared_k))) ? 0.0 : Float64(cleared_k)

        # Pmin setting
        # 기존에는 pmin[i] = 0.0으로 고정했지만,
        # 이제 CSV의 Economic Min을 PMIN_SCALE만큼 적용한다.
        # 단, APPLY_PMIN_ONLY_TO_ONLINE_UNITS=true이면 현재 5분 구간에서
        # 실제 cleared MW가 있는 발전기만 online으로 보고 Pmin을 적용한다.
        if APPLY_PMIN_ONLY_TO_ONLINE_UNITS
            pmin[i] = status_k == 1 ? PMIN_SCALE * pmin_raw : 0.0
        else
            pmin[i] = PMIN_SCALE * pmin_raw
        end

        # Pmin이 Pmax보다 커지는 비정상 데이터 방지
        pmin[i] = min(pmin[i], pmax_raw)
        pmax[i] = pmax_raw

        reg_elig[i]  = Bool(row[:Reg_Elig])
        spin_elig[i] = Bool(row[:Spin_Elig])
        supp_elig[i] = Bool(row[:Supp_Elig])
    end

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    # -------------------------
    # Variables
    # -------------------------
    @variable(model, x[I, J] >= 0)
    @variable(model, g[I] >= 0)

    @variable(model, r_reg[I]  >= 0)
    @variable(model, r_spin[I] >= 0)
    @variable(model, r_supp[I] >= 0)

    @variable(model, s_reg  >= 0)
    @variable(model, s_spin >= 0)
    @variable(model, l >= 0)


    # ORDC segment decomposition
    @variable(model, s_supp >= 0)
    @variable(model, z1 >= 0)
    @variable(model, z2 >= 0)
    @variable(model, z3 >= 0)
    @variable(model, z4 >= 0)

    @variable(model, spill >= 0)

    # -------------------------
    # Expressions
    # -------------------------
    @expression(model, R_cleared,
        sum(r_reg[i] + r_spin[i] + r_supp[i] for i in I)
    )

    @expression(model, ordc_penalty,
        ORDC_P1 * z1 +
        ORDC_P2 * z2 +
        ORDC_P2 * z3 +
        (4900.0 / (0.76 * TOTAL_REQ)) * z3^2 +
        ORDC_P4 * z4
    )

    @expression(model, s_total, s_reg + s_spin + s_supp)


    # -------------------------
    # Objective
    # -------------------------
    @objective(model, Min,
        sum(block_price[(i,j)] * x[i,j] for i in I, j in J) +
        C_REG  * sum(r_reg[i]  for i in I) +
        C_SPIN * sum(r_spin[i] for i in I) +
        C_SUPP * sum(r_supp[i] for i in I) +
        ordc_penalty +
        VOLL * l +
        SPILL_PENALTY * spill
    )

    # -------------------------
    # Constraints
    # -------------------------

    # Generation definition
    @constraint(model, gen_def[i in I], g[i] == pmin[i] + sum(x[i,j] for j in J))

    # Offer block upper bounds
    @constraint(model, block_cap[i in I, j in J], x[i,j] <= block_width[(i,j)])

    # Generation limits
    @constraint(model, gen_max[i in I], g[i] <= pmax[i])

    # Headroom
    @constraint(model, headroom[i in I], g[i] + r_reg[i] + r_spin[i] + r_supp[i] <= pmax[i])

    # Ramp
    @constraint(model, ramp_lim[i in I], r_reg[i] + r_spin[i] + r_supp[i] <= ramp_5min[i])

    # ORDC segment width limits
    @constraint(model, z1 <= 0.10 * TOTAL_REQ)
    @constraint(model, z2 <= 0.02 * TOTAL_REQ)
    @constraint(model, z3 <= 0.38 * TOTAL_REQ)

    @constraint(model, s_total == z1 + z2 + z3 + z4)
    #@constraint(model, s_total <= z1 + z2 + z3 + z4)
    @constraint(model, total_req_con,
    sum(r_reg[i] + r_spin[i] + r_supp[i] for i in I) + s_reg + s_spin + s_supp >= TOTAL_REQ
)

    # Eligibility
    for i in I
        if !reg_elig[i]
            @constraint(model, r_reg[i] == 0)
        end
        if !spin_elig[i]
            @constraint(model, r_spin[i] == 0)
        end
        if !supp_elig[i]
            @constraint(model, r_supp[i] == 0)
        end
    end

    # Reserve hierarchy
    @constraint(model, reg_req_con,
        sum(r_reg[i] for i in I) + s_reg >= REG_REQ
    )

    @constraint(model, spin_req_min_con,
        sum(r_reg[i] + r_spin[i] for i in I) + s_reg + s_spin >= SPIN_MIN
    )

   # @constraint(model, spin_req_max_con,
   #     sum(r_reg[i] + r_spin[i] for i in I) + s_reg + s_spin <= SPIN_MAX
   # )

    # Power balance
    @constraint(model, power_balance,
    sum(g[i] for i in I) + l - spill == D
    )
    """
    println("========================================")
    println("h=$h k=$k")

    println("Demand D = ", D)
    println("sumPmin = ", sum(pmin))
    println("sumPmax = ", sum(pmax))

    # 실제 가능한 최대 reserve 계산 (핵심!)
    energy_needed = D
    total_possible_reserve = sum(max(pmax[i] - pmin[i], 0.0) for i in I)

    # 더 정확한 것 (에너지 충족 후 남는 것)
    system_spare = sum(pmax) - D

    println("system spare after load = ", system_spare)
    println("total_possible_reserve (pmax-pmin sum) = ", total_possible_reserve)

    # reserve 요구 조건 체크
    println("----------------------------------------")
    println("REG_REQ = ", REG_REQ)
    println("SPIN_MIN = ", SPIN_MIN)
    println("TOTAL_REQ = ", TOTAL_REQ)

    # 실제 가능한 reserve upper bound (핵심)
    max_reg_possible = sum(reg_elig[i] ? (pmax[i] - pmin[i]) : 0.0 for i in I)
    max_spin_possible = sum(spin_elig[i] ? (pmax[i] - pmin[i]) : 0.0 for i in I)
    max_total_possible = sum((pmax[i] - pmin[i]) for i in I)

    println("max_reg_possible = ", max_reg_possible)
    println("max_spin_possible = ", max_spin_possible)
    println("max_total_possible = ", max_total_possible)

    println("----------------------------------------")

    if max_reg_possible < REG_REQ
        println("❗ REG infeasible: reg capacity 부족")
    end

    if max_spin_possible < SPIN_MIN
        println("❗ SPIN infeasible: spin capacity 부족")
    end

    if max_total_possible < TOTAL_REQ
        println("❗ TOTAL infeasible: total reserve 부족")
    end

    if system_spare < TOTAL_REQ
        println("❗ ENERGY-CONSTRAINED: 에너지 맞추고 나면 reserve 확보 불가능")
    end

    println("----------------------------------------")
    println("🔥 ENERGY vs RESERVE 충돌 체크")

    # 에너지 맞추고 남는 reserve 가능량
    max_reserve_after_energy = sum(pmax) - D

    println("max_reserve_after_energy = ", max_reserve_after_energy)

    if max_reserve_after_energy < TOTAL_REQ
        println("❗ 핵심 원인: 에너지 맞추면 reserve 확보 불가능")
    end

    offer_max = sum(pmin[i] + sum(block_width[(i,j)] for j in J) for i in I)
    println("offer-based max generation = ", offer_max)

    if offer_max < D
        println("❗ OFFER INFEASIBLE: block 구조로는 발전량 부족")
    end
    """
    optimize!(model)

    println("termination_status = ", termination_status(model))
    println("primal_status      = ", primal_status(model))
    println("dual_status        = ", dual_status(model))
    println("result_count       = ", result_count(model))

    term = termination_status(model)
    primal = primal_status(model)

    if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED))
        return (
            status = string(term),
            primal_status = string(primal),
            objective = NaN,
            lmp = NaN,
            load_shed = NaN,
            s_reg = NaN,
            s_spin = NaN,
            dispatch = DataFrame()
        )
    end

    # LMP = dual(power_balance)
    λ = dual(power_balance)

    g_val      = collect(value.(g))
    rreg_val   = collect(value.(r_reg))
    rspin_val  = collect(value.(r_spin))
    rsupp_val  = collect(value.(r_supp))
    unitcodes  = Vector(hour_df[:, :UnitCode])
    ramp_vec   = [ramp_5min[i] for i in I]

    dispatch = DataFrame(
        UnitCode = unitcodes,
        hour = fill(h, n),
        subinterval = fill(k, n),
        g = g_val,
        r_reg = rreg_val,
        r_spin = rspin_val,
        r_supp = rsupp_val,
        ramp_5min = ramp_vec,
        Pmin = collect(pmin),
        Pmax = collect(pmax),
        Reg_Elig = collect(reg_elig),
        Spin_Elig = collect(spin_elig),
        Supp_Elig = collect(supp_elig)
    )

    return (
        status = string(term),
        primal_status = string(primal),
        objective = objective_value(model),
        lmp = λ,
        load_shed = value(l),
        s_reg = value(s_reg),
        s_spin = value(s_spin),
        dispatch = dispatch
    )
end

# =========================================================
# 6. 하루 전체(24시간 x 12개 5분 구간) 실행
# =========================================================

function solve_all_day(rt_file::String, load_file::String, elig_file::String)

    rt_co  = load_rt_co(rt_file)
    elig   = load_eligibility(elig_file)
    loads  = load_actual_load(load_file)

    hourly_unit_data = build_hourly_unit_data(rt_co, elig)

    summary_rows = DataFrame(
        hour = Int[],
        subinterval = Int[],
        demand = Float64[],
        status = String[],
        objective = Float64[],
        lmp = Float64[],
        load_shed = Float64[],
        s_reg = Float64[],
        s_spin = Float64[],
    )

    dispatch_all = DataFrame()

    for h in 1:24
        hour_df = hourly_unit_data[h]
        D_h = get(loads, h, NaN)

        if isnan(D_h)
            println("WARNING: hour=$h 에 대한 load가 없습니다. skip")
            continue
        end

        for k in 1:12
            result = solve_sced_5min(hour_df, D_h, h, k)

            push!(summary_rows, (
                h,
                k,
                D_h,
                result.status,
                result.objective,
                result.lmp,
                result.load_shed,
                result.s_reg,
                result.s_spin,
            ))

            if nrow(result.dispatch) > 0
                append!(dispatch_all, result.dispatch)
            end
        end
    end

    return summary_rows, dispatch_all
end

# =========================================================
# 7. 실행
# =========================================================

summary_df, dispatch_df = solve_all_day(RT_CO_FILE, LOAD_FILE, ELIG_FILE)

CSV.write("MISO_RTM_summary.csv", summary_df)
CSV.write("MISO_RTM_dispatch.csv", dispatch_df)

println("=== Summary Preview ===")
println(first(summary_df, 10))

println("\n=== Dispatch Preview ===")
println(first(dispatch_df, 10))