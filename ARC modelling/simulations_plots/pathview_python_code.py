import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Adder,
    Amplifier,
    Integrator,
    PulseSource,
    Scope,
    Constant,
    Multiplier,
    Pow,
    Switch,
)

from pathsim_chem import Process, Splitter, GLC

from pathsim.solvers import RKBS32, RKCK54
from pathsim.events import ScheduleList


def arc_same_as_meschini(duration):

   # ────────────────────────────────────────────────────────────────────────────
    # USER-DEFINED CODE
    # ────────────────────────────────────────────────────────────────────────────

    fusion_power = 525  # MWth
    tritium_burn_rate = 9.3e-7  # kg/s
    pulse_duration = 1800  # s
    time_between_pulses = 60  # s
    TBE = 0.02
    non_rad_loss_fraction = 1e-4
    AF = 0.7
    tritium_processing_time = 4 * 3600  # s
    dir_frac = 0.3

    tau_blanket = 1.25 * 3600  # s
    tau_fw = 1000  # s
    tau_divertor = 1000  # s
    tau_tes = 24 * 3600  # s
    tau_hx = 1000  # s
    tau_vacuum_pump = 600  # s
    tau_fuel_cleanup = 0.3 * 3600  # s
    tau_iss = 3 * 3600  # s
    tau_detritiation = 1 * 3600  # s
    tau_membrane = 100  # s

    f_p3 = 1e-4
    f_p4 = 1e-4

    TBR = 1.05
    tes_efficiency = 0.95
    startup_inventory = 1.14

    # ────────────────────────────────────────────────────────────────────────────
    # BLOCKS
    # ────────────────────────────────────────────────────────────────────────────

    # Sources
    fusion_reaction_rate = PulseSource(
        T=pulse_duration,
        amplitude=tritium_burn_rate,
        duty=AF,
        t_rise=pulse_duration * 0.01,
        t_fall=pulse_duration * 0.01
    )

    # Dynamic
    storage = Integrator(
        initial_value=startup_inventory
    )

    # Algebraic
    plasma_to_div = Amplifier(
        gain=f_p4/TBE
    )
    plasma_to_fw = Amplifier(
        gain=f_p3/TBE
    )
    x_tbr = Amplifier(
        gain=TBR
    )
    injection_rate = Amplifier(
        gain=-1/TBE
    )
    pumping_rate = Amplifier(
        gain=(1 - TBE - f_p3 - f_p4) / TBE
    )
    adder = Adder()

    # Recording
    outer_fuel_cycle = Scope(
        labels=['Divertor','FW','Blanket','TES','HX']
    )
    fusion_rate = Scope(
        labels=['Fusion rate']
    )
    inner_fuel_cycle = Scope(
        labels=['Storage','Pump','ISS','Cleanup']
    )

    # Chemical
    divertor = Process(
        tau=tau_divertor
    )
    fw = Process(
        tau=tau_fw
    )
    blanket = Process(
        tau=tau_blanket
    )
    t_separation_membrane = Process(
        tau=tau_membrane
    )
    tes = Process(
        tau=tau_tes
    )
    heat_exchanger = Process(
        tau=tau_hx
    )
    pump = Process(
        tau=tau_vacuum_pump
    )
    fuel_cleanup = Process(
        tau=tau_fuel_cleanup
    )
    iss = Process(
        tau=tau_iss
    )
    detritiation = Process(
        tau=tau_detritiation
    )
    hx_splitter = Splitter(
        fractions=[1/3, 1/3, 1/3]
    )
    detrit__storage = Splitter(
        fractions=[0.9 , 0.1]
    )
    storage__cleanup = Splitter(
        fractions=[dir_frac, 1 - dir_frac]
    )
    tes_eff = Splitter(
        fractions=[tes_efficiency, 1 - tes_efficiency]
    )

    blocks = [
        fusion_reaction_rate,
        storage,
        plasma_to_div,
        plasma_to_fw,
        x_tbr,
        injection_rate,
        pumping_rate,
        adder,
        outer_fuel_cycle,
        fusion_rate,
        inner_fuel_cycle,
        divertor,
        fw,
        blanket,
        t_separation_membrane,
        tes,
        heat_exchanger,
        pump,
        fuel_cleanup,
        iss,
        detritiation,
        hx_splitter,
        detrit__storage,
        storage__cleanup,
        tes_eff,
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # CONNECTIONS
    # ────────────────────────────────────────────────────────────────────────────

    connections = [
        Connection(fusion_reaction_rate[0], plasma_to_div[0], plasma_to_fw[0], x_tbr[0], injection_rate[0], pumping_rate[0], fusion_rate[0]),
        Connection(plasma_to_div[0], divertor[0]),
        Connection(plasma_to_fw[0], fw[0]),
        Connection(x_tbr[0], blanket[0]),
        Connection(fw[1], blanket[1]),
        Connection(divertor[1], blanket[2]),
        Connection(blanket[1], tes[0]),
        Connection(pumping_rate[0], pump[0]),
        Connection(fuel_cleanup[1], iss[0]),
        Connection(detritiation[1], iss[1]),
        Connection(divertor[0], outer_fuel_cycle[0]),
        Connection(fw[0], outer_fuel_cycle[1]),
        Connection(blanket[0], outer_fuel_cycle[2]),
        Connection(tes[0], outer_fuel_cycle[3]),
        Connection(heat_exchanger[0], outer_fuel_cycle[4]),
        Connection(pump[0], inner_fuel_cycle[1]),
        Connection(iss[0], inner_fuel_cycle[2]),
        Connection(fuel_cleanup[0], inner_fuel_cycle[3]),
        Connection(heat_exchanger[1], hx_splitter[0]),
        Connection(hx_splitter[2], divertor[1]),
        Connection(hx_splitter[1], fw[1]),
        Connection(hx_splitter[0], blanket[3]),
        Connection(iss[1], detrit__storage[0]),
        Connection(detrit__storage[1], detritiation[0]),
        Connection(storage__cleanup[1], fuel_cleanup[0]),
        Connection(pump[1], storage__cleanup[0]),
        Connection(tes[1], tes_eff[0]),
        Connection(tes_eff[1], heat_exchanger[0]),
        Connection(tes_eff[0], t_separation_membrane[0]),
        Connection(storage[0], inner_fuel_cycle[0]),
        Connection(adder[0], storage[0]),
        Connection(storage__cleanup[0], adder[0]),
        Connection(injection_rate[0], adder[1]),
        Connection(t_separation_membrane[1], adder[2]),
        Connection(detrit__storage[0], adder[3]),
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # SIMULATION
    # ────────────────────────────────────────────────────────────────────────────

    sim = Simulation(
        blocks,
        connections,
        Solver=RKBS32,
        dt=0.02,
        dt_min=1e-16,
        tolerance_lte_rel=0.0001,
        tolerance_lte_abs=1e-08,
        tolerance_fpi=1e-10,
    )


    
    # ────────────────────────────────────────────────────────────────────────────
    # MAIN
    # ────────────────────────────────────────────────────────────────────────────


    # Run simulation
    sim.run(duration)

    # ────────────────────────────────────────────────────────────────────────────
    ### SAVE RESULTS ###
    # ────────────────────────────────────────────────────────────────────────────

    import json

    results = sim.collect()
    scopes = results['scopes']

    # Create a structured dictionary for saving
    save_dict = {}
    scope_metadata = {}

    for scope_idx, (scope_id, scope_dict) in enumerate(scopes.items()):
        time = scope_dict['time']
        data = scope_dict['data']
        labels = scope_dict.get('labels', [])
        
        # Create scope name from labels or use generic name
        if labels and any(labels):  # If labels exist and aren't all empty
            scope_name = f"scope_{scope_idx}_{'_'.join(labels)}"
        else:
            scope_name = f"scope_{scope_idx}"
        
        # Clean up scope name to be filesystem-friendly
        scope_name = scope_name.replace(" ", "_").replace("-", "_")[:60]  # Limit length
        
        # Save time (only once per scope)
        save_dict[f"{scope_name}_time"] = time
        
        # Save data
        save_dict[f"{scope_name}_data"] = data
        
        # Store metadata for labels (JSON string since npz can't store lists directly)
        scope_metadata[scope_name] = {
            'labels': labels,
            'data_shape': data.shape,
            'time_length': len(time)
        }

    # Create results directory if it doesn't exist
    import os
    results_dir = '/results'  
    os.makedirs(results_dir, exist_ok=True)

    # Save the main data
    output_file = os.path.join(results_dir, 'ARC_same_as_meschini_results.npz')
    np.savez(output_file, **save_dict)

    # Save metadata as JSON in the same directory
    import os
    metadata_file = os.path.join(os.path.dirname(output_file), 'ARC_same_as_meschini_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(scope_metadata, f, indent=2)

    print(f"Results saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")

def arc_single_bcr(duration):


    # ────────────────────────────────────────────────────────────────────────────
    # USER-DEFINED CODE
    # ────────────────────────────────────────────────────────────────────────────

    ### Residence time model ###

    fusion_power = 525  # MWth
    tritium_burn_rate = 9.3e-7  # kg/s
    pulse_duration = 1800  # s
    time_between_pulses = 60  # s
    TBE = 0.02
    non_rad_loss_fraction = 1e-4
    AF = 0.7
    tritium_processing_time = 4 * 3600  # s
    dir_frac = 0.3

    tau_blanket = 1.25 * 3600  # s
    tau_fw = 1000  # s
    tau_divertor = 1000  # s
    tau_tes = 24 * 3600  # s
    tau_hx = 1000  # s
    tau_vacuum_pump = 600  # s
    tau_fuel_cleanup = 0.3 * 3600  # s
    tau_iss = 3 * 3600  # s
    tau_detritiation = 1 * 3600  # s
    tau_membrane = 100  # s

    f_p3 = 1e-4
    f_p4 = 1e-4

    TBR = 1.05
    tes_efficiency = 0.95
    startup_inventory = 1.14

    ### BCR model ###
    total_flow_l = 560 # kg / s
    total_flow_g = 0.19 # mol / s
    T = 623 # K
    D = 0.5 # m
    L = 3 # m
    P_in = 5e5 # Pa
    BCs = "C-C"
    T_mol_mass = 3.01605 # [g / mol]
    rho_l = 10.45e3 * (1 - 1.61e-4 * T) # kg / m3

    # ────────────────────────────────────────────────────────────────────────────
    # BLOCKS
    # ────────────────────────────────────────────────────────────────────────────

    # Sources
    fusion_reaction_rate = PulseSource(
        T=pulse_duration,
        amplitude=tritium_burn_rate,
        duty=AF,
        t_rise=pulse_duration * 0.01,
        t_fall=pulse_duration * 0.01
    )
    n_1000__t_mol_mass_mol__kg = Constant(
        value=1000 / T_mol_mass
    )
    total_flow_l_kgs = Constant(
        value=total_flow_l
    )
    n_1__rho_l_m3__kg = Constant(
        value=1 / rho_l
    )
    y_t2_in = Constant(
        value=0
    )
    flow_g_m3__s = Constant(
        value=total_flow_g
    )
    t_mol_mass_kg__mol = Constant(
        value=T_mol_mass / 1000
    )

    # Dynamic
    storage = Integrator(
        initial_value=startup_inventory
    )

    # Algebraic
    plasma_to_div = Amplifier(
        gain=f_p4/TBE
    )
    plasma_to_fw = Amplifier(
        gain=f_p3/TBE
    )
    x_tbr = Amplifier(
        gain=TBR
    )
    injection_rate = Amplifier(
        gain=-1/TBE
    )
    pumping_rate = Amplifier(
        gain=(1 - TBE - f_p3 - f_p4) / TBE
    )
    t_out_blanket_mol__s = Multiplier()
    q_l_m3__s = Multiplier()
    n_1__q_l_s__m3 = Pow(
        exponent=-1
    )
    c_t_in_mol__m3 = Multiplier()
    n_t_out_liquid_kg__s = Multiplier()
    n_t_out_gas_kg__s = Multiplier()
    adder = Adder()

    # Recording
    outer_fuel_cycle = Scope(
        labels=["Divertor","FW","Blanket","HX"]
    )
    fusion_rate = Scope(
        labels=['Fusion rate']
    )
    inner_fuel_cycle = Scope(
        labels=["Storage","Pump","ISS","Cleanup"]
    )
    bcr_eff = Scope()
    c_t_in__out = Scope(
        labels=["c_T_in","c_T_out"]
    )
    y_t2_in__out = Scope(
        labels=["y_T2_in","y_T2_out"]
    )

    # Chemical
    divertor = Process(
        tau=tau_divertor
    )
    fw = Process(
        tau=tau_fw
    )
    blanket = Process(
        tau=tau_blanket,
        initial_value=1e-4
    )
    t_separation_membrane = Process(
        tau=tau_membrane
    )
    heat_exchanger = Process(
        tau=tau_hx
    )
    pump = Process(
        tau=tau_vacuum_pump
    )
    fuel_cleanup = Process(
        tau=tau_fuel_cleanup
    )
    iss = Process(
        tau=tau_iss
    )
    detritiation = Process(
        tau=tau_detritiation
    )
    hx_splitter = Splitter()
    detrit__storage = Splitter(
        fractions=[0.9 , 0.1]
    )
    storage__cleanup = Splitter(
        fractions=[dir_frac, 1 - dir_frac]
    )
    glc = GLC(
        BCs=BCs,
        T=T,
        D=D,
        L=L,
        P_in=P_in
    )

    blocks = [
        fusion_reaction_rate,
        n_1000__t_mol_mass_mol__kg,
        total_flow_l_kgs,
        n_1__rho_l_m3__kg,
        y_t2_in,
        flow_g_m3__s,
        t_mol_mass_kg__mol,
        storage,
        plasma_to_div,
        plasma_to_fw,
        x_tbr,
        injection_rate,
        pumping_rate,
        t_out_blanket_mol__s,
        q_l_m3__s,
        n_1__q_l_s__m3,
        c_t_in_mol__m3,
        n_t_out_liquid_kg__s,
        n_t_out_gas_kg__s,
        adder,
        outer_fuel_cycle,
        fusion_rate,
        inner_fuel_cycle,
        bcr_eff,
        c_t_in__out,
        y_t2_in__out,
        divertor,
        fw,
        blanket,
        t_separation_membrane,
        heat_exchanger,
        pump,
        fuel_cleanup,
        iss,
        detritiation,
        hx_splitter,
        detrit__storage,
        storage__cleanup,
        glc,
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # CONNECTIONS
    # ────────────────────────────────────────────────────────────────────────────

    connections = [
        Connection(fusion_reaction_rate[0], plasma_to_div[0], plasma_to_fw[0], x_tbr[0], injection_rate[0], pumping_rate[0], fusion_rate[0]),
        Connection(plasma_to_div[0], divertor[0]),
        Connection(plasma_to_fw[0], fw[0]),
        Connection(x_tbr[0], blanket[0]),
        Connection(fw[1], blanket[1]),
        Connection(divertor[1], blanket[2]),
        Connection(pumping_rate[0], pump[0]),
        Connection(fuel_cleanup[1], iss[0]),
        Connection(detritiation[1], iss[1]),
        Connection(divertor[0], outer_fuel_cycle[0]),
        Connection(fw[0], outer_fuel_cycle[1]),
        Connection(blanket[0], outer_fuel_cycle[2]),
        Connection(pump[0], inner_fuel_cycle[1]),
        Connection(iss[0], inner_fuel_cycle[2]),
        Connection(fuel_cleanup[0], inner_fuel_cycle[3]),
        Connection(storage[0], inner_fuel_cycle[0]),
        Connection(heat_exchanger[1], hx_splitter[0]),
        Connection(hx_splitter[2], divertor[1]),
        Connection(hx_splitter[1], fw[1]),
        Connection(hx_splitter[0], blanket[3]),
        Connection(iss[1], detrit__storage[0]),
        Connection(detrit__storage[1], detritiation[0]),
        Connection(storage__cleanup[1], fuel_cleanup[0]),
        Connection(pump[1], storage__cleanup[0]),
        Connection(blanket[1], t_out_blanket_mol__s[0]),
        Connection(n_1000__t_mol_mass_mol__kg[0], t_out_blanket_mol__s[1]),
        Connection(q_l_m3__s[0], n_1__q_l_s__m3[0]),
        Connection(t_out_blanket_mol__s[0], c_t_in_mol__m3[0]),
        Connection(n_1__q_l_s__m3[0], c_t_in_mol__m3[1]),
        Connection(n_1__rho_l_m3__kg[0], q_l_m3__s[0]),
        Connection(total_flow_l_kgs[0], q_l_m3__s[1], glc[1]),
        Connection(c_t_in_mol__m3[0], glc[0], c_t_in__out[0]),
        Connection(y_t2_in[0], glc[2], y_t2_in__out[0]),
        Connection(flow_g_m3__s[0], glc[3]),
        Connection(n_t_out_liquid_kg__s[0], heat_exchanger[0]),
        Connection(t_mol_mass_kg__mol[0], n_t_out_liquid_kg__s[0], n_t_out_gas_kg__s[1]),
        Connection(glc[6], n_t_out_liquid_kg__s[1]),
        Connection(glc[7], n_t_out_gas_kg__s[0]),
        Connection(n_t_out_gas_kg__s[0], t_separation_membrane[0]),
        Connection(glc[2], bcr_eff[0]),
        Connection(glc[0], c_t_in__out[1]),
        Connection(glc[1], y_t2_in__out[1]),
        Connection(heat_exchanger[0], outer_fuel_cycle[3]),
        Connection(adder[0], storage[0]),
        Connection(storage__cleanup[0], adder[0]),
        Connection(injection_rate[0], adder[1]),
        Connection(t_separation_membrane[1], adder[2]),
        Connection(detrit__storage[0], adder[3]),
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # SIMULATION
    # ────────────────────────────────────────────────────────────────────────────

    sim = Simulation(
        blocks,
        connections,
        Solver=RKBS32,
        dt=0.02,
        dt_min=1e-16,
        tolerance_lte_rel=0.0001,
        tolerance_lte_abs=1e-08,
        tolerance_fpi=1e-10,
    )

        # ────────────────────────────────────────────────────────────────────────────
    # MAIN
    # ────────────────────────────────────────────────────────────────────────────


    # Run simulation
    sim.run(duration)

    # ────────────────────────────────────────────────────────────────────────────
    ### SAVE RESULTS ###
    # ────────────────────────────────────────────────────────────────────────────

    import json

    results = sim.collect()
    scopes = results['scopes']

    # Create a structured dictionary for saving
    save_dict = {}
    scope_metadata = {}

    for scope_idx, (scope_id, scope_dict) in enumerate(scopes.items()):
        time = scope_dict['time']
        data = scope_dict['data']
        labels = scope_dict.get('labels', [])
        
        # Create scope name from labels or use generic name
        if labels and any(labels):  # If labels exist and aren't all empty
            scope_name = f"scope_{scope_idx}_{'_'.join(labels)}"
        else:
            scope_name = f"scope_{scope_idx}"
        
        # Clean up scope name to be filesystem-friendly
        scope_name = scope_name.replace(" ", "_").replace("-", "_")[:60]  # Limit length
        
        # Save time (only once per scope)
        save_dict[f"{scope_name}_time"] = time
        
        # Save data
        save_dict[f"{scope_name}_data"] = data
        
        # Store metadata for labels (JSON string since npz can't store lists directly)
        scope_metadata[scope_name] = {
            'labels': labels,
            'data_shape': data.shape,
            'time_length': len(time)
        }

    # Create results directory if it doesn't exist
    import os
    results_dir = '/results'  
    os.makedirs(results_dir, exist_ok=True)

    # Save the main data
    output_file = os.path.join(results_dir, 'ARC_single_bcr_results.npz')
    np.savez(output_file, **save_dict)

    # Save metadata as JSON in the same directory
    import os
    metadata_file = os.path.join(os.path.dirname(output_file), 'ARC_single_bcr_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(scope_metadata, f, indent=2)

    print(f"Results saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")

def arc_series_bcrs(duration):
    ### Residence time model ###

    fusion_power = 525  # MWth
    tritium_burn_rate = 9.3e-7  # kg/s
    pulse_duration = 1800  # s
    time_between_pulses = 60  # s
    TBE = 0.02
    non_rad_loss_fraction = 1e-4
    AF = 0.7
    tritium_processing_time = 4 * 3600  # s
    dir_frac = 0.3

    tau_blanket = 1.25 * 3600  # s
    tau_fw = 1000  # s
    tau_divertor = 1000  # s
    tau_tes = 24 * 3600  # s
    tau_hx = 1000  # s
    tau_vacuum_pump = 600  # s
    tau_fuel_cleanup = 0.3 * 3600  # s
    tau_iss = 3 * 3600  # s
    tau_detritiation = 1 * 3600  # s
    tau_membrane = 100  # s

    f_p3 = 1e-4
    f_p4 = 1e-4

    TBR = 1.05
    tes_efficiency = 0.95
    startup_inventory = 1.14

    ### BCR model ###
    total_flow_l = 560 # kg / s
    total_flow_g = 0.19 # mol / s
    T = 623 # K
    D = 0.5 # m
    L = 1 # m
    P_in = 5e5 # Pa
    BCs = "C-C"
    T_mol_mass = 3.01605 # [g / mol]
    rho_l = 10.45e3 * (1 - 1.61e-4 * T) # kg / m3

    # ────────────────────────────────────────────────────────────────────────────
    # BLOCKS
    # ────────────────────────────────────────────────────────────────────────────

    # Sources
    fusion_reaction_rate = PulseSource(
        T=pulse_duration,
        amplitude=tritium_burn_rate,
        duty=AF,
        t_rise=pulse_duration * 0.01,
        t_fall=pulse_duration * 0.01
    )
    n_1000__t_mol_mass_mol__kg = Constant(
        value=1000 / T_mol_mass
    )
    total_flow_l_kgs = Constant(
        value=total_flow_l
    )
    n_1__rho_l_m3__kg = Constant(
        value=1 / rho_l
    )
    y_t2_in = Constant(
        value=0
    )
    flow_g_m3__s = Constant(
        value=total_flow_g
    )
    t_mol_mass_kg__mol = Constant(
        value=T_mol_mass / 1000
    )

    # Dynamic
    storage = Integrator(
        initial_value=startup_inventory
    )

    # Algebraic
    plasma_to_div = Amplifier(
        gain=f_p4/TBE
    )
    plasma_to_fw = Amplifier(
        gain=f_p3/TBE
    )
    x_tbr = Amplifier(
        gain=TBR
    )
    injection_rate = Amplifier(
        gain=-1/TBE
    )
    pumping_rate = Amplifier(
        gain=(1 - TBE - f_p3 - f_p4) / TBE
    )
    t_out_blanket_mol__s = Multiplier()
    q_l_m3__s = Multiplier()
    n_1__q_l_s__m3 = Pow(
        exponent=-1
    )
    c_t_in_mol__m3 = Multiplier()
    n_t_out_liquid_kg__s = Multiplier()
    n_t_out_gas_kg__s = Multiplier()
    adder = Adder()

    # Recording
    outer_fuel_cycle = Scope(
        labels=["Divertor", "FW", "Blanket", "HX"]
    )
    fusion_rate = Scope(
        labels=['Fusion rate']
    )
    inner_fuel_cycle = Scope(
        labels=["Storage", "Pump", "ISS", "Cleanup"]
    )
    bcr_eff = Scope(
        labels=["BCR_1","BCR_2","BCR_3"]
    )
    c_t_in__out = Scope(
        labels=["c_T_in","c_T_out_1","c_T_out_2","c_T_out_3"]
    )
    y_t2_in__out = Scope(
        labels=["y_T2_in","y_T2_out_1","y_T2_out_2","y_T2_out_3"]
    )

    # Chemical
    divertor = Process(
        tau=tau_divertor
    )
    fw = Process(
        tau=tau_fw
    )
    blanket = Process(
        tau=tau_blanket,
        initial_value=1e-4
    )
    t_separation_membrane = Process(
        tau=tau_membrane
    )
    heat_exchanger = Process(
        tau=tau_hx
    )
    pump = Process(
        tau=tau_vacuum_pump
    )
    fuel_cleanup = Process(
        tau=tau_fuel_cleanup
    )
    iss = Process(
        tau=tau_iss
    )
    detritiation = Process(
        tau=tau_detritiation
    )
    hx_splitter = Splitter()
    detrit__storage = Splitter(
        fractions=[0.9 , 0.1]
    )
    storage__cleanup = Splitter(
        fractions=[dir_frac, 1 - dir_frac]
    )
    bcr_1 = GLC(
        BCs=BCs,
        T=T,
        D=D,
        L=L,
        P_in=P_in
    )
    bcr_2 = GLC(
        BCs=BCs,
        T=T,
        D=D,
        L=L,
        P_in=P_in
    )
    bcr_3 = GLC(
        BCs=BCs,
        T=T,
        D=D,
        L=L,
        P_in=P_in
    )

    blocks = [
        fusion_reaction_rate,
        n_1000__t_mol_mass_mol__kg,
        total_flow_l_kgs,
        n_1__rho_l_m3__kg,
        y_t2_in,
        flow_g_m3__s,
        t_mol_mass_kg__mol,
        storage,
        plasma_to_div,
        plasma_to_fw,
        x_tbr,
        injection_rate,
        pumping_rate,
        t_out_blanket_mol__s,
        q_l_m3__s,
        n_1__q_l_s__m3,
        c_t_in_mol__m3,
        n_t_out_liquid_kg__s,
        n_t_out_gas_kg__s,
        adder,
        outer_fuel_cycle,
        fusion_rate,
        inner_fuel_cycle,
        bcr_eff,
        c_t_in__out,
        y_t2_in__out,
        divertor,
        fw,
        blanket,
        t_separation_membrane,
        heat_exchanger,
        pump,
        fuel_cleanup,
        iss,
        detritiation,
        hx_splitter,
        detrit__storage,
        storage__cleanup,
        bcr_1,
        bcr_2,
        bcr_3,
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # CONNECTIONS
    # ────────────────────────────────────────────────────────────────────────────

    connections = [
        Connection(fusion_reaction_rate[0], plasma_to_div[0], plasma_to_fw[0], x_tbr[0], injection_rate[0], pumping_rate[0], fusion_rate[0]),
        Connection(plasma_to_div[0], divertor[0]),
        Connection(plasma_to_fw[0], fw[0]),
        Connection(x_tbr[0], blanket[0]),
        Connection(fw[1], blanket[1]),
        Connection(divertor[1], blanket[2]),
        Connection(pumping_rate[0], pump[0]),
        Connection(fuel_cleanup[1], iss[0]),
        Connection(detritiation[1], iss[1]),
        Connection(divertor[0], outer_fuel_cycle[0]),
        Connection(fw[0], outer_fuel_cycle[1]),
        Connection(blanket[0], outer_fuel_cycle[2]),
        Connection(pump[0], inner_fuel_cycle[1]),
        Connection(iss[0], inner_fuel_cycle[2]),
        Connection(fuel_cleanup[0], inner_fuel_cycle[3]),
        Connection(storage[0], inner_fuel_cycle[0]),
        Connection(heat_exchanger[1], hx_splitter[0]),
        Connection(hx_splitter[2], divertor[1]),
        Connection(hx_splitter[1], fw[1]),
        Connection(hx_splitter[0], blanket[3]),
        Connection(iss[1], detrit__storage[0]),
        Connection(detrit__storage[1], detritiation[0]),
        Connection(storage__cleanup[1], fuel_cleanup[0]),
        Connection(pump[1], storage__cleanup[0]),
        Connection(blanket[1], t_out_blanket_mol__s[0]),
        Connection(n_1000__t_mol_mass_mol__kg[0], t_out_blanket_mol__s[1]),
        Connection(q_l_m3__s[0], n_1__q_l_s__m3[0]),
        Connection(t_out_blanket_mol__s[0], c_t_in_mol__m3[0]),
        Connection(n_1__q_l_s__m3[0], c_t_in_mol__m3[1]),
        Connection(n_1__rho_l_m3__kg[0], q_l_m3__s[0]),
        Connection(total_flow_l_kgs[0], q_l_m3__s[1], bcr_1[1], bcr_2[1], bcr_3[1]),
        Connection(c_t_in_mol__m3[0], bcr_1[0], c_t_in__out[0]),
        Connection(y_t2_in[0], bcr_1[2], y_t2_in__out[0]),
        Connection(flow_g_m3__s[0], bcr_1[3], bcr_2[3], bcr_3[3]),
        Connection(n_t_out_liquid_kg__s[0], heat_exchanger[0]),
        Connection(t_mol_mass_kg__mol[0], n_t_out_liquid_kg__s[0], n_t_out_gas_kg__s[1]),
        Connection(n_t_out_gas_kg__s[0], t_separation_membrane[0]),
        Connection(bcr_1[2], bcr_eff[0]),
        Connection(bcr_1[0], c_t_in__out[1], bcr_2[0]),
        Connection(bcr_1[1], y_t2_in__out[1], bcr_2[2]),
        Connection(bcr_2[0], bcr_3[0], c_t_in__out[2]),
        Connection(bcr_2[1], bcr_3[2], y_t2_in__out[2]),
        Connection(bcr_3[7], n_t_out_gas_kg__s[0]),
        Connection(bcr_3[6], n_t_out_liquid_kg__s[1]),
        Connection(bcr_3[0], c_t_in__out[3]),
        Connection(bcr_3[1], y_t2_in__out[3]),
        Connection(heat_exchanger[0], outer_fuel_cycle[3]),
        Connection(bcr_2[2], bcr_eff[1]),
        Connection(bcr_3[2], bcr_eff[2]),
        Connection(adder[0], storage[0]),
        Connection(storage__cleanup[0], adder[0]),
        Connection(injection_rate[0], adder[1]),
        Connection(t_separation_membrane[1], adder[2]),
        Connection(detrit__storage[0], adder[3]),
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # SIMULATION
    # ────────────────────────────────────────────────────────────────────────────

    sim = Simulation(
        blocks,
        connections,
        Solver=RKBS32,
        dt=0.02,
        dt_min=1e-16,
        tolerance_lte_rel=0.0001,
        tolerance_lte_abs=1e-08,
        tolerance_fpi=1e-10,
    )

    # ────────────────────────────────────────────────────────────────────────────
    # MAIN
    # ────────────────────────────────────────────────────────────────────────────


    # Run simulation
    sim.run(duration)

    # ────────────────────────────────────────────────────────────────────────────
    ### SAVE RESULTS ###
    # ────────────────────────────────────────────────────────────────────────────

    import json

    results = sim.collect()
    scopes = results['scopes']

    # Create a structured dictionary for saving
    save_dict = {}
    scope_metadata = {}

    for scope_idx, (scope_id, scope_dict) in enumerate(scopes.items()):
        time = scope_dict['time']
        data = scope_dict['data']
        labels = scope_dict.get('labels', [])
        
        # Create scope name from labels or use generic name
        if labels and any(labels):  # If labels exist and aren't all empty
            scope_name = f"scope_{scope_idx}_{'_'.join(labels)}"
        else:
            scope_name = f"scope_{scope_idx}"
        
        # Clean up scope name to be filesystem-friendly
        scope_name = scope_name.replace(" ", "_").replace("-", "_")[:60]  # Limit length
        
        # Save time (only once per scope)
        save_dict[f"{scope_name}_time"] = time
        
        # Save data
        save_dict[f"{scope_name}_data"] = data
        
        # Store metadata for labels (JSON string since npz can't store lists directly)
        scope_metadata[scope_name] = {
            'labels': labels,
            'data_shape': data.shape,
            'time_length': len(time)
        }

    # Create results directory if it doesn't exist
    import os
    results_dir = '/results'  
    os.makedirs(results_dir, exist_ok=True)

    # Save the main data
    output_file = os.path.join(results_dir, 'ARC_series_bcr_results.npz')
    np.savez(output_file, **save_dict)

    # Save metadata as JSON in the same directory
    import os
    metadata_file = os.path.join(os.path.dirname(output_file), 'ARC_series_bcr_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(scope_metadata, f, indent=2)

    print(f"Results saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")

def arc_single_bcr_shutdown(duration):
    ### Residence time model ###

    fusion_power = 525  # MWth
    tritium_burn_rate = 9.3e-7  # kg/s
    pulse_duration = 1800  # s
    time_between_pulses = 60  # s
    TBE = 0.02
    non_rad_loss_fraction = 1e-4
    AF = 0.7
    tritium_processing_time = 4 * 3600  # s
    dir_frac = 0.3

    tau_blanket = 1.25 * 3600  # s
    tau_fw = 1000  # s
    tau_divertor = 1000  # s
    tau_tes = 24 * 3600  # s
    tau_hx = 1000  # s
    tau_vacuum_pump = 600  # s
    tau_fuel_cleanup = 0.3 * 3600  # s
    tau_iss = 3 * 3600  # s
    tau_detritiation = 1 * 3600  # s
    tau_membrane = 100  # s

    f_p3 = 1e-4
    f_p4 = 1e-4

    TBR = 1.05
    tes_efficiency = 0.95
    startup_inventory = 1.14

    ### BCR model ###
    total_flow_l = 560 # kg / s
    total_flow_g = 0.19 # mol / s
    T = 623 # K
    D = 0.5 # m
    L = 3 # m
    P_in = 5e5 # Pa
    BCs = "C-C"
    T_mol_mass = 3.01605 # [g / mol]
    rho_l = 10.45e3 * (1 - 1.61e-4 * T) # kg / m3

    ### Shutdown / Startup events ###
    def act_sd(t):
        liquid_switch.select(0)
        gas_switch.select(1)

    def act_su(t):
        liquid_switch.select(1)
        gas_switch.select(0)

    # ────────────────────────────────────────────────────────────────────────────
    # BLOCKS
    # ────────────────────────────────────────────────────────────────────────────

    # Sources
    fusion_reaction_rate = PulseSource(
        T=pulse_duration,
        amplitude=tritium_burn_rate,
        duty=AF,
        t_rise=pulse_duration * 0.01,
        t_fall=pulse_duration * 0.01
    )
    n_1000__t_mol_mass_mol__kg = Constant(
        value=1000 / T_mol_mass
    )
    total_flow_l_kgs = Constant(
        value=total_flow_l
    )
    n_1__rho_l_m3__kg = Constant(
        value=1 / rho_l
    )
    y_t2_in = Constant(
        value=0
    )
    flow_g_m3__s = Constant(
        value=total_flow_g
    )
    t_mol_mass_kg__mol = Constant(
        value=T_mol_mass / 1000
    )

    # Dynamic
    storage = Integrator(
        initial_value=startup_inventory
    )

    # Algebraic
    plasma_to_div = Amplifier(
        gain=f_p4/TBE
    )
    plasma_to_fw = Amplifier(
        gain=f_p3/TBE
    )
    x_tbr = Amplifier(
        gain=TBR
    )
    injection_rate = Amplifier(
        gain=-1/TBE
    )
    pumping_rate = Amplifier(
        gain=(1 - TBE - f_p3 - f_p4) / TBE
    )
    t_out_blanket_mol__s = Multiplier()
    q_l_m3__s = Multiplier()
    n_1__q_l_s__m3 = Pow(
        exponent=-1
    )
    c_t_in_mol__m3 = Multiplier()
    n_t_out_liquid_kg__s = Multiplier()
    n_t_out_gas_kg__s = Multiplier()
    liquid_switch = Switch(
        state=1
    )
    gas_switch = Switch(
        state=1
    )
    adder = Adder()

    # Recording
    outer_fuel_cycle = Scope(
        labels=["Divertor","FW","Blanket","HX"]
    )
    fusion_rate = Scope(
        labels=['Fusion rate']
    )
    inner_fuel_cycle = Scope(
        labels=["Storage","Pump","ISS","Cleanup","Detritiation"]
    )
    bcr_eff = Scope()
    c_t_in__out = Scope(
        labels=["c_T_in","c_T_out"]
    )
    y_t2_in__out = Scope(
        labels=["y_T2_in","y_T2_out"]
    )

    # Chemical
    divertor = Process(
        tau=tau_divertor
    )
    fw = Process(
        tau=tau_fw
    )
    blanket = Process(
        tau=tau_blanket,
        initial_value=1e-4
    )
    t_separation_membrane = Process(
        tau=tau_membrane
    )
    heat_exchanger = Process(
        tau=tau_hx
    )
    pump = Process(
        tau=tau_vacuum_pump
    )
    fuel_cleanup = Process(
        tau=tau_fuel_cleanup
    )
    iss = Process(
        tau=tau_iss
    )
    detritiation = Process(
        tau=tau_detritiation
    )
    hx_splitter = Splitter()
    detrit__storage = Splitter(
        fractions=[0.9 , 0.1]
    )
    storage__cleanup = Splitter(
        fractions=[dir_frac, 1 - dir_frac]
    )
    glc = GLC(
        BCs=BCs,
        T=T,
        D=D,
        L=L,
        P_in=P_in
    )

    blocks = [
        fusion_reaction_rate,
        n_1000__t_mol_mass_mol__kg,
        total_flow_l_kgs,
        n_1__rho_l_m3__kg,
        y_t2_in,
        flow_g_m3__s,
        t_mol_mass_kg__mol,
        storage,
        plasma_to_div,
        plasma_to_fw,
        x_tbr,
        injection_rate,
        pumping_rate,
        t_out_blanket_mol__s,
        q_l_m3__s,
        n_1__q_l_s__m3,
        c_t_in_mol__m3,
        n_t_out_liquid_kg__s,
        n_t_out_gas_kg__s,
        liquid_switch,
        gas_switch,
        adder,
        outer_fuel_cycle,
        fusion_rate,
        inner_fuel_cycle,
        bcr_eff,
        c_t_in__out,
        y_t2_in__out,
        divertor,
        fw,
        blanket,
        t_separation_membrane,
        heat_exchanger,
        pump,
        fuel_cleanup,
        iss,
        detritiation,
        hx_splitter,
        detrit__storage,
        storage__cleanup,
        glc,
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # CONNECTIONS
    # ────────────────────────────────────────────────────────────────────────────

    connections = [
        Connection(fusion_reaction_rate[0], plasma_to_div[0], plasma_to_fw[0], x_tbr[0], injection_rate[0], pumping_rate[0], fusion_rate[0]),
        Connection(plasma_to_div[0], divertor[0]),
        Connection(plasma_to_fw[0], fw[0]),
        Connection(x_tbr[0], blanket[0]),
        Connection(fw[1], blanket[1]),
        Connection(divertor[1], blanket[2]),
        Connection(pumping_rate[0], pump[0]),
        Connection(fuel_cleanup[1], iss[0]),
        Connection(detritiation[1], iss[1]),
        Connection(divertor[0], outer_fuel_cycle[0]),
        Connection(fw[0], outer_fuel_cycle[1]),
        Connection(blanket[0], outer_fuel_cycle[2]),
        Connection(pump[0], inner_fuel_cycle[1]),
        Connection(iss[0], inner_fuel_cycle[2]),
        Connection(fuel_cleanup[0], inner_fuel_cycle[3]),
        Connection(storage[0], inner_fuel_cycle[0]),
        Connection(iss[1], detrit__storage[0]),
        Connection(detrit__storage[1], detritiation[0]),
        Connection(storage__cleanup[1], fuel_cleanup[0]),
        Connection(pump[1], storage__cleanup[0]),
        Connection(blanket[1], t_out_blanket_mol__s[0], liquid_switch[0]),
        Connection(n_1000__t_mol_mass_mol__kg[0], t_out_blanket_mol__s[1]),
        Connection(q_l_m3__s[0], n_1__q_l_s__m3[0]),
        Connection(t_out_blanket_mol__s[0], c_t_in_mol__m3[0]),
        Connection(n_1__q_l_s__m3[0], c_t_in_mol__m3[1]),
        Connection(n_1__rho_l_m3__kg[0], q_l_m3__s[0]),
        Connection(total_flow_l_kgs[0], q_l_m3__s[1], glc[1]),
        Connection(c_t_in_mol__m3[0], glc[0], c_t_in__out[0]),
        Connection(y_t2_in[0], glc[2], y_t2_in__out[0], gas_switch[0]),
        Connection(flow_g_m3__s[0], glc[3]),
        Connection(t_mol_mass_kg__mol[0], n_t_out_liquid_kg__s[0], n_t_out_gas_kg__s[1]),
        Connection(glc[6], n_t_out_liquid_kg__s[1]),
        Connection(glc[7], n_t_out_gas_kg__s[0]),
        Connection(glc[2], bcr_eff[0]),
        Connection(glc[0], c_t_in__out[1]),
        Connection(glc[1], y_t2_in__out[1]),
        Connection(heat_exchanger[0], outer_fuel_cycle[3]),
        Connection(hx_splitter[0], divertor[1]),
        Connection(hx_splitter[1], fw[1]),
        Connection(hx_splitter[2], blanket[3]),
        Connection(n_t_out_gas_kg__s[0], gas_switch[1]),
        Connection(gas_switch[0], t_separation_membrane[0]),
        Connection(heat_exchanger[1], hx_splitter[0]),
        Connection(liquid_switch[0], heat_exchanger[0]),
        Connection(n_t_out_liquid_kg__s[0], liquid_switch[1]),
        Connection(detritiation[0], inner_fuel_cycle[4]),
        Connection(adder[0], storage[0]),
        Connection(storage__cleanup[0], adder[0]),
        Connection(injection_rate[0], adder[1]),
        Connection(t_separation_membrane[1], adder[2]),
        Connection(detrit__storage[0], adder[3]),
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # EVENTS
    # ────────────────────────────────────────────────────────────────────────────

    shutdown_event = ScheduleList(
        times_evt=[2 * 24 * 3600],
        func_act=act_sd
    )
    resume_event = ScheduleList(
        times_evt=[3 * 24 * 3600],
        func_act=act_su
    )

    events = [
        shutdown_event,
        resume_event,
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # SIMULATION
    # ────────────────────────────────────────────────────────────────────────────

    sim = Simulation(
        blocks,
        connections,
        events,
        Solver=RKCK54,
        dt=0.02,
        dt_min=1e-16,
        tolerance_lte_rel=0.0001,
        tolerance_lte_abs=1e-08,
        tolerance_fpi=1e-10,
    )

    # ────────────────────────────────────────────────────────────────────────────
    # MAIN
    # ────────────────────────────────────────────────────────────────────────────


    # Run simulation
    sim.run(duration)

    # ────────────────────────────────────────────────────────────────────────────
    ### SAVE RESULTS ###
    # ────────────────────────────────────────────────────────────────────────────

    import json

    results = sim.collect()
    scopes = results['scopes']

    # Create a structured dictionary for saving
    save_dict = {}
    scope_metadata = {}

    for scope_idx, (scope_id, scope_dict) in enumerate(scopes.items()):
        time = scope_dict['time']
        data = scope_dict['data']
        labels = scope_dict.get('labels', [])
        
        # Create scope name from labels or use generic name
        if labels and any(labels):  # If labels exist and aren't all empty
            scope_name = f"scope_{scope_idx}_{'_'.join(labels)}"
        else:
            scope_name = f"scope_{scope_idx}"
        
        # Clean up scope name to be filesystem-friendly
        scope_name = scope_name.replace(" ", "_").replace("-", "_")[:60]  # Limit length
        
        # Save time (only once per scope)
        save_dict[f"{scope_name}_time"] = time
        
        # Save data
        save_dict[f"{scope_name}_data"] = data
        
        # Store metadata for labels (JSON string since npz can't store lists directly)
        scope_metadata[scope_name] = {
            'labels': labels,
            'data_shape': data.shape,
            'time_length': len(time)
        }

    # Create results directory if it doesn't exist
    import os
    results_dir = '../../results'  # Go up to ARC modelling folder, then into results
    os.makedirs(results_dir, exist_ok=True)

    # Save the main data
    output_file = os.path.join(results_dir, 'ARC_single_bcr_shutdown_results.npz')
    np.savez(output_file, **save_dict)

    # Save metadata as JSON in the same directory
    import os
    metadata_file = os.path.join(os.path.dirname(output_file), 'ARC_single_bcr_shutdown_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(scope_metadata, f, indent=2)

    print(f"Results saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")

def arc_parallel_bcrs_shutdown(duration):
    ### Residence time model ###

    fusion_power = 525  # MWth
    tritium_burn_rate = 9.3e-7  # kg/s
    pulse_duration = 1800  # s
    time_between_pulses = 60  # s
    TBE = 0.02
    non_rad_loss_fraction = 1e-4
    AF = 0.7
    tritium_processing_time = 4 * 3600  # s
    dir_frac = 0.3

    tau_blanket = 1.25 * 3600  # s
    tau_fw = 1000  # s
    tau_divertor = 1000  # s
    tau_tes = 24 * 3600  # s
    tau_hx = 1000  # s
    tau_vacuum_pump = 600  # s
    tau_fuel_cleanup = 0.3 * 3600  # s
    tau_iss = 3 * 3600  # s
    tau_detritiation = 1 * 3600  # s
    tau_membrane = 100  # s

    f_p3 = 1e-4
    f_p4 = 1e-4

    TBR = 1.05
    tes_efficiency = 0.95
    startup_inventory = 1.14

    ### BCR model ###
    total_flow_l = 560 # kg / s
    total_flow_g = 0.19 # mol / s
    T = 623 # K
    D = 0.5 # m
    L = 3 # m
    P_in = 5e5 # Pa
    BCs = "C-C"
    T_mol_mass = 3.01605 # [g / mol]
    rho_l = 10.45e3 * (1 - 1.61e-4 * T) # kg / m3

    ### Shutdown / Startup events ###
    def act_sd(t):
        liquid_in_switch_1.select(0)
        liquid_in_switch_2.select(1)
        gas_in_switch_1.select(0)
        gas_in_switch_2.select(1)

    def act_su(t):
        liquid_in_switch_1.select(1)
        liquid_in_switch_2.select(0)
        gas_in_switch_1.select(1)
        gas_in_switch_2.select(0)

    # ────────────────────────────────────────────────────────────────────────────
    # BLOCKS
    # ────────────────────────────────────────────────────────────────────────────

    # Sources
    fusion_reaction_rate = PulseSource(
        T=pulse_duration,
        amplitude=tritium_burn_rate,
        duty=AF,
        t_rise=pulse_duration * 0.01,
        t_fall=pulse_duration * 0.01
    )
    n_1000__t_mol_mass_mol__kg = Constant(
        value=1000 / T_mol_mass
    )
    total_flow_l_kgs = Constant(
        value=total_flow_l
    )
    n_1__rho_l_m3__kg = Constant(
        value=1 / rho_l
    )
    y_t2_in = Constant(
        value=0
    )
    total_flow_g_mol__s = Constant(
        value=total_flow_g
    )
    t_mol_mass_kg__mol = Constant(
        value=T_mol_mass / 1000
    )
    n_1e4 = Constant(
        value=1e-4
    )
    block_8 = Constant(
        value=1e-4
    )

    # Dynamic
    storage = Integrator(
        initial_value=startup_inventory
    )

    # Algebraic
    plasma_to_div = Amplifier(
        gain=f_p4/TBE
    )
    plasma_to_fw = Amplifier(
        gain=f_p3/TBE
    )
    x_tbr = Amplifier(
        gain=TBR
    )
    injection_rate = Amplifier(
        gain=-1/TBE
    )
    pumping_rate = Amplifier(
        gain=(1 - TBE - f_p3 - f_p4) / TBE
    )
    t_out_blanket_mol__s = Multiplier()
    q_l_m3__s = Multiplier()
    n_1__q_l_s__m3 = Pow(
        exponent=-1
    )
    c_t_in_mol__m3 = Multiplier()
    n_t_out_liquid_kg__s = Multiplier()
    n_t_out_gas_kg__s = Multiplier()
    liquid_in_switch_1 = Switch(
        state=1
    )
    liquid_in_switch_2 = Switch(
        state=0
    )
    gas_in_switch_1 = Switch(
        state=1
    )
    gas_in_switch_2 = Switch(
        state=0
    )
    combine_gas = Adder()
    combine_liquid = Adder()
    adder = Adder()

    # Recording
    outer_fuel_cycle = Scope(
        labels=["Divertor","FW","Blanket","HX"]
    )
    fusion_rate = Scope(
        labels=['Fusion rate']
    )
    inner_fuel_cycle = Scope(
        labels=["Storage","Pump","ISS","Cleanup","Detritiation"]
    )
    bcr_eff = Scope(
        labels=["BCR_1", "BCR_2"]
    )
    c_t_in__out = Scope(
        labels=["c_T_in","c_T_out_BCR_1","c_T_out_BCR_2"]
    )
    y_t2_out = Scope(
        labels=["y_T2_out_BCR_1","y_T2_out_BCR_2"]
    )

    # Chemical
    divertor = Process(
        tau=tau_divertor
    )
    fw = Process(
        tau=tau_fw
    )
    blanket = Process(
        tau=tau_blanket,
        initial_value=1e-4
    )
    t_separation_membrane = Process(
        tau=tau_membrane
    )
    heat_exchanger = Process(
        tau=tau_hx
    )
    pump = Process(
        tau=tau_vacuum_pump
    )
    fuel_cleanup = Process(
        tau=tau_fuel_cleanup
    )
    iss = Process(
        tau=tau_iss
    )
    detritiation = Process(
        tau=tau_detritiation
    )
    hx_splitter = Splitter()
    detrit__storage = Splitter(
        fractions=[0.9 , 0.1]
    )
    storage__cleanup = Splitter(
        fractions=[dir_frac, 1 - dir_frac]
    )
    bcr_1 = GLC(
        BCs=BCs,
        T=T,
        D=D,
        L=L,
        P_in=P_in
    )
    bcr_2 = GLC(
        BCs=BCs,
        T=T,
        D=D,
        L=L,
        P_in=P_in
    )
    liquid_splitter = Splitter(
        fractions=[0.5,0.5]
    )
    gas_splitter = Splitter(
        fractions=[0.5,0.5]
    )

    blocks = [
        fusion_reaction_rate,
        n_1000__t_mol_mass_mol__kg,
        total_flow_l_kgs,
        n_1__rho_l_m3__kg,
        y_t2_in,
        total_flow_g_mol__s,
        t_mol_mass_kg__mol,
        n_1e4,
        block_8,
        storage,
        plasma_to_div,
        plasma_to_fw,
        x_tbr,
        injection_rate,
        pumping_rate,
        t_out_blanket_mol__s,
        q_l_m3__s,
        n_1__q_l_s__m3,
        c_t_in_mol__m3,
        n_t_out_liquid_kg__s,
        n_t_out_gas_kg__s,
        liquid_in_switch_1,
        liquid_in_switch_2,
        gas_in_switch_1,
        gas_in_switch_2,
        combine_gas,
        combine_liquid,
        adder,
        outer_fuel_cycle,
        fusion_rate,
        inner_fuel_cycle,
        bcr_eff,
        c_t_in__out,
        y_t2_out,
        divertor,
        fw,
        blanket,
        t_separation_membrane,
        heat_exchanger,
        pump,
        fuel_cleanup,
        iss,
        detritiation,
        hx_splitter,
        detrit__storage,
        storage__cleanup,
        bcr_1,
        bcr_2,
        liquid_splitter,
        gas_splitter,
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # CONNECTIONS
    # ────────────────────────────────────────────────────────────────────────────

    connections = [
        Connection(fusion_reaction_rate[0], plasma_to_div[0], plasma_to_fw[0], x_tbr[0], injection_rate[0], pumping_rate[0], fusion_rate[0]),
        Connection(plasma_to_div[0], divertor[0]),
        Connection(plasma_to_fw[0], fw[0]),
        Connection(x_tbr[0], blanket[0]),
        Connection(fw[1], blanket[1]),
        Connection(divertor[1], blanket[2]),
        Connection(pumping_rate[0], pump[0]),
        Connection(fuel_cleanup[1], iss[0]),
        Connection(detritiation[1], iss[1]),
        Connection(divertor[0], outer_fuel_cycle[0]),
        Connection(fw[0], outer_fuel_cycle[1]),
        Connection(blanket[0], outer_fuel_cycle[2]),
        Connection(pump[0], inner_fuel_cycle[1]),
        Connection(iss[0], inner_fuel_cycle[2]),
        Connection(fuel_cleanup[0], inner_fuel_cycle[3]),
        Connection(storage[0], inner_fuel_cycle[0]),
        Connection(iss[1], detrit__storage[0]),
        Connection(detrit__storage[1], detritiation[0]),
        Connection(storage__cleanup[1], fuel_cleanup[0]),
        Connection(pump[1], storage__cleanup[0]),
        Connection(blanket[1], t_out_blanket_mol__s[0]),
        Connection(n_1000__t_mol_mass_mol__kg[0], t_out_blanket_mol__s[1]),
        Connection(q_l_m3__s[0], n_1__q_l_s__m3[0]),
        Connection(t_out_blanket_mol__s[0], c_t_in_mol__m3[0]),
        Connection(n_1__q_l_s__m3[0], c_t_in_mol__m3[1]),
        Connection(n_1__rho_l_m3__kg[0], q_l_m3__s[0]),
        Connection(total_flow_l_kgs[0], q_l_m3__s[1], liquid_splitter[0], liquid_in_switch_1[0]),
        Connection(c_t_in_mol__m3[0], bcr_1[0], c_t_in__out[0], bcr_2[0]),
        Connection(y_t2_in[0], bcr_1[2], bcr_2[2]),
        Connection(t_mol_mass_kg__mol[0], n_t_out_gas_kg__s[1], n_t_out_liquid_kg__s[1]),
        Connection(heat_exchanger[0], outer_fuel_cycle[3]),
        Connection(hx_splitter[0], divertor[1]),
        Connection(hx_splitter[1], fw[1]),
        Connection(hx_splitter[2], blanket[3]),
        Connection(heat_exchanger[1], hx_splitter[0]),
        Connection(detritiation[0], inner_fuel_cycle[4]),
        Connection(liquid_splitter[0], liquid_in_switch_1[1]),
        Connection(liquid_in_switch_1[0], bcr_1[1]),
        Connection(liquid_in_switch_2[0], bcr_2[1]),
        Connection(n_1e4[0], liquid_in_switch_2[1]),
        Connection(liquid_splitter[1], liquid_in_switch_2[0]),
        Connection(total_flow_g_mol__s[0], gas_splitter[0], gas_in_switch_1[0]),
        Connection(gas_splitter[0], gas_in_switch_1[1]),
        Connection(gas_splitter[1], gas_in_switch_2[0]),
        Connection(block_8[0], gas_in_switch_2[1]),
        Connection(gas_in_switch_1[0], bcr_1[3]),
        Connection(gas_in_switch_2[0], bcr_2[3]),
        Connection(bcr_1[7], combine_gas[1]),
        Connection(bcr_2[7], combine_gas[0]),
        Connection(combine_gas[0], n_t_out_gas_kg__s[0]),
        Connection(bcr_1[6], combine_liquid[0]),
        Connection(bcr_2[6], combine_liquid[1]),
        Connection(combine_liquid[0], n_t_out_liquid_kg__s[0]),
        Connection(n_t_out_gas_kg__s[0], t_separation_membrane[0]),
        Connection(n_t_out_liquid_kg__s[0], heat_exchanger[0]),
        Connection(bcr_1[0], c_t_in__out[1]),
        Connection(bcr_2[0], c_t_in__out[2]),
        Connection(bcr_1[1], y_t2_out[0]),
        Connection(bcr_2[1], y_t2_out[1]),
        Connection(bcr_1[2], bcr_eff[0]),
        Connection(bcr_2[2], bcr_eff[1]),
        Connection(adder[0], storage[0]),
        Connection(storage__cleanup[0], adder[0]),
        Connection(injection_rate[0], adder[1]),
        Connection(t_separation_membrane[1], adder[2]),
        Connection(detrit__storage[0], adder[3]),
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # EVENTS
    # ────────────────────────────────────────────────────────────────────────────

    shutdown_event = ScheduleList(
        times_evt=[2 * 24 * 3600],
        func_act=act_sd
    )
    resume_event = ScheduleList(
        times_evt=[3 * 24 * 3600],
        func_act=act_su
    )

    events = [
        shutdown_event,
        resume_event,
    ]

    # ────────────────────────────────────────────────────────────────────────────
    # SIMULATION
    # ────────────────────────────────────────────────────────────────────────────

    sim = Simulation(
        blocks,
        connections,
        events,
        Solver=RKCK54,
        dt=0.02,
        dt_min=1e-16,
        tolerance_lte_rel=0.0001,
        tolerance_lte_abs=1e-08,
        tolerance_fpi=1e-10,
    )
    # ────────────────────────────────────────────────────────────────────────────
    # MAIN
    # ────────────────────────────────────────────────────────────────────────────

    # Run simulation
    sim.run(duration)

    # ────────────────────────────────────────────────────────────────────────────
    ### SAVE RESULTS ###
    # ────────────────────────────────────────────────────────────────────────────

    import json

    results = sim.collect()
    scopes = results['scopes']

    # Create a structured dictionary for saving
    save_dict = {}
    scope_metadata = {}

    for scope_idx, (scope_id, scope_dict) in enumerate(scopes.items()):
        time = scope_dict['time']
        data = scope_dict['data']
        labels = scope_dict.get('labels', [])
        
        # Create scope name from labels or use generic name
        if labels and any(labels):  # If labels exist and aren't all empty
            scope_name = f"scope_{scope_idx}_{'_'.join(labels)}"
        else:
            scope_name = f"scope_{scope_idx}"
        
        # Clean up scope name to be filesystem-friendly
        scope_name = scope_name.replace(" ", "_").replace("-", "_")[:60]  # Limit length
        
        # Save time (only once per scope)
        save_dict[f"{scope_name}_time"] = time
        
        # Save data
        save_dict[f"{scope_name}_data"] = data
        
        # Store metadata for labels (JSON string since npz can't store lists directly)
        scope_metadata[scope_name] = {
            'labels': labels,
            'data_shape': data.shape,
            'time_length': len(time)
        }

    # Create results directory if it doesn't exist
    import os
    results_dir = '../../results'  # Go up to ARC modelling folder, then into results
    os.makedirs(results_dir, exist_ok=True)

    # Save the main data
    output_file = os.path.join(results_dir, 'ARC_parallel_bcrs_shutdown_results.npz')
    np.savez(output_file, **save_dict)

    # Save metadata as JSON in the same directory
    import os
    metadata_file = os.path.join(os.path.dirname(output_file), 'ARC_parallel_bcrs_shutdown_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(scope_metadata, f, indent=2)

    print(f"Results saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")