
#MD


!pip install openmm OpenMM-CUDA-12 MDAnalysis plotly pandas numpy
from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout



inpcrd=AmberInpcrdFile("/kaggle/input/datasets/kirushi/reference-complex/reference_complex.inpcrd")
prmtop=AmberPrmtopFile("/kaggle/input/datasets/kirushi/reference-complex/reference_complex.prmtop")

#system=prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds,    implicitSolvent=OBC2,    soluteDielectric=1.0,   solventDielectric=78.5)


system=prmtop.createSystem(nonbondedMethod=PME,nonbondedCutoff=1.0*nanometer, constraints=HBonds)
integrator=LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
platform = Platform.getPlatformByName('CUDA')
platformProperties = {
    'Precision': 'mixed',
    'DeviceIndex': '0,1'  # Tells OpenMM to use both T4 GPUs!
}
simulation=Simulation(prmtop.topology, system, integrator,platform,platformProperties)

# Line 15: Sets the atom coordinates
simulation.context.setPositions(inpcrd.positions)



simulation.minimizeEnergy(tolerance=0.1*kilojoule/mole/nanometer, maxIterations=50000)
print("Minimization done", flush=True)

# check energy is finite before continuing
state = simulation.context.getState(getEnergy=True)
print(f"Post-minimization PE: {state.getPotentialEnergy()}", flush=True)


simulation.reporters.append(PDBReporter("reference_100_100_ns.pdb", 100000))
simulation.reporters.append(DCDReporter("reserence_DCD_100ns.dcd",100000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, time=True, potentialEnergy=True, temperature=True, speed=True))
simulation.context.setVelocitiesToTemperature(300*kelvin)



simulation.step(50000000)
print("Production done", flush=True)






#Analysis


!pip install MDAnalysis plotly pandas numpy

import os
import csv
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
import plotly.graph_objects as go

# =====================================================================
# 0. FILE PATHS & UNIVERSES
# =====================================================================
pdb_trajectory_1="/kaggle/input/datasets/kirushi/100-ns-md/100_ns_DCD.dcd"
pdb_trajectory_2 = "/kaggle/input/datasets/kirushi/100-200-ns-md/100_200_ns_trajectory.dcd"
prmtop_path = "/kaggle/input/datasets/kirushi/100-ns-md/complex_pro_pep.prmtop"

u = mda.Universe(prmtop_path, [pdb_trajectory_1,pdb_trajectory_2])
ref = mda.Universe(prmtop_path, [pdb_trajectory_1,pdb_trajectory_2])  # Reference frame 0

# Determine Ligand Residue Name (defaults to UNL, falls back to LIG)
ligand_sel = "resname UNL"
if len(u.select_atoms(ligand_sel)) == 0:
    ligand_sel = "protein and resid 167-171"

print(f"Ligand selection: {ligand_sel} ({len(u.select_atoms(ligand_sel))} atoms found)")


# =====================================================================
# 1. LIGAND RMSD
# =====================================================================
print("Calculating Ligand RMSD...")

# RMSD calculation using selection string
R_ligand = rms.RMSD(u, ref, select=ligand_sel, ref_frame=0).run()

frames = R_ligand.results.rmsd[:, 1]/1000.0
ligand_rmsd = R_ligand.results.rmsd[:, 2]

# Save to CSV
df_ligand = pd.DataFrame({'Frame': frames, 'Ligand_RMSD_A': ligand_rmsd})
df_ligand.to_csv("ligand_rmsd.csv", index=False)

# Plot Ligand RMSD
fig_ligand = go.Figure()
fig_ligand.add_trace(go.Scatter(
    x=frames, y=ligand_rmsd, mode='lines',
    line=dict(color='orange', width=2), name='Ligand RMSD'
))
fig_ligand.update_layout(
    title='Ligand RMSD over Time',
    xaxis_title='ns',
    yaxis_title='RMSD (Å)',
    template='plotly_white'
)
fig_ligand.show()
fig_ligand.write_html("ligand_rmsd_0_200_ns.html")


R_protein = rms.RMSD(u, ref, select="protein and backbone and not resid 167-171", ref_frame=0).run()


protein_rmsd = R_protein.results.rmsd[:, 2]

# Save to CSV
df_protein = pd.DataFrame({'Frame': frames, 'Protein_RMSD_A': protein_rmsd})
df_protein.to_csv("protein_rmsd.csv", index=False)

# Plot Ligand RMSD
fig_protein = go.Figure()
fig_protein.add_trace(go.Scatter(
    x=frames, y=protein_rmsd, mode='lines',
    line=dict(color='orange', width=2), name='Portein RMSD'
))
fig_protein.update_layout(
    title='Protein RMSD over Time',
    xaxis_title='ns',
    yaxis_title='RMSD (Å)',
    template='plotly_white'
)
fig_protein.show()
fig_protein.write_html("protein_rmsd_0_200_ns.html")


# =====================================================================
# 2. HYDROGEN BOND ANALYSIS (Peptide resid 1-5 <-> Target)
# =====================================================================
print("Running Hydrogen Bond Analysis...")

hbonds = HBA(
    universe=u,
    between=['protein and not resid 167-171', 'protein and resid 167-171'],
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=150
)
hbonds.run()

all_hbond_data = []
total_frames = len(u.trajectory)
hbond_count = np.zeros(total_frames)

for bond in hbonds.results.hbonds:
    frame_idx = int(bond[0])
    donor_atom = u.atoms[int(bond[1])]
    acceptor_atom = u.atoms[int(bond[3])]
    distance = bond[4]
    angle = bond[5]
    
    hbond_count[frame_idx] += 1
    all_hbond_data.append([
        frame_idx,
        f"{donor_atom.resname}{donor_atom.resid}",
        donor_atom.name,
        f"{acceptor_atom.resname}{acceptor_atom.resid}",
        acceptor_atom.name,
        round(distance, 3),
        round(angle, 3)
    ])

# Save Detailed H-bond CSV
with open("all_frames_hbonds.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "donor_res", "donor_atom", "acceptor_res", "acceptor_atom", "distance_a", "angle_deg"])
    writer.writerows(all_hbond_data)

# Save Frame-by-Frame Count CSV
with open("hbond_count.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "count"])
    for i in range(total_frames):
        writer.writerow([i, int(hbond_count[i])])

# Plot H-Bonds
df_hb = pd.read_csv("hbond_count.csv")
fig_hb = go.Figure()
fig_hb.add_trace(go.Scatter(
    x=df_hb["frame"], y=df_hb["count"], mode="lines+markers",
    name="H-bonds", marker=dict(size=6),
    line=dict(color="firebrick", width=2), line_shape="hv"
))
fig_hb.update_layout(
    title="Hydrogen Bond Analysis (Peptide [resid 1-5] <-> Target Protein)",
    xaxis_title="Frame",
    yaxis_title="Count of H-bonds",
    template="plotly_white",
    hovermode="x unified"
)
fig_hb.show()
fig_hb.write_html("h-bond_analysis_0_200_ns.html")

print("Finished! Output files created: ligand_rmsd.csv, ligand_rmsd.html, all_frames_hbonds.csv, hbond_count.csv, h-bond_analysis.html")
!pip install MDAnalysis pandas plotly 



import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import plotly.graph_objects as go


u_complex=mda.Universe("/kaggle/input/datasets/kirushi/100-ns-md/complex_pro_pep.prmtop",["/kaggle/input/datasets/kirushi/100-ns-md/100_ns_DCD.dcd","/kaggle/input/datasets/kirushi/100-200-ns-md/100_200_ns_trajectory.dcd"])
calphas = u_complex.select_atoms("protein and name CA and not resid 167-171")

print("Calculating RMSF...")
rmsf_analysis = rms.RMSF(calphas).run()
res_ids = calphas.resids
rmsf_values = rmsf_analysis.results.rmsf

df_rmsf = pd.DataFrame({
    'Residue_Number': res_ids,
    'RMSF_Angstrom': rmsf_values
})
df_rmsf.to_csv("rmsf_analysis.csv", index=False)

# --- Plot 5: RMSF Graph ---
fig_rmsf = go.Figure()
fig_rmsf.add_trace(go.Scatter(
    x=res_ids, y=rmsf_values, mode='lines+markers',
    marker=dict(size=4), line=dict(color='teal', width=2),
    name='C-alpha RMSF'
))
fig_rmsf.update_layout(
    title="Per-Residue Root-Mean-Square Fluctuation (RMSF)",
    xaxis_title="Residue Index / Number",
    yaxis_title="RMSF (Å)",
    template="plotly_white",
    hovermode="x unified"
)
fig_rmsf.show()
fig_rmsf.write_html("rmsf_analysis_0_200_ns.html")
print("All tasks executed and figures saved successfully!")

backbone=u.select_atoms("protein and name CA and not resid 167-171")
rg_values = [backbone.radius_of_gyration() for ts in u.trajectory]

# --- Plot 2: Radius of Gyration Graph ---
fig_rg = go.Figure()
fig_rg.add_trace(go.Scatter(
    x=frames, 
    y=rg_values, 
    mode='lines', 
    line=dict(color='navy', width=2),
    name='Rg'
))
fig_rg.update_layout(
    title="Radius of Gyration ($R_g$) Timeline",
    xaxis_title="ns",
    yaxis_title="Radius of Gyration ($\AA$)",
    template="plotly_white"
)
fig_rg.show()

fig_rg.write_html("Rg_plot_0_200_ns.html")
