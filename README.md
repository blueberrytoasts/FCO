# FCO: Immediate Readout In-Vivo Dosimetry using Film Cut Outs

A research project developing and validating the IR-FCO (Immediate-Read Film Cut Out) dosimetry protocol using a handheld point densitometer for rapid in-vivo radiation dose measurement.

---

## Project Overview

This project aims to establish a rapid, accurate dosimetry method using film cut outs (FCOs) read with a handheld densitometer within 30 minutes of irradiation. The IR-FCO protocol will be validated against standard flatbed scanner methods and compared with established TLD and OSLD dosimetry systems.

---

## Specific Aims

### Specific Aim 1: Protocol Development & Validation

**Goal:** Develop and validate the IR-FCO dosimetry protocol against a standard flatbed scanner protocol.

| Sub-Aim | Description |
|---------|-------------|
| **1a** | Investigate equivalency of calibration between IR-FCO and flatbed scanner; demonstrate reliable conversion between calibration methods |
| **1b** | Investigate longitudinal stability of densitometer sensitivity |
| **1c** | Determine uncertainty budget following AAPM TG-191 guidelines (as done for TLDs and OSLDs) |

---

### Specific Aim 2: Therapeutic Dose Verification

**Goal:** Compare accuracy and precision of IR-FCO with standard OSLD and TLD in-vivo dosimetry systems for prescribed therapeutic dose verification.

**Dose Range:** 100–1000 cGy per fraction

#### Sub-Aim 2a: Reference Conditions
- Controlled reference-field conditions (open square fields on water-equivalent phantom)
- Characterize and compare off-axis dose profiles
- Measure with and without buildup material
- Compare IR-FCO against TLD and OSLD systems

#### Sub-Aim 2b: Clinical Scenarios
Using anthropomorphic phantom geometry and realistic treatment plans, compare IR-FCO, OSLD, and TLD across four clinical-like scenarios:

| Scenario | Field Type | Bolus |
|----------|------------|-------|
| 1 | Photon | No bolus |
| 2 | Photon | With bolus |
| 3 | Electron | No bolus |
| 4 | Electron | With bolus (Superflab and custom machinable blue-wax) |

---

### Specific Aim 3: Low-Dose Out-of-Field Monitoring

**Goal:** Compare accuracy and precision of IR-FCO with standard OSLD and TLD in-vivo dosimetry systems for out-of-field, low-dose monitoring.

**Dose Range:** 0.1–100 cGy

**Guidance:** AAPM TG-158 and TG-203

#### Sub-Aim 3a: Reference Conditions
- Controlled reference-field conditions
- Characterize off-axis dose profile and low-dose tail
- Measure from high-gradient penumbra to tens of centimeters from central axis
- Compare IR-FCO, OSLD, and TLD

#### Sub-Aim 3b: Clinical Scenarios
Using anthropomorphic phantoms and realistic treatment plans, compare IR-FCO, OSLD, and TLD in two clinical-like scenarios:

| Scenario | Description |
|----------|-------------|
| 1 | Dose to an implanted device (e.g., pacemaker) located outside the primary field |
| 2 | Dose to the eye beneath a lead block during electron therapy |

---

## Key Terminology

| Term | Definition |
|------|------------|
| **FCO** | Film Cut Out |
| **IR-FCO** | Immediate-Read Film Cut Out |
| **TLD** | Thermoluminescent Dosimeter |
| **OSLD** | Optically Stimulated Luminescence Dosimeter |
| **AAPM** | American Association of Physicists in Medicine |
| **TG** | Task Group |

---

## Relevant AAPM Task Group Reports

- **TG-191:** Uncertainty analysis for TLD and OSLD dosimetry
- **TG-158:** Out-of-field dose measurement guidance
- **TG-203:** Low-dose monitoring guidance

---

## Project Structure

```
FCO/
├── README.md                 # Project overview (this file)
├── data/                     # Experimental data
│   ├── calibration/          # Calibration measurements
│   ├── reference/            # Reference condition data
│   └── clinical/             # Clinical scenario data
├── analysis/                 # Data analysis scripts
├── docs/                     # Documentation and protocols
└── results/                  # Processed results and figures
```

---

## License

*To be determined*

---

## Contact

*To be added*
