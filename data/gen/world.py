"""Fictional aerospace supply base: suppliers, programs, parts, materials.

All company and program names are invented. Industry standards (NAS, MS,
AMS, MIL-DTL) are public specifications and are used for realism only.
"""

SUPPLIERS = [
    # name, short, city, country, region
    ("Kestrel Forge Inc.", "Kestrel", "Cleveland, OH", "USA", "north_america"),
    ("Tri-Valley Fastener Corp.", "Tri-Valley", "Torrance, CA", "USA", "north_america"),
    ("Northgate Aerostructures LLC", "Northgate", "Wichita, KS", "USA", "north_america"),
    ("Cobalt Ridge Alloys", "Cobalt Ridge", "Pittsburgh, PA", "USA", "north_america"),
    ("Lakeshore Avionics", "Lakeshore", "Milwaukee, WI", "USA", "north_america"),
    ("Sierra Composite Works", "Sierra", "Ogden, UT", "USA", "north_america"),
    ("Ironwood Hydraulics", "Ironwood", "Spokane, WA", "USA", "north_america"),
    ("Redstone Harness Systems", "Redstone", "Huntsville, AL", "USA", "north_america"),
    ("Gulfstar Coatings", "Gulfstar", "Baton Rouge, LA", "USA", "north_america"),
    ("Maple Leaf Precision", "Maple Leaf", "Montreal, QC", "Canada", "north_america"),
    ("Bajio Aeromachining", "Bajio", "Queretaro", "Mexico", "north_america"),
    ("Halvorsen Precision Forge AS", "Halvorsen", "Kongsberg", "Norway", "europe"),
    ("Vexin Aero Usinage", "Vexin", "Toulouse", "France", "europe"),
    ("Brennerwerk Luftfahrttechnik GmbH", "Brennerwerk", "Bremen", "Germany", "europe"),
    ("Castellane Composites SA", "Castellane", "Nantes", "France", "europe"),
    ("Pennine Wire Systems Ltd", "Pennine", "Derby", "UK", "europe"),
    ("Valsusa Aerospace Bearings S.p.A.", "Valsusa", "Turin", "Italy", "europe"),
    ("Vistula Avionics Sp. z o.o.", "Vistula", "Rzeszow", "Poland", "europe"),
    ("Ebro Fasteners SL", "Ebro", "Getafe", "Spain", "europe"),
    ("Moravia Castings s.r.o.", "Moravia", "Brno", "Czech Republic", "europe"),
    ("Shinwa Aero Components", "Shinwa", "Nagoya", "Japan", "asia_pacific"),
    ("Hanbit Precision Co.", "Hanbit", "Changwon", "South Korea", "asia_pacific"),
    ("Pune Aerotech Pvt Ltd", "Pune Aerotech", "Pune", "India", "asia_pacific"),
    ("Hengtai Aviation Parts Co.", "Hengtai", "Xi'an", "China", "asia_pacific"),
    ("Selangor Aerocomposites Sdn Bhd", "Selangor Aero", "Shah Alam", "Malaysia", "asia_pacific"),
    ("Haiphong Wire Harness JSC", "Haiphong WH", "Hai Phong", "Vietnam", "asia_pacific"),
    ("Taichung Electro Components", "Taichung EC", "Taichung", "Taiwan", "asia_pacific"),
    ("Brisbane Aero Machining Pty", "Brisbane Aero", "Brisbane", "Australia", "asia_pacific"),
    ("Kaveri Avionics Systems", "Kaveri", "Bengaluru", "India", "asia_pacific"),
]

BROKERS = [
    "Apex Component Brokers", "GlobalChip Surplus", "Eastbridge Trading Ltd",
    "QuickSource Electronics (independent distributor)", "Pacific Rim Excess Inventory",
]

PROGRAMS = [
    # name, description, military
    ("NB-700", "single-aisle commercial airliner", False),
    ("WB-950", "twin-aisle commercial airliner", False),
    ("RJ-90", "regional jet", False),
    ("BJ-6000", "large-cabin business jet", False),
    ("HX-9", "commercial medium helicopter", False),
    ("MT-3", "military jet trainer", True),
    ("VTX-12", "military tiltrotor", True),
    ("AF-21", "fighter aircraft", True),
    ("RQ-30", "military ISR unmanned aircraft", True),
    ("SB-200", "defense satellite bus", True),
]

# commodity -> list of (part name, short name)
PARTS = {
    "castings_forgings": [
        ("Inconel 718 HPT disk forging", "718 disks"),
        ("Ti-6Al-4V main landing gear trunnion forging", "MLG trunnion forgings"),
        ("7050 aluminum wing rib forging", "rib forgings"),
        ("A357 aluminum investment-cast gearbox housing", "gearbox castings"),
        ("17-4PH investment-cast pylon fitting", "pylon castings"),
        ("titanium fan blade forging", "fan blade forgings"),
        ("nickel superalloy turbine case ring forging", "case rings"),
    ],
    "fasteners": [
        ("NAS6606 titanium hex bolts", "NAS6606 bolts"),
        ("MS21042 self-locking nuts", "MS21042 nuts"),
        ("Hi-Lok HL10 titanium pins", "Hi-Lok pins"),
        ("NAS1097 aluminum solid rivets", "rivets"),
        ("blind lockbolts, 3/16 in", "lockbolts"),
        ("A286 fuel-tank fasteners with sealing collars", "tank fasteners"),
    ],
    "electronics": [
        ("radiation-tolerant FPGA for flight control computer", "FPGAs"),
        ("ARINC 429 transceiver IC", "ARINC 429 ICs"),
        ("MIL-DTL-38999 circular connectors", "38999 connectors"),
        ("28V DC-DC converter module", "DC-DC converters"),
        ("cockpit display processor board", "display boards"),
        ("ceramic capacitor kits (MIL-PRF-55681)", "capacitors"),
        ("inertial measurement unit sensor board", "IMU boards"),
    ],
    "composites": [
        ("carbon fiber epoxy prepreg (IM-class, 350F cure)", "prepreg"),
        ("Nomex honeycomb core panels", "honeycomb core"),
        ("fiberglass nose radome", "radomes"),
        ("CFRP wing skin stringers", "stringers"),
        ("thermoplastic floor panels", "floor panels"),
        ("composite fan case", "fan cases"),
    ],
    "machined_parts": [
        ("titanium wing-to-body fitting", "WTB fittings"),
        ("aluminum fuselage frame segment", "frame segments"),
        ("7075 aluminum floor beam", "floor beams"),
        ("titanium flap track", "flap tracks"),
        ("steel engine mount bracket", "mount brackets"),
        ("aluminum avionics rack chassis", "rack chassis"),
    ],
    "raw_material": [
        ("Ti-6Al-4V plate, 2.0 in", "Ti plate"),
        ("7050-T7451 aluminum plate", "7050 plate"),
        ("Inconel 718 bar stock", "718 bar"),
        ("2024-T3 aluminum sheet", "2024 sheet"),
        ("15-5PH stainless bar", "15-5 bar"),
        ("7075 aluminum extrusions", "extrusions"),
    ],
    "harnesses_cables": [
        ("engine nacelle wire harness", "nacelle harnesses"),
        ("cockpit avionics harness assembly", "cockpit harnesses"),
        ("coaxial RF cable assemblies", "RF cables"),
        ("wing leading-edge harness", "LE harnesses"),
        ("cabin power distribution harness", "cabin harnesses"),
    ],
    "chemicals": [
        ("polysulfide fuel tank sealant (AMS-S-8802)", "tank sealant"),
        ("epoxy structural film adhesive", "film adhesive"),
        ("chromate-free epoxy primer", "primer"),
        ("polyurethane exterior topcoat", "topcoat"),
        ("phosphate-ester hydraulic fluid", "hydraulic fluid"),
        ("conformal coating for PCBs", "conformal coating"),
    ],
    "mechanical_assemblies": [
        ("primary flight control hydraulic actuator", "flight control actuators"),
        ("main landing gear retraction actuator", "MLG actuators"),
        ("fuel boost pump assembly", "boost pumps"),
        ("spherical plain bearings", "bearings"),
        ("hydraulic servo valve", "servo valves"),
        ("elastomeric seal kits for actuators", "seal kits"),
    ],
    "other": [
        ("cabin seat frames", "seat frames"),
        ("galley insert assemblies", "galley inserts"),
        ("lavatory module", "lav modules"),
        ("overhead stowage bin assemblies", "bins"),
        ("cabin trim panels", "trim panels"),
    ],
}

# commodity -> upstream inputs that can be short
MATERIALS = {
    "castings_forgings": ["Inconel 718 billet", "Ti-6Al-4V billet", "aluminum forging stock", "nickel superalloy ingot"],
    "fasteners": ["titanium wire rod", "A286 wire", "aluminum rod stock"],
    "electronics": ["FPGA die", "MLCC capacitors", "connector contacts", "PCB laminate"],
    "composites": ["carbon fiber tow", "epoxy resin system", "aramid paper for honeycomb"],
    "machined_parts": ["titanium plate", "7075 plate stock", "steel bar"],
    "raw_material": ["titanium sponge", "aluminum ingot", "nickel", "melt-shop capacity"],
    "harnesses_cables": ["MIL-spec wire", "connector backshells", "PTFE insulation"],
    "chemicals": ["polysulfide base polymer", "epoxy hardener", "specialty solvents"],
    "mechanical_assemblies": ["bearing-grade steel", "seal elastomer compound", "valve spool blanks"],
    "other": ["aluminum honeycomb", "fabric", "seat foam"],
}

PORTS = ["Port of Long Beach", "Rotterdam", "Hamburg", "Busan", "Shanghai", "Laredo border crossing",
         "Felixstowe", "Singapore", "Chennai", "Memphis air hub", "Leipzig air hub", "Montreal"]

DEFECTS = {
    "castings_forgings": ["surface cracks found by fluorescent penetrant inspection",
                          "grain flow nonconformance in macro-etch", "porosity beyond radiographic limits",
                          "hardness below spec after heat treat"],
    "fasteners": ["thread form nonconformance", "incorrect head marking dimensions",
                  "hydrogen embrittlement test failures", "coating thickness below spec"],
    "electronics": ["solder joint voids on BGA packages", "intermittent failures in burn-in",
                    "ESD damage on input stage", "wrong component revision populated"],
    "composites": ["delamination detected by ultrasonic inspection", "out-of-life prepreg used in layup",
                   "porosity in cured laminate", "fiber orientation outside tolerance"],
    "machined_parts": ["bore diameter out of tolerance by 0.002 in", "missing chamfer on fastener holes",
                       "surface finish exceeding Ra limit", "wrong anodize class applied"],
    "raw_material": ["chemistry out of spec on mill cert", "ultrasonic indications in plate",
                     "tensile results below minimum", "plate flatness out of tolerance"],
    "harnesses_cables": ["crimp pull-test failures", "wrong wire gauge in branch", "insulation damage",
                         "connector pin-out errors"],
    "chemicals": ["batch failed viscosity test", "cure time outside spec", "expired shelf life on delivered kits",
                  "contamination in containers"],
    "mechanical_assemblies": ["internal leakage above ATP limit", "seal extrusion during proof test",
                              "bearing race spalling", "servo valve null shift out of spec"],
    "other": ["flammability test failure on panels", "dimensional mismatch at seat track interface",
              "finish defects", "missing placards"],
}

COUNTERFEIT_SIGNS = [
    "inconsistent date codes and evidence of sanding and re-marking on the package tops",
    "no traceability back to the original component manufacturer",
    "a certificate of conformance that appears to be falsified",
    "lot markings that the original manufacturer says it never produced",
    "package weight and dimensions that do not match genuine parts",
    "X-ray showing a different die than the manufacturer's genuine part",
    "blacktopping and remarking found under the solvent test",
    "head markings that do not match the approved manufacturer's stamp",
]

DOC_ERRORS = [
    "a typo in the PO number on the certificate of conformance; corrected cert received",
    "the packing slip listed the wrong revision; supplier sent a corrected slip",
]

CHIPS = ["FPGA", "microcontroller", "DC-DC controller IC", "ARINC transceiver", "memory device"]

# supplier short name -> commodities it plausibly supplies
SUPPLIER_COMMODITIES = {
    "Kestrel": ["castings_forgings"], "Tri-Valley": ["fasteners"], "Northgate": ["machined_parts", "composites"],
    "Cobalt Ridge": ["raw_material", "castings_forgings"], "Lakeshore": ["electronics"],
    "Sierra": ["composites"], "Ironwood": ["mechanical_assemblies"], "Redstone": ["harnesses_cables"],
    "Gulfstar": ["chemicals"], "Maple Leaf": ["machined_parts", "mechanical_assemblies"],
    "Bajio": ["machined_parts", "harnesses_cables", "other"], "Halvorsen": ["castings_forgings"],
    "Vexin": ["machined_parts"], "Brennerwerk": ["mechanical_assemblies", "machined_parts"],
    "Castellane": ["composites", "other"], "Pennine": ["harnesses_cables"], "Valsusa": ["mechanical_assemblies"],
    "Vistula": ["electronics"], "Ebro": ["fasteners"], "Moravia": ["castings_forgings"],
    "Shinwa": ["machined_parts", "composites", "other"], "Hanbit": ["machined_parts", "raw_material"],
    "Pune Aerotech": ["machined_parts", "other"], "Hengtai": ["raw_material", "fasteners", "castings_forgings"],
    "Selangor Aero": ["composites"], "Haiphong WH": ["harnesses_cables"], "Taichung EC": ["electronics"],
    "Brisbane Aero": ["machined_parts"], "Kaveri": ["electronics", "harnesses_cables"],
}
# chemicals has only one natural supplier above; distributors cover the rest
SUPPLIER_COMMODITIES["Cobalt Ridge"].append("chemicals")
SUPPLIER_COMMODITIES["Castellane"].append("chemicals")


def suppliers_for(commodity):
    return [s for s in SUPPLIERS if commodity in SUPPLIER_COMMODITIES[s[1]]]
