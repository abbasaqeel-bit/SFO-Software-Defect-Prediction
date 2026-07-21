from functools import partial

from mealpy import (
    PSO,
    GWO,
    DE,
    GA,
    ACOR,
    WOA,
    HHO,
    HBA,
    AGTO,
    MGO,
    SeaHO,
    CoatiOA,
    SCSO,
)

from algorithms.sfo import SFO
from algorithms.mealpy_adapter import MealpyOptimizerAdapter


SFO_PARAMETERS = {}

def mealpy_adapter(optimizer_class):
    return partial(
        MealpyOptimizerAdapter,
        mealpy_optimizer_class=optimizer_class,
    )



OPTIMIZERS = {


    "SFO": {
        "class": SFO,
        "parameters": SFO_PARAMETERS,
        "source": "Local implementation of original SFO",
        "category": "Proposed optimizer",
    },


    "PSO": {
        "class": mealpy_adapter(
            PSO.OriginalPSO
        ),
        "parameters": {
            "mealpy_parameters": {
                "c1": 2.05,
                "c2": 2.05,
                "w": 0.4,
            }
        },
        "source": "MEALPY 3.0.3 - OriginalPSO",
        "category": "Classical swarm-based",
    },

    "GWO": {
        "class": mealpy_adapter(
            GWO.OriginalGWO
        ),
        "parameters": {
            "mealpy_parameters": {}
        },
        "source": "MEALPY 3.0.3 - OriginalGWO",
        "category": "Classical swarm-based",
    },

    "DE": {
        "class": mealpy_adapter(
            DE.OriginalDE
        ),
        "parameters": {
            "mealpy_parameters": {
                "wf": 0.1,
                "cr": 0.9,
                "strategy": 0,
            }
        },
        "source": "MEALPY 3.0.3 - OriginalDE",
        "category": "Evolution-based",
    },

    "GA": {
        "class": mealpy_adapter(
            GA.BaseGA
        ),
        "parameters": {
            "mealpy_parameters": {
                "pc": 0.95,
                "pm": 0.025,
            }
        },
        "source": "MEALPY 3.0.3 - BaseGA",
        "category": "Evolution-based",
    },

    "ACOR": {
        "class": mealpy_adapter(
            ACOR.OriginalACOR
        ),
        "parameters": {
            "mealpy_parameters": {
                "sample_count": 10,
                "intent_factor": 0.5,
                "zeta": 1.0,
            }
        },
        "source": "MEALPY 3.0.3 - OriginalACOR",
        "category": "Ant-colony-based",
    },


    "WOA": {
        "class": mealpy_adapter(
            WOA.OriginalWOA
        ),
        "parameters": {
            "mealpy_parameters": {}
        },
        "source": "MEALPY 3.0.3 - OriginalWOA",
        "category": "Swarm-based",
    },

    "HHO": {
        "class": mealpy_adapter(
            HHO.OriginalHHO
        ),
        "parameters": {
            "mealpy_parameters": {}
        },
        "source": "MEALPY 3.0.3 - OriginalHHO",
        "category": "Swarm-based",
    },


    "HBA": {
        "class": mealpy_adapter(
            HBA.OriginalHBA
        ),
        "parameters": {
            "mealpy_parameters": {}
        },
        "source": "MEALPY 3.0.3 - OriginalHBA",
        "category": "Recent swarm-based",
    },

    "AGTO": {
        "class": mealpy_adapter(
            AGTO.OriginalAGTO
        ),
        "parameters": {
            "mealpy_parameters": {
                "p1": 0.03,
                "p2": 0.8,
                "beta": 3.0,
            }
        },
        "source": "MEALPY 3.0.3 - OriginalAGTO",
        "category": "Recent swarm-based",
    },

    "MGO": {
        "class": mealpy_adapter(
            MGO.OriginalMGO
        ),
        "parameters": {
            "mealpy_parameters": {}
        },
        "source": "MEALPY 3.0.3 - OriginalMGO",
        "category": "Recent swarm-based",
    },

    "SeaHO": {
        "class": mealpy_adapter(
            SeaHO.OriginalSeaHO
        ),
        "parameters": {
            "mealpy_parameters": {}
        },
        "source": "MEALPY 3.0.3 - OriginalSeaHO",
        "category": "Recent swarm-based",
    },

    "CoatiOA": {
        "class": mealpy_adapter(
            CoatiOA.OriginalCoatiOA
        ),
        "parameters": {
            "mealpy_parameters": {}
        },
        "source": "MEALPY 3.0.3 - OriginalCoatiOA",
        "category": "Recent swarm-based",
    },

    "SCSO": {
        "class": mealpy_adapter(
            SCSO.OriginalSCSO
        ),
        "parameters": {
            "mealpy_parameters": {}
        },
        "source": "MEALPY 3.0.3 - OriginalSCSO",
        "category": "Recent swarm-based",
    },
}
