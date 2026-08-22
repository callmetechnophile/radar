# PhotonShield AI — Phase V7.5 Range-Conditioned Hybrid Domain Adapter

## Scientific Result: **VALIDATED**

## Key Numerical Summary

| Model | Params | No-Shift MPJPE | A-medium MPJPE | A-high MPJPE | PA-MPJPE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A: Static Linear | `36` | `61.6mm` | `1899.3mm` | `1980.1mm` | `26.8mm` |
| B: Static+ADALINE | `72` | `65.9mm` | `1282.8mm` | `1654.5mm` | `26.8mm` |
| C: Range Linear | `72` | `56.4mm` | `1954.6mm` | `2034.2mm` | `26.8mm` |
| D: Range+ADALINE | `108` | `60.6mm` | `1271.1mm` | `1618.3mm` | `26.8mm` |
| E: Range+Residual | `587` | `90.3mm` | `1901.8mm` | `1872.5mm` | `26.8mm` |
| **F: Full Hybrid** | **`623`** | **`92.6mm`** | **`1273.8mm`** | **`1597.6mm`** | **`26.8mm`** |

## Range Dependence
- Range-MPJPE correlation: `-0.024`
- Range conditioning: **NOT CONFIRMED**

## Compute Audit
| Component | Params | Extra Latency |
| :--- | :---: | :---: |
| Range-conditioned linear | `72` | `0.1420 ms` |
| Tiny residual | `515` | `0.3778 ms` |
| ADALINE | `36` | `0.0044 ms` |
| **Total new** | **`623`** | **`0.5242 ms`** |

## V7.5 Decision: **VALIDATED**
