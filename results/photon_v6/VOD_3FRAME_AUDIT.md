# View-of-Delft (VoD) 3-Frame Radar Temporal Accumulation Audit

- **Directory**: `view_of_delft_PUBLIC/radar_3frames/training/velodyne/`
- **Files**: `8,682` `.bin` files matching `00000.bin .. 08681.bin`
- **Frame Composition**: Rigidly motion-compensated accumulation of scans $[t-2, t-1, t]$
- **Coordinate Reference**: Target frame $t$ vehicle/radar coordinate system
- **Relative Time IDs**: $\{-2.0, -1.0, 0.0\}$
- **Point Density**: $\approx 3\times$ single-scan point density ($\sim 900-1200$ points/scan)
