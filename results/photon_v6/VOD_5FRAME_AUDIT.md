# View-of-Delft (VoD) 5-Frame Radar Temporal Accumulation Audit

- **Directory**: `view_of_delft_PUBLIC/radar_5frames/training/velodyne/`
- **Files**: `8,682` `.bin` files matching `00000.bin .. 08681.bin`
- **Frame Composition**: Rigidly motion-compensated accumulation of scans $[t-4, t-3, t-2, t-1, t]$
- **Coordinate Reference**: Target frame $t$ vehicle/radar coordinate system
- **Relative Time IDs**: $\{-4.0, -3.0, -2.0, -1.0, 0.0\}$
- **Point Density**: $\approx 5\times$ single-scan point density ($\sim 1500-2200$ points/scan)
