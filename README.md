# RotateAroundSolution comparison image

Review image for the marker-yaw collision-bounds fix. This media branch is not intended for merging.

Left: historical CAP rotation function, original syringe assets, seed 42, and a -120-degree marker. Placement reports success despite 6.745 mm of initial syringe-to-syringe penetration.

Right: corrected bounds AND existing MESH collision mode for the syringes. Placement succeeds with zero initial syringe-to-syringe contacts. That mesh configuration is separate from the code PR; corrected AABB bounds alone did not find a valid layout in this crowded tray.

The isolated replay uses current surrounding placement code and a compatible Newton runtime. It omits the robot and policy. Colors aid visibility. It is not a reproduction of the complete historical runtime or the reported high-speed ejection. Objects still move during settling.
