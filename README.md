# RotateAroundSolution simulation comparison

Review media for the one-line marker-yaw bounds fix. This branch contains media only and is not intended for merging.

The left image/video uses the historical CAP rotation function with the original syringe scene assets, seed 42, and a -120-degree marker. Placement reports success despite 6.745 mm of initial syringe-to-syringe penetration.

The right uses corrected rotation bounds AND the existing MESH collision mode for the syringes. Placement succeeds with zero initial syringe-to-syringe contacts. The mesh scene configuration is not part of the code PR; corrected AABB bounds alone did not find a valid layout in this crowded tray.

This is an isolated replay using current placement code around the historical rotation function. It omits the robot and policy and uses a compatible Newton runtime, not the complete historical runtime. The video records two simulation seconds at half speed, with an initial one-second hold. Colors aid visibility. Both runs remain finite and objects move during settling. The reported high-speed ejection was not reproduced.
