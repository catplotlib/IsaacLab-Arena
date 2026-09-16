# Rotation collision comparison

Review media for the marker-yaw collision-bounds fix. This branch is not intended for merging.

Left: incorrect bounds allow an overlapping initial placement.
Right: corrected bounds with mesh collision checking produce no initial overlap.
The right-hand mesh configuration is separate from the code PR. This image illustrates initial placement only.
