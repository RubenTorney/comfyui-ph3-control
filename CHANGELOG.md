# Changelog

## 0.4.0-experimental
- Final quality: control directly at ~4 MP (`ph3_pose_control_4mp_hands.json`); the 1 MP + 16 MP edit pass is removed (waxy skin).
- Hand skeletons (21 points): measured on a selfie and a two-handed diver; they place the hands, the prompt sets the gesture.
- New 4 MP gallery and hand comparison sheets.

## 0.3.0-experimental
- Marked experimental. Three example workflows (basic, 16 MP finishing, mannequin reference), method comparison, `scripts/build_release.sh --no-table`.
- 0.2.0: 2-frame sampling, first-frame residual, target-only correction, steps as fractions, 3D puppet, Light / Normal / Strict presets.
- 0.1.0: first release.
