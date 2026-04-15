# Toribash TODO

## Physics Issues

- [ ] **Body parts clip into ground or opponent** — `physics/`
  - Segments sometimes get stuck inside the ground plane or embedded in opponent's body
  - Likely needs higher solver iterations, collision margin tuning, or position correction

## Code Quality

- [ ] **Flaky test_physics gravity test** — `tests/test_physics.py`
  - `test_gravity` intermittently fails: chest can briefly move upward due to joint motor forces
  - Needs a more tolerant assertion or a longer simulation before checking
