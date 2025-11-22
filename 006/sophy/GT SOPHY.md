# Gran Turismo Sophy AI Observation space

- 3D Velocity
- 3D Angular velocity
- 3D Acceleration
- Tyre load (per tyre)
- Tyre slip angle (per tyre)
- Tyre slip ratio (per tyre) <- it's not documented but seems quite obvious
- track progress
- surface incline <- useless for us, surce is always flat
- orientation (relative to what ?)
- upcoming course point <- similar to what's already done (waypoints) but with temporal distance (6s lookahead)
- barrier flag <- useless for us
- "off course" flag <- we're doing it already with "off track wheel"
