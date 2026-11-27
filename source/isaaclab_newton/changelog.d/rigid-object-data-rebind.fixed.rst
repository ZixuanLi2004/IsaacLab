* Refreshed cached rigid object poses and velocities and restarted acceleration history when a simulation reset replaces
  the Newton state. Retained data containers now reject physics reads after their source model is replaced.
* Preserved fixed rigid bodies' zero velocity shapes when binding a new state.
