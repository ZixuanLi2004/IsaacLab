* Rejected selections containing multiple rigid bodies per world before initializing the data buffers.
* Fixed partial rigid object resets clearing pending wrenches in unselected environments when using a mask.
* Rebased the selected floating bodies' acceleration history on their new center-of-mass velocities
  after explicit velocity writes, including link-frame writes.
