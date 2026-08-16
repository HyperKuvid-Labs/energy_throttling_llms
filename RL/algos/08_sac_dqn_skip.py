# discrete sac / dqn only makes sense once configs get switched mid-session,
# so one choice actually affects the next state and gamma > 0 means something.
# the sweep we have is one-shot: launch a server, measure a config, tear it
# down, nothing carries over between rows. training either of these here would
# just be fitted-q with extra steps, no bootstrap target to speak of, so
# there's nothing real for the "sequential" part of these algos to do.
#
# would need a genuinely different data collection pass first -- a live session
# that keeps one server up and hot-swaps speculative params between requests,
# logging (state, action, reward, next_state) transitions instead of
# independent rows. not something this table's training-time estimate can
# shortcut, it's hours of new sweep time before there's anything to train on.

print("skipping sac/dqn -- no transition data exists yet, see comment above for what's needed first")
