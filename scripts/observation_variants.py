"""Single-field observation ablation; historical behavior remains the default."""
VARIANTS = ('historical_status', 'actual_tool_name_v1')


def observation_tool_name(observation, previous_tool_name, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown observation variant')
    if variant == 'historical_status':
        return str(observation.get('status', 'unknown'))
    if not previous_tool_name:
        # Parse failures have no tool name; retain historical representation.
        return str(observation.get('status', 'unknown'))
    return previous_tool_name
