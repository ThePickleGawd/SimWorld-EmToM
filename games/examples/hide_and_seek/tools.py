"""
Tools (actions) specific to Hide and Seek game.
"""

from games.agent_config import AgentTool


# Common tools for all agents
MOVE_TOOL = AgentTool(
    name="move",
    description="Move in a direction for a specified duration",
    parameters={
        "direction": "forward, backward, left, or right",
        "duration": "seconds to move (1-5)",
    }
)

TURN_TOOL = AgentTool(
    name="turn",
    description="Turn to face a different direction",
    parameters={
        "angle": "degrees to turn (0-180)",
        "direction": "left or right",
    }
)

LOOK_AROUND_TOOL = AgentTool(
    name="look_around",
    description="Look around to observe the environment (takes one step)",
    parameters={}
)

WAIT_TOOL = AgentTool(
    name="wait",
    description="Wait in place for a duration",
    parameters={
        "duration": "seconds to wait (1-5)",
    }
)


# Hider-specific tools
HIDE_TOOL = AgentTool(
    name="hide",
    description="Attempt to hide behind/inside a nearby object or structure",
    parameters={
        "target": "name or description of object to hide behind",
    }
)

PEEK_TOOL = AgentTool(
    name="peek",
    description="Carefully peek out from hiding spot to check surroundings",
    parameters={}
)


# Seeker-specific tools
SEARCH_TOOL = AgentTool(
    name="search",
    description="Actively search the immediate area for hiders",
    parameters={}
)

TAG_TOOL = AgentTool(
    name="tag",
    description="Attempt to tag a hider (must be within 2 meters)",
    parameters={
        "target": "ID of the hider to tag",
    }
)

CALL_OUT_TOOL = AgentTool(
    name="call_out",
    description="Call out to other seekers about a potential hider location",
    parameters={
        "message": "what to communicate to teammates",
    }
)


# Tool sets for each role
HIDER_TOOLS = [
    MOVE_TOOL,
    TURN_TOOL,
    LOOK_AROUND_TOOL,
    WAIT_TOOL,
    HIDE_TOOL,
    PEEK_TOOL,
]

SEEKER_TOOLS = [
    MOVE_TOOL,
    TURN_TOOL,
    LOOK_AROUND_TOOL,
    WAIT_TOOL,
    SEARCH_TOOL,
    TAG_TOOL,
    CALL_OUT_TOOL,
]
