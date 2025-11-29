import logging
import json
from pathlib import Path
from typing import Annotated
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class GameMasterAgent(Agent):
    def __init__(self, universe: str = "fantasy") -> None:
        self.universe = universe
        self.session_history = []
        
        # Define universe-specific prompts
        universes = {
            "fantasy": {
                "setting": "a high-fantasy realm with ancient forests, sprawling kingdoms, and forgotten magic",
                "tone": "dramatic and immersive, with vivid descriptions of magical encounters",
                "context": """You are the Game Master of an epic fantasy adventure. The world is filled with:
                - Ancient dragons guarding mountain peaks
                - Dense forests hiding mysterious creatures
                - Medieval kingdoms with court intrigue
                - Powerful magic and forgotten ruins
                
                Start the player in a tavern at the edge of a small town. Introduce hooks for adventure naturally."""
            },
            "sci-fi": {
                "setting": "a distant future with space stations, alien worlds, and advanced technology",
                "tone": "tense and exploratory, with the wonder and danger of space",
                "context": """You are the Game Master of a sci-fi survival adventure. The setting features:
                - A derelict space station slowly losing power
                - Unknown alien signals from nearby planets
                - Advanced technology mixed with malfunctioning systems
                - A crew with unclear origins
                
                Start the player waking up in a cryopod with fragmented memories. Begin in the station's main corridor."""
            },
            "horror": {
                "setting": "a creeping, supernatural world of darkness and dread",
                "tone": "spooky and unsettling, with suspenseful pacing",
                "context": """You are the Game Master of a horror adventure. The world contains:
                - Abandoned buildings with dark histories
                - Inexplicable supernatural events
                - Hints of something ancient and malevolent
                - The creeping sense that you're being watched
                
                Start the player in an old mansion on a stormy night, drawn here by mysterious circumstances."""
            },
            "cyberpunk": {
                "setting": "a neon-lit megacity controlled by megacorporations and hackers",
                "tone": "gritty and fast-paced, with moral ambiguity",
                "context": """You are the Game Master of a cyberpunk adventure. Navigate:
                - Towering megacities with vertical slums
                - Rogue AIs and corporate security
                - Underground hacker networks
                - Augmented humans and digital consciousness
                
                Start the player in a dingy ramen shop in the lower levels where a job offer arrives."""
            }
        }
        
        universe_config = universes.get(universe, universes["fantasy"])
        
        system_prompt = f"""You are a dynamic Game Master running an interactive {universe} adventure.

UNIVERSE & TONE:
{universe_config['context']}

CORE GM RESPONSIBILITIES:
1. Describe scenes vividly but concisely (2-3 sentences per scene)
2. Make the world feel alive by introducing NPCs, challenges, and consequences
3. Remember every detail the player mentions - locations, character names, items
4. Build tension and pacing - alternate between action, exploration, and dialogue
5. Honor player choices - your story adapts based on what they do
6. Use sound design in your descriptions (audio + voice makes it immersive)

CONVERSATION STRUCTURE:
- Describe the current scene and the player's situation
- Present 1-3 realistic options or open-ended possibilities
- Ask "What do you do?" to prompt the player's action
- React dynamically to their choice, updating the world state
- Build toward mini-arcs: discovery, conflict resolution, minor victories

TONE GUIDELINES:
- Keep language vivid but clear (you're speaking, not writing)
- Use dramatic pauses for emphasis
- Build mystery gradually - don't reveal everything at once
- Make consequences meaningful but fair
- Balance challenge with moments of triumph

Remember: You're speaking to one player via voice. Keep responses natural, conversational, and under 150 words per turn to maintain good pacing."""
        
        super().__init__(instructions=system_prompt)
    
    @function_tool
    async def log_session(
        self,
        context: RunContext,
        title: Annotated[str, "Title of the gaming session"],
        summary: Annotated[str, "Brief summary of what happened"]
    ):
        """Save a session log for record-keeping."""
        
        session_log = {
            "timestamp": datetime.now().isoformat(),
            "universe": self.universe,
            "title": title,
            "summary": summary,
            "turns": len(self.session_history)
        }
        
        # Create sessions directory if it doesn't exist
        sessions_dir = Path("game_sessions")
        sessions_dir.mkdir(exist_ok=True)
        
        # Save session log
        filename = sessions_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_log, f, indent=2)
        
        logger.info(f"Session logged: {title}")
        return f"Session saved! You completed '{title}' in {len(self.session_history)} turns."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Choose universe from room metadata or default to fantasy
    universe = "fantasy"
    try:
        if ctx.room.metadata:
            metadata = json.loads(ctx.room.metadata) if isinstance(ctx.room.metadata, str) else ctx.room.metadata
            universe = metadata.get("universe", "fantasy")
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Set up voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with our Game Master agent
    await session.start(
        agent=GameMasterAgent(universe=universe),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))